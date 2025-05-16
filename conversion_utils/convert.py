import os
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from .pdf_parser import extract_elements
from .chunker import chunk_elements
from .groq_client import convert_chunks_to_markdown, APIClientError, RateLimitError
from .markdown_assembler import assemble_markdown_from_processed_chunks # Updated import
from .math_detector import process_math # Import process_math directly
from .checkpoint import CheckpointManager

# Configure logging
logger = logging.getLogger(__name__)

def setup_output_dirs(output_md_path: str, image_output_dir: str) -> tuple[str, str]:
    """Create necessary output directories and return normalized paths."""
    output_dir = os.path.dirname(os.path.abspath(output_md_path))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    return output_md_path, image_output_dir

def process_chunk_to_llm_input(chunk: List[Dict[str, Any]], image_output_dir: str) -> str:
    """
    Process a single chunk of elements into a single string suitable for LLM input.
    Text elements are processed with math detection and wrapping.
    Image elements are converted to placeholders.
    """
    chunk_text_parts = []
    for el in chunk:
        if el["type"] == "text":
            # Apply math processing (cleaning + wrapping) before sending to LLM
            chunk_text_parts.append(process_math(el["content"]))
        elif el["type"] == "image":
            abs_image_output_dir = os.path.abspath(image_output_dir)
            try:
                if not os.path.isabs(el["content"]):
                    # Fallback if image path is not absolute, try to make it relative to image_output_dir parent
                    logger.warning(f"Image path {el['content']} is not absolute. Attempting relative path from parent of {abs_image_output_dir}.")
                    # This assumes el['content'] is like 'image.png' and should be under abs_image_output_dir
                    img_file_name = os.path.basename(el["content"])
                    abs_img_path = os.path.join(abs_image_output_dir, img_file_name)
                    rel_path = os.path.relpath(abs_img_path, abs_image_output_dir)

                else:
                    rel_path = os.path.relpath(el["content"], abs_image_output_dir)
            except ValueError: 
                # This can happen if paths are on different drives (Windows) or other issues
                logger.warning(f"Could not create relative path for {el['content']} against {abs_image_output_dir}. Using basename.")
                rel_path = os.path.basename(el["content"]) 
            chunk_text_parts.append(f"[IMAGE: {rel_path}]") # LLM is prompted to handle this
    return "\n\n".join(chunk_text_parts)

def _save_partial_markdown(
    output_path: str,
    processed_markdown_list: List[Optional[str]], # Takes list of processed strings
    processed_chunks_count: int # Number of chunks in the list that are considered processed
) -> None:
    """Save the current state of processed markdown chunks to the output file."""
    try:
        # Include only the processed chunks up to processed_chunks_count
        # assemble_markdown_from_processed_chunks will filter Nones and handle error placeholders
        chunks_to_save = processed_markdown_list[:processed_chunks_count]
        
        markdown = assemble_markdown_from_processed_chunks(chunks_to_save)
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        temp_path = f"{output_path}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)
        
        # Use length of chunks_to_save as it reflects what was actually passed to assembler
        logger.debug(f"Saved partial markdown with {len(chunks_to_save)} fully processed or error-marked chunks to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save partial markdown: {str(e)}", exc_info=True)

def _process_chunks_in_batches(
    llm_input_texts: List[str], 
    final_markdown_outputs: List[Optional[str]], 
    original_chunked_elements: List[List[Dict[str, Any]]], 
    image_output_dir: str, 
    checkpoint_manager: CheckpointManager,
    output_md_path: str,
    batch_size: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    start_chunk_idx: int = 0, 
    pdf_path_for_checkpoint: str = "",
    **conversion_kwargs
) -> int:
    """
    Process text chunks in batches, sends them to LLM, and stores the output in final_markdown_outputs.
    Returns the total count of chunks that have been processed (attempted or successful).
    """
    if not llm_input_texts:
        logger.warning("No LLM input texts to process.")
        return 0
        
    total_chunks = len(llm_input_texts)
    # This count reflects how many entries in final_markdown_outputs are filled (attempted/succeeded)
    processed_chunks_count = start_chunk_idx 
    
    # Filter only the arguments that convert_chunks_to_markdown accepts
    allowed_args = {
        'max_retries', 'backoff_factor', 'timeout', 'parallel', 'max_workers'
    }
    llm_convert_kwargs = {
        k: v for k, v in conversion_kwargs.items() if k in allowed_args
    }
    
    # tqdm progress bar setup
    with tqdm(
        initial=processed_chunks_count, # Start progress from where we left off
        total=total_chunks,
        desc="Converting",
        unit="chunk",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    ) as pbar:
        max_api_failures_per_batch = 3 # Max retries for a batch failing due to API issues

        # Iterate from the start_chunk_idx, effectively skipping already processed chunks if resuming
        for current_batch_start_idx in range(start_chunk_idx, total_chunks, batch_size):
            current_batch_end_idx = min(current_batch_start_idx + batch_size, total_chunks)
            
            # Slice the llm_input_texts for the current batch
            batch_llm_inputs_to_process = llm_input_texts[current_batch_start_idx:current_batch_end_idx]
            
            if not batch_llm_inputs_to_process: # Should not happen with correct loop logic
                continue

            pbar.set_description(
                f"Batch {(current_batch_start_idx // batch_size) + 1}/{(total_chunks + batch_size - 1) // batch_size}"
            )
            logger.info(
                f"Processing batch {(current_batch_start_idx // batch_size) + 1} "
                f"(chunks {current_batch_start_idx + 1}-{current_batch_end_idx} of {total_chunks})"
            )

            api_batch_attempt = 0
            batch_processed_successfully = False
            while api_batch_attempt < max_api_failures_per_batch:
                try:
                    # Call LLM for the current batch
                    markdown_batch_llm_output = convert_chunks_to_markdown(
                        chunks=batch_llm_inputs_to_process,
                        **llm_convert_kwargs 
                    )
                    
                    # Store results in the main list
                    for i, markdown_content in enumerate(markdown_batch_llm_output):
                        output_storage_idx = current_batch_start_idx + i
                        if output_storage_idx < total_chunks: # Boundary check
                            final_markdown_outputs[output_storage_idx] = markdown_content
                        else:
                            logger.error(f"Index {output_storage_idx} out of bounds for final_markdown_outputs (size {total_chunks})")
                    
                    # Update overall processed count and progress bar
                    # pbar.n is the current count in progress bar, pbar.update increases it
                    # We want to update by the number of items processed in this successful batch
                    num_processed_in_batch = len(batch_llm_inputs_to_process)
                    pbar.update(num_processed_in_batch)
                    processed_chunks_count = current_batch_end_idx # Update to the end of the current batch
                    batch_processed_successfully = True
                    break # Exit retry loop for this batch

                except (RateLimitError, APIClientError) as e:
                    api_batch_attempt += 1
                    logger.warning(
                        f"API error on batch {current_batch_start_idx+1}-{current_batch_end_idx} (Attempt {api_batch_attempt}/{max_api_failures_per_batch}): {str(e)}"
                    )
                    if api_batch_attempt >= max_api_failures_per_batch:
                        logger.error(
                            f"Max API retries for batch {current_batch_start_idx+1}-{current_batch_end_idx} reached. Marking items as errored."
                        )
                        for i in range(len(batch_llm_inputs_to_process)):
                            final_markdown_outputs[current_batch_start_idx + i] = f"[Error: API processing failed for this chunk - {str(e)}]"
                        pbar.update(len(batch_llm_inputs_to_process)) # Still update progress
                        processed_chunks_count = current_batch_end_idx 
                        break # Exit retry loop for this batch
                    
                    backoff_duration = (2 ** (api_batch_attempt -1)) * 5 
                    logger.info(f"Retrying current batch in {backoff_duration} seconds...")
                    time.sleep(backoff_duration)
                
                except Exception as e: # Catch other unexpected errors during batch LLM processing
                    logger.error(f"Unexpected critical error processing batch {current_batch_start_idx+1}-{current_batch_end_idx}: {str(e)}", exc_info=True)
                    for i in range(len(batch_llm_inputs_to_process)):
                        final_markdown_outputs[current_batch_start_idx + i] = f"[Error: Unexpected critical error during processing - {str(e)}]"
                    pbar.update(len(batch_llm_inputs_to_process))
                    processed_chunks_count = current_batch_end_idx
                    # This is a more severe error, re-raise to stop all processing.
                    raise RuntimeError(f"Fatal error processing batch: {str(e)}") from e
            
            # After each batch is processed (or failed max retries)
            if not batch_processed_successfully and api_batch_attempt >= max_api_failures_per_batch:
                 logger.error(f"Batch {current_batch_start_idx+1}-{current_batch_end_idx} ultimately failed after max API retries.")
                 # Errors are already marked in final_markdown_outputs
                
            # Update progress details
            elapsed_time = pbar.format_dict["elapsed"]
            current_rate = pbar.n / elapsed_time if elapsed_time > 0 else 0
            pbar.set_postfix_str(f"{current_rate:.1f} chunks/s")

            # Save partial markdown and checkpoint after each batch
            _save_partial_markdown(
                output_path=output_md_path,
                processed_markdown_list=final_markdown_outputs,
                processed_chunks_count=processed_chunks_count 
            )
            
            # Create checkpoint using the CheckpointManager's create_checkpoint method
            checkpoint_manager.create_checkpoint(
                pdf_path=pdf_path_for_checkpoint,
                output_md_path=output_md_path,
                image_output_dir=image_output_dir,
                chunks=original_chunked_elements,
                processed_chunks=processed_chunks_count,
                total_chunks=total_chunks
            )
            
            if progress_callback:
                progress_callback(processed_chunks_count, total_chunks)
            
            logger.debug(f"End of batch. Processed count: {processed_chunks_count}/{total_chunks} chunks.")

    return processed_chunks_count # Return the total number of chunks processed/attempted


def convert_pdf_to_markdown(
    pdf_path: str, 
    output_md_path: str, 
    image_output_dir: str,
    batch_size: int = 10,
    max_retries: int = 10,
    backoff_factor: float = 1.5,
    timeout: int = 60,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    checkpoint_interval: int = 100,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> str:
    """
    Convert a PDF file to Markdown, processing content through an LLM.
    
    Args:
        pdf_path: Path to the input PDF file
        output_md_path: Path to save the output Markdown file
        image_output_dir: Directory to save extracted images
        batch_size: Number of chunks to process in each batch
        max_retries: Maximum number of retry attempts for API calls
        backoff_factor: Base multiplier for exponential backoff between retries
        timeout: Request timeout in seconds
        parallel: Whether to process chunks in parallel
        max_workers: Maximum number of parallel workers (if parallel=True)
        checkpoint_manager: Optional checkpoint manager for resuming
        checkpoint_interval: Number of chunks between checkpoints
        progress_callback: Optional callback for progress updates
        
    Raises:
        FileNotFoundError: If the input PDF is not found.
        ValueError: If input PDF is invalid or no content is found.
        APIClientError: For unrecoverable API-related errors.
        Exception: For other unexpected errors during processing.
    """
    start_time = datetime.now()
    logger.info(f"Starting PDF to Markdown conversion for: {pdf_path}")

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Input PDF file not found: {pdf_path}")

    output_md_path, image_output_dir = setup_output_dirs(output_md_path, image_output_dir)
    # Initialize checkpoint manager with output_md_path
    checkpoint_manager = CheckpointManager(output_md_path=output_md_path) 

    original_chunked_elements: List[List[Dict[str, Any]]]
    final_markdown_outputs: List[Optional[str]]
    llm_input_texts: List[str]
    processed_chunks_count = 0
    total_chunks = 0
    
    loaded_checkpoint_data = None
    checkpoint = checkpoint_manager.load_checkpoint(pdf_path)
    if checkpoint:
        loaded_checkpoint_data = {
            "pdf_path": pdf_path,
            "original_chunked_elements": checkpoint.chunks,
            "final_markdown_outputs": [None] * len(checkpoint.chunks),  # Initialize with None
            "processed_chunks_count": checkpoint.metadata.processed_chunks,
            "total_chunks": checkpoint.metadata.total_chunks
        }
    else:
        loaded_checkpoint_data = None

    if loaded_checkpoint_data and loaded_checkpoint_data["pdf_path"] == pdf_path:
        logger.info(f"Resuming from checkpoint for {pdf_path}")
        original_chunked_elements = loaded_checkpoint_data["original_chunked_elements"]
        final_markdown_outputs = loaded_checkpoint_data["final_markdown_outputs"]
        processed_chunks_count = loaded_checkpoint_data["processed_chunks_count"]
        total_chunks = loaded_checkpoint_data["total_chunks"]
        
        # Ensure final_markdown_outputs has the correct length if checkpoint was saved mid-batch
        # or if the checkpoint format was from an older version.
        if len(final_markdown_outputs) != total_chunks:
            logger.warning(
                f"Checkpoint data for final_markdown_outputs has length {len(final_markdown_outputs)}, "
                f"but total_chunks is {total_chunks}. Re-initializing and attempting to copy."
            )
            correct_size_outputs = [None] * total_chunks
            for i in range(min(len(final_markdown_outputs), total_chunks)):
                correct_size_outputs[i] = final_markdown_outputs[i]
            final_markdown_outputs = correct_size_outputs
            # Adjust processed_chunks_count if it's now out of bounds due to array resizing
            processed_chunks_count = min(processed_chunks_count, sum(1 for x in final_markdown_outputs if x is not None))


        llm_input_texts = [process_chunk_to_llm_input(chunk, image_output_dir) for chunk in original_chunked_elements]
        
        # Check if already completed
        if processed_chunks_count >= total_chunks and all(final_markdown_outputs[i] is not None for i in range(total_chunks)):
            logger.info("Conversion was already completed according to checkpoint.")
        else:
             # If not fully complete, log where we are resuming from
             logger.info(f"Resuming from chunk {processed_chunks_count + 1} of {total_chunks}")

    else: # Conditions for starting a new conversion
        if loaded_checkpoint_data:
            logger.info(f"Resuming conversion from checkpoint for {pdf_path}")
        else:
            logger.info("Starting new conversion")

        logger.info("Extracting elements from PDF...")
        elements = extract_elements(pdf_path, image_output_dir)
        if not elements:
            raise ValueError("No text or images found in the PDF.")
        logger.info(f"Extracted {len(elements)} elements.")

        logger.info("Chunking elements...")
        original_chunked_elements = chunk_elements(elements)
        if not original_chunked_elements:
            raise ValueError("Failed to chunk PDF content.")
        total_chunks = len(original_chunked_elements)
        logger.info(f"Created {total_chunks} chunks for processing.")

        llm_input_texts = [process_chunk_to_llm_input(chunk, image_output_dir) for chunk in original_chunked_elements]
        final_markdown_outputs = [None] * total_chunks # Initialize with Nones
        processed_chunks_count = 0 # Start from the beginning

    # Core processing loop if not already completed
    if processed_chunks_count < total_chunks or not all(final_markdown_outputs[i] is not None for i in range(total_chunks)):
        logger.info("Processing chunks through LLM...")
        logger.info(f"Starting batch processing with {len(llm_input_texts)} chunks")
        try:
            processed_chunks_count = _process_chunks_in_batches(
                llm_input_texts=llm_input_texts,
                final_markdown_outputs=final_markdown_outputs, # Pass this list to be populated
                original_chunked_elements=original_chunked_elements, # For checkpointing
                image_output_dir=image_output_dir, # For checkpointing
                checkpoint_manager=checkpoint_manager,
                output_md_path=output_md_path,
                batch_size=batch_size,
                progress_callback=lambda current, total: None, # Placeholder, can be expanded
                start_chunk_idx=processed_chunks_count, # Where to start processing
                pdf_path_for_checkpoint=pdf_path, 
                # Pass through relevant conversion arguments
                max_retries=max_retries,       
                backoff_factor=backoff_factor,
                timeout=timeout,
                parallel=parallel,
                max_workers=max_workers
            )
            logger.info(f"Completed processing {processed_chunks_count} chunks")
        except RuntimeError as e: 
            logger.error(f"Critical error during batch processing: {e}. Conversion halted.")
            # Checkpoint should have been saved by _process_chunks_in_batches
            raise # Re-raise the error to stop execution and indicate failure
            
    logger.info("Assembling final Markdown document...")
    # Ensure all items in final_markdown_outputs are strings for assembly
    # If any are None at this stage, it means they were skipped or failed critically
    # The assembler already handles None by inserting an error placeholder.
            
    final_markdown = assemble_markdown_from_processed_chunks(final_markdown_outputs)
    
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(final_markdown)

    logger.info("Finalizing checkpoint as complete.")
    final_checkpoint_data = {
        "pdf_path": pdf_path,
        "output_md_path": output_md_path,
        "image_output_dir": image_output_dir,
        "original_chunked_elements": original_chunked_elements,
        "final_markdown_outputs": final_markdown_outputs,
        "processed_chunks_count": total_chunks, # Mark all as processed
        "total_chunks": total_chunks,
        # "completed": True # This can be inferred if processed_chunks_count == total_chunks
    }
    checkpoint_manager.save_checkpoint_data(final_checkpoint_data)
    checkpoint_manager.cleanup() # Remove checkpoint file on successful completion

    elapsed_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Successfully converted PDF to Markdown in {elapsed_time:.2f} seconds.")
    logger.info(f"Output Markdown saved to: {output_md_path}")
    logger.info(f"Images saved to: {image_output_dir}")

# Example usage (if this file were to be run directly)
if __name__ == '__main__':
    # This is a placeholder for potential command-line integration or testing
    # For actual use, import and call convert_pdf_to_markdown from another script
    
    # Configure basic logging for testing
    logging.basicConfig(
        level=logging.INFO, # Use INFO or DEBUG for more verbosity during testing
        format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
    )
    
    # Replace with actual paths for testing
    # test_pdf_path = "path/to/your/test.pdf"  # <--- SET YOUR TEST PDF PATH HERE
    # test_md_output = "path/to/your/output.md" # <--- SET YOUR TEST MD OUTPUT PATH HERE
    # test_image_dir = "path/to/your/images"    # <--- SET YOUR TEST IMAGE DIR HERE
    
    # # Ensure the test PDF exists before trying to convert
    # if test_pdf_path != "path/to/your/test.pdf" and os.path.exists(test_pdf_path):
    #     logger.info(f"--- Starting test conversion for {test_pdf_path} ---")
    #     try:
    #         convert_pdf_to_markdown(
    #             pdf_path=test_pdf_path,
    #             output_md_path=test_md_output,
    #             image_output_dir=test_image_dir,
    #             resume=True, 
    #             batch_size=2, # Small batch size for testing
    #             parallel=False # Easier to debug sequentially first
    #         )
    #         logger.info(f"--- Test conversion finished for {test_pdf_path} ---")
    #     except Exception as e:
    #         logger.error(f"Test conversion failed: {e}", exc_info=True)
    # else:
    #     if test_pdf_path == "path/to/your/test.pdf":
    #         logger.warning("Test PDF path not set. Skipping example run.")
    #     else:
    #         logger.warning(f"Test PDF not found at {test_pdf_path}, skipping example run.")
    pass
