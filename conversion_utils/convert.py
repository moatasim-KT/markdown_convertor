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
from .markdown_assembler import assemble_markdown_from_elements, wrap_math_expressions
from .checkpoint import CheckpointManager

# Configure logging
logger = logging.getLogger(__name__)

def setup_output_dirs(output_md_path: str, image_output_dir: str) -> tuple[str, str]:
    """Create necessary output directories and return normalized paths."""
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_md_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Create images directory
    os.makedirs(image_output_dir, exist_ok=True)
    
    return output_md_path, image_output_dir

def process_chunk(chunk: List[Dict[str, Any]], image_output_dir: str) -> str:
    """Process a single chunk of elements into markdown text."""
    chunk_text = []
    for el in chunk:
        if el["type"] == "text":
            chunk_text.append(wrap_math_expressions(el["content"]))
        elif el["type"] == "image":
            rel_path = os.path.relpath(el["content"], os.path.dirname(image_output_dir))
            chunk_text.append(f"[IMAGE: {rel_path}]")
    return "\n\n".join(chunk_text)

def _save_partial_markdown(
    output_path: str,
    chunked_elements: List[List[Dict[str, Any]]],
    image_output_dir: str,
    processed_chunks: int
) -> None:
    """Save the current state of processed chunks to the output file.
    
    Args:
        output_path: Path to the output Markdown file
        chunked_elements: All chunk elements (both processed and unprocessed)
        image_output_dir: Directory for extracted images
        processed_chunks: Number of chunks that have been processed so far
    """
    try:
        # Only include processed chunks in the output
        processed_chunks_list = chunked_elements[:processed_chunks]
        
        # Generate markdown for processed elements
        markdown = assemble_markdown_from_elements(processed_chunks_list, image_output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write to a temporary file first
        temp_path = f"{output_path}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        # Atomically replace the output file
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)
        
        logger.debug(f"Saved partial markdown with {processed_chunks} chunks to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save partial markdown: {str(e)}", exc_info=True)
        # Don't raise, as we want to continue processing even if saving fails

def _process_chunks_in_batches(
    chunk_texts: List[str],
    chunked_elements: List[List[Dict[str, Any]]],
    image_output_dir: str,
    checkpoint_manager: CheckpointManager,
    output_md_path: str,
    batch_size: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    start_chunk: int = 0,
    **conversion_kwargs
) -> None:
    """Process chunks in batches with checkpointing and error handling.
    
    Args:
        chunk_texts: List of chunk texts to process
        chunked_elements: Original chunk elements to update with processed markdown
        image_output_dir: Directory for extracted images
        checkpoint_manager: Checkpoint manager instance
        output_md_path: Path to save the output Markdown file
        batch_size: Number of chunks to process in each batch
        progress_callback: Optional callback for progress updates
        start_chunk: Chunk index to start processing from (for resuming)
        **conversion_kwargs: Additional arguments for convert_chunks_to_markdown
    
    Raises:
        RuntimeError: If processing fails and cannot be recovered
    """
    # Validate inputs
    if not chunk_texts or not chunked_elements:
        logger.warning("No chunks to process")
        return
        
    if len(chunk_texts) != len(chunked_elements):
        raise ValueError("Mismatch between chunk_texts and chunked_elements lengths")
        
    total_chunks = len(chunk_texts)
    processed_chunks = start_chunk
    
    # Filter only the arguments that convert_chunks_to_markdown accepts
    allowed_args = {
        'max_retries', 'backoff_factor', 'timeout', 'parallel', 'max_workers'
    }
    convert_kwargs = {
        k: v for k, v in conversion_kwargs.items() if k in allowed_args
    }
    """Process chunks in batches with checkpointing.
    
    Args:
        chunk_texts: List of chunk texts to process
        chunked_elements: Original chunk elements to update with processed markdown
        image_output_dir: Directory for extracted images
        checkpoint_manager: Checkpoint manager instance
        batch_size: Number of chunks to process in each batch
        **conversion_kwargs: Additional arguments for convert_chunks_to_markdown
    """
    total_chunks = len(chunk_texts)
    processed_chunks = 0
    
    # Create progress bar
    with tqdm(
        total=total_chunks,
        desc="Converting",
        unit="chunk",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    ) as pbar:
        # Initialize retry counter
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        try:
            # Process chunks in batches
            for i in range(0, total_chunks, batch_size):
                batch_start = i
                batch_end = min(i + batch_size, total_chunks)
                
                # Update progress bar description with batch info
                pbar.set_description(f"Batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
                
                # Log detailed batch info
                logger.info(
                    f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} "
                    f"(chunks {batch_start+1}-{batch_end} of {total_chunks})"
                )
                
                try:
                    # Process the current batch
                    markdown_batch = convert_chunks_to_markdown(
                        chunks=chunk_texts[batch_start:batch_end],
                        **convert_kwargs
                    )
                    consecutive_failures = 0  # Reset counter on success
                    
                except (RateLimitError, APIClientError) as e:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Too many consecutive failures ({consecutive_failures}): {str(e)}")
                        logger.info("Checkpoint saved. You can resume the conversion later.")
                        raise
                        
                    # Exponential backoff
                    backoff_time = (2 ** (consecutive_failures - 1)) * 5  # 5, 10, 20, ... seconds
                    logger.warning(
                        f"API error (attempt {consecutive_failures}/{max_consecutive_failures}). "
                        f"Retrying in {backoff_time} seconds..."
                    )
                    time.sleep(backoff_time)
                    continue  # Retry the same batch
                    
                except Exception as e:
                    logger.error(f"Unexpected error processing batch: {str(e)}", exc_info=True)
                    # Mark this chunk as failed but continue with the next one
                    markdown_batch = ["[Error: Failed to process chunk]"] * (batch_end - batch_start)
                    # Still update the elements to include the error message
                
                # Update chunked_elements with processed markdown
                for j, (chunk, markdown) in enumerate(zip(
                    chunked_elements[batch_start:batch_end], 
                    markdown_batch
                )):
                    # Update the first text element in each chunk with the processed markdown
                    for el in chunk:
                        if el["type"] == "text":
                            el["content"] = markdown
                            break
                
                # Update progress
                processed_chunks = batch_end
                chunks_processed = processed_chunks - pbar.n
                
                # Update progress bar
                pbar.update(chunks_processed)
                
                # Update progress details
                elapsed_time = pbar.format_dict["elapsed"]
                rate = pbar.n / elapsed_time if elapsed_time > 0 else 0
                pbar.set_postfix({
                    'rate': f"{rate:.1f} chunks/s",
                })
                
                # Save the processed chunks to the output file
                try:
                    _save_partial_markdown(
                        output_path=output_md_path,
                        chunked_elements=chunked_elements,
                        image_output_dir=image_output_dir,
                        processed_chunks=processed_chunks
                    )
                except Exception as e:
                    logger.warning(f"Failed to update output file: {str(e)}")
                
                # Save checkpoint after each successful batch
                try:
                    checkpoint = checkpoint_manager.create_checkpoint(
                        pdf_path=conversion_kwargs.get('pdf_path', output_md_path + '.pdf'),
                        output_md_path=output_md_path,
                        image_output_dir=image_output_dir,
                        chunks=chunked_elements,
                        processed_chunks=processed_chunks,
                        total_chunks=total_chunks,
                        completed=(processed_chunks >= total_chunks)
                    )
                    
                    # Save partial markdown to output file
                    _save_partial_markdown(
                        output_path=output_md_path,
                        chunked_elements=chunked_elements,
                        image_output_dir=image_output_dir,
                        processed_chunks=processed_chunks
                    )
                    
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}", exc_info=True)
                    # Continue processing even if checkpoint fails
                    if processed_chunks >= total_chunks:
                        break  # Exit if we've processed all chunks
                
                checkpoint_manager.save_checkpoint(checkpoint)
                
                # Update progress callback if provided
                if progress_callback:
                    progress_callback(processed_chunks, total_chunks)
                
                logger.debug(f"Completed {processed_chunks}/{total_chunks} chunks")
                
        except Exception as e:
            logger.critical(f"Fatal error in batch processing: {str(e)}", exc_info=True)
            logger.info("Checkpoint saved. You can resume the conversion later.")
            # Try to save one last checkpoint before exiting
            try:
                checkpoint_manager.create_checkpoint(
                    pdf_path=conversion_kwargs.get('pdf_path', output_md_path + '.pdf'),
                    output_md_path=output_md_path,
                    image_output_dir=image_output_dir,
                    chunks=chunked_elements,
                    processed_chunks=processed_chunks,
                    total_chunks=total_chunks,
                    completed=False
                )
            except Exception as save_error:
                logger.error(f"Failed to save final checkpoint: {save_error}", exc_info=True)
                raise RuntimeError(f"Failed to process batch {batch_start}-{batch_end}: {str(e)}") from e

def convert_pdf_to_markdown(
    pdf_path: str, 
    output_md_path: str, 
    image_output_dir: str,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
    timeout: int = 30,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    resume: bool = True,
    batch_size: int = 5
) -> None:
    """
    Convert a PDF file to Markdown with inline images using Groq API.
    
    Args:
        pdf_path: Path to the input PDF file
        output_md_path: Path to save the output Markdown file
        image_output_dir: Directory to save extracted images
        max_retries: Maximum number of retry attempts for API calls
        backoff_factor: Base multiplier for exponential backoff between retries
        timeout: Request timeout in seconds
        parallel: Whether to process chunks in parallel
        max_workers: Maximum number of parallel workers (if parallel=True)
        resume: Whether to resume from previous checkpoint if available
        batch_size: Number of chunks to process in each batch
        
    Raises:
        FileNotFoundError: If the input PDF does not exist
        ValueError: If no content is found in the PDF
        APIClientError: For API-related errors
        Exception: For other unexpected errors
    """
    start_time = datetime.now()
    logger.info(f"Starting PDF to Markdown conversion for: {pdf_path}")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(output_md_path)
    
    # Check for existing checkpoint if resuming is enabled
    total_chunks = 0
    processed_chunks = 0
    if resume:
        can_resume, checkpoint, message = checkpoint_manager.is_resumable(pdf_path)
        if can_resume and checkpoint:
            logger.info(f"Resuming from previous checkpoint: {message}")
            # Use the checkpoint data
            all_chunks = checkpoint.chunks
            processed_chunks = checkpoint.metadata["processed_chunks"]
            total_chunks = checkpoint.metadata["total_chunks"]
            
            logger.info(f"Resuming from chunk {processed_chunks + 1} of {total_chunks}")
            
            # Use all chunks for processing, but track where to start
            chunked_elements = all_chunks
        else:
            logger.info(f"Starting new conversion: {message}")
            chunked_elements = None
    else:
        logger.info("Starting new conversion (resume disabled)")
        chunked_elements = None
    
    try:
        # Validate input
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Setup output directories
        output_md_path, image_output_dir = setup_output_dirs(output_md_path, image_output_dir)
        
        # If we're not resuming, extract and chunk elements
        if not chunked_elements:
            # 1. Extract elements (text blocks and images)
            logger.info("Extracting elements from PDF...")
            elements = extract_elements(pdf_path, image_output_dir)
            if not elements:
                raise ValueError("No text or images found in the PDF.")
            logger.info(f"Extracted {len(elements)} elements from PDF")
            
            # 2. Chunk elements
            logger.info("Chunking elements...")
            chunked_elements = chunk_elements(elements)
            if not chunked_elements:
                raise ValueError("Failed to chunk PDF content.")
            logger.info(f"Created {len(chunked_elements)} chunks for processing")
        
        # 3. Process chunks to Markdown in batches
        logger.info("Converting chunks to Markdown...")
        
        # Convert chunks to text first
        chunk_texts = [process_chunk(chunk, image_output_dir) for chunk in chunked_elements]
        total_chunks = len(chunk_texts)
        
        # Process chunks in batches with checkpointing and incremental saving
        _process_chunks_in_batches(
            chunk_texts=chunk_texts,
            chunked_elements=chunked_elements,
            image_output_dir=image_output_dir,
            checkpoint_manager=checkpoint_manager,
            output_md_path=output_md_path,
            batch_size=batch_size,
            pdf_path=pdf_path,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            timeout=timeout,
            parallel=parallel,
            max_workers=max_workers,
            progress_callback=lambda current, total: None,  # No-op callback for now
            start_chunk=processed_chunks
        )
        
        # The final markdown is already saved by _process_chunks_in_batches
        # We just need to ensure the last version is complete and properly formatted
        logger.info("Finalizing Markdown...")
        markdown = assemble_markdown_from_elements(chunked_elements, image_output_dir)
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        # Mark conversion as complete in the checkpoint
        checkpoint = checkpoint_manager.create_checkpoint(
            pdf_path=pdf_path,
            output_md_path=output_md_path,
            image_output_dir=image_output_dir,
            chunks=chunked_elements,
            processed_chunks=total_chunks,
            total_chunks=total_chunks,
            completed=True
        )
        checkpoint_manager.save_checkpoint(checkpoint)
        
        # Clean up checkpoint on successful completion
        checkpoint_manager.cleanup()
            
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Successfully converted PDF to Markdown in {elapsed:.2f} seconds")
        logger.info(f"Markdown saved to: {output_md_path}")
        logger.info(f"Images saved to: {image_output_dir}")
            
    except (RateLimitError, APIClientError) as e:
        logger.error(f"API error during conversion: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during PDF to Markdown conversion: {str(e)}")
        # Clean up partially created files on error
        if os.path.exists(output_md_path):
            try:
                os.remove(output_md_path)
                logger.warning(f"Removed incomplete output file: {output_md_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up output file: {cleanup_error}")
        raise
