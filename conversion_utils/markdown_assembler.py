from typing import List, Optional

def assemble_markdown_from_processed_chunks(
    processed_markdown_chunks: List[Optional[str]],
) -> str:
    """
    Combines LLM-processed markdown chunks into a single Markdown string.
    Assumes each string in the list is a fully processed markdown chunk.
    
    Args:
        processed_markdown_chunks: List of strings, where each string is 
                                   the LLM-processed markdown for a chunk.
                                   May contain None for unprocessed or failed chunks.
        
    Returns:
        str: Combined markdown content.
    """
    # Filter out any None entries which might represent unprocessed or failed chunks
    # and ensure all parts are strings.
    valid_chunks = [
        chunk.strip() if isinstance(chunk, str) 
        else "[Error: Chunk not processed]" if chunk is None 
        else "[Error: Invalid chunk type]"
        for chunk in processed_markdown_chunks
    ]
    
    # Separate chunks with horizontal rules (or chosen separator)
    return "\n\n---\n\n".join(valid_chunks)
