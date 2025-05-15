from typing import List, Dict, Any

def chunk_elements(elements: List[Dict[str, Any]], max_chars: int = 2000) -> List[List[Dict[str, Any]]]:
    """
    Groups elements (text blocks and images) into chunks, keeping images and their surrounding text together.
    Each chunk is a list of elements, not exceeding max_chars (approximate, by text length).
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for el in elements:
        if el["type"] == "text":
            el_len = len(el["content"])
        else:
            el_len = 50  # Arbitrary small value for images
        if current_length + el_len > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.append(el)
        current_length += el_len
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
