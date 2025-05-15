from typing import List, Dict, Any, Optional
import os
import re
from .math_detector import process_math, detect_math

def wrap_math_expressions(text: str) -> str:
    """
    Process text to properly format math expressions using the enhanced math detector.
    
    Args:
        text: Input text potentially containing math expressions
        
    Returns:
        str: Text with properly formatted math expressions
    """
    return process_math(text)

def assemble_markdown_from_elements(
    chunked_elements: List[List[Dict[str, Any]]], 
    image_dir: str
) -> str:
    """
    Combines element-based chunks into a Markdown string, placing image references inline.
    Applies math wrapping to text blocks.
    
    Args:
        chunked_elements: List of chunks, where each chunk is a list of elements
        image_dir: Directory where images are stored
        
    Returns:
        str: Combined markdown content
    """
    md_chunks = []
    
    for chunk in chunked_elements:
        md_chunk = []
        
        for el in chunk:
            if el["type"] == "text":
                # Process text with enhanced math detection
                processed_text = process_math(el["content"])
                md_chunk.append(processed_text)
                
            elif el["type"] == "image":
                # Handle image references
                rel_path = os.path.relpath(el["content"], os.path.dirname(image_dir))
                alt_text = f"Image from page {el.get('page', '')}"
                
                # Check if this is likely a math image (based on size and aspect ratio)
                bbox = el.get("bbox", [0, 0, 100, 20])  # Default small bbox
                width = bbox[2] - bbox[0] if len(bbox) > 2 else 100
                height = bbox[3] - bbox[1] if len(bbox) > 3 else 20
                aspect_ratio = width / height if height > 0 else 1
                
                # Heuristic: Math images are often wider than they are tall
                if aspect_ratio > 2.0 or height < 100:
                    alt_text = f"Equation image from page {el.get('page', '')}"
                    
                md_chunk.append(f"![{alt_text}]({rel_path})")
        
        # Join elements within the chunk with double newlines
        if md_chunk:
            md_chunks.append("\n\n".join(md_chunk))
    
    # Separate chunks with horizontal rules
    return "\n\n---\n\n".join(md_chunks)
