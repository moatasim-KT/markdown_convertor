"""
Checkpointing system for PDF to Markdown conversion.

This module provides functionality to save and load conversion progress,
allowing the process to be resumed after interruptions.
"""

import json
import os
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, TypedDict, Union, Tuple
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

class CheckpointMetadata(TypedDict):
    """Metadata for a conversion checkpoint."""
    pdf_path: str
    pdf_hash: str
    total_chunks: int
    processed_chunks: int
    completed: bool
    output_md_path: str
    image_output_dir: str
    timestamp: str
    version: str = "1.0"

@dataclass
class Checkpoint:
    """Represents a conversion checkpoint."""
    metadata: CheckpointMetadata
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        return {
            "metadata": dict(self.metadata),
            "chunks": self.chunks
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """Create checkpoint from dictionary."""
        return cls(
            metadata=data["metadata"],
            chunks=data["chunks"]
        )

class CheckpointManager:
    """Manages checkpoints for PDF to Markdown conversion."""
    
    CHECKPOINT_EXT = ".checkpoint.json"
    
    def __init__(self, output_md_path: str):
        """Initialize checkpoint manager.
        
        Args:
            output_md_path: Path to the output Markdown file.
        """
        self.output_md_path = Path(output_md_path).resolve()
        self.checkpoint_path = self._get_checkpoint_path()
        self.checkpoint_dir = self.checkpoint_path.parent
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_checkpoint_path(self) -> Path:
        """Get the path for the checkpoint file."""
        return self.output_md_path.parent / ".checkpoints" / f"{self.output_md_path.stem}{self.CHECKPOINT_EXT}"
    
    @staticmethod
    def _calculate_file_hash(file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 hash of a file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def create_checkpoint(
        self,
        pdf_path: str,
        output_md_path: str,
        image_output_dir: str,
        chunks: List[Dict[str, Any]],
        processed_chunks: int,
        total_chunks: int,
        completed: bool = False
    ) -> Checkpoint:
        """Create a new checkpoint.
        
        Args:
            pdf_path: Path to the input PDF file.
            output_md_path: Path to the output Markdown file.
            image_output_dir: Directory for extracted images.
            chunks: List of processed chunks.
            processed_chunks: Number of chunks processed so far.
            total_chunks: Total number of chunks to process.
            completed: Whether the conversion is completed.
            
        Returns:
            The created Checkpoint object.
        """
        try:
            # Calculate PDF hash
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_hash = hashlib.sha256(f.read()).hexdigest()
            except IOError as e:
                logger.error(f"Error reading PDF file for hashing: {e}")
                pdf_hash = ""  # Use empty hash if we can't read the file
            
            # Ensure paths are absolute
            abs_pdf_path = os.path.abspath(pdf_path)
            abs_output_path = os.path.abspath(output_md_path)
            abs_image_dir = os.path.abspath(image_output_dir)
            
            # Validate chunks data
            if not isinstance(chunks, list):
                raise ValueError("chunks must be a list")
                
            # Create metadata
            metadata = {
                'pdf_path': abs_pdf_path,
                'pdf_hash': pdf_hash,
                'total_chunks': int(total_chunks),
                'processed_chunks': int(processed_chunks),
                'completed': bool(completed),
                'output_md_path': abs_output_path,
                'image_output_dir': abs_image_dir,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0'
            }
            
            # Create checkpoint with a copy of chunks to prevent modification
            checkpoint = Checkpoint(
                metadata=metadata,
                chunks=[chunk.copy() for chunk in chunks]  # Create deep copy to prevent reference issues
            )
            
            # Save to a temporary file first, then rename atomically
            self._save_checkpoint(checkpoint)
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}", exc_info=True)
            raise
    
    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to disk.
        
        Args:
            checkpoint: The checkpoint to save.
            
        Raises:
            IOError: If there's an error writing the checkpoint file.
        """
        try:
            # Ensure checkpoint directory exists
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a temporary file first
            temp_path = str(self.checkpoint_path) + '.tmp'
            
            # Write to temporary file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Rename temp file to final name (atomic operation)
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
            os.rename(temp_path, self.checkpoint_path)
            
            logger.debug(f"Checkpoint saved to {self.checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}", exc_info=True)
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise
    
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Public method to save a checkpoint.
        
        Args:
            checkpoint: The checkpoint to save.
        """
        self._save_checkpoint(checkpoint)
    
    def load_checkpoint(self, pdf_path: str) -> Optional[Checkpoint]:
        """Load a checkpoint if it exists and matches the PDF."""
        if not os.path.exists(self.checkpoint_dir):
            return None
            
        base_name = os.path.basename(pdf_path)
        # Remove .pdf extension if present
        if base_name.lower().endswith('.pdf'):
            base_name = base_name[:-4]
            
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{base_name}.checkpoint.json")
        if not os.path.exists(checkpoint_path):
            return None
            
        try:
            # First, validate the JSON syntax
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                content = f.read()
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in checkpoint file: {e}")
                    # Try to recover by reading valid JSON up to the error
                    try:
                        # This is a simple recovery attempt - it might not work for all cases
                        valid_json = content[:e.pos] + '}' * (content.count('{') - content.count('}'))
                        data = json.loads(valid_json)
                        logger.warning("Recovered partial checkpoint data")
                    except Exception as e2:
                        logger.error(f"Could not recover checkpoint data: {e2}")
                        return None
            
            # Verify required fields exist
            if 'metadata' not in data or 'chunks' not in data:
                logger.error("Checkpoint is missing required fields")
                return None
                
            # Verify PDF hasn't changed
            try:
                with open(pdf_path, 'rb') as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()
                
                if data['metadata'].get('pdf_hash') != current_hash:
                    logger.warning("PDF has changed since checkpoint was created. Starting fresh.")
                    return None
            except (IOError, KeyError) as e:
                logger.error(f"Error verifying PDF hash: {e}")
                return None
                
            # Validate chunks structure
            if not isinstance(data['chunks'], list):
                logger.error("Invalid chunks format in checkpoint")
                return None
                
            return Checkpoint.from_dict(data)
            
        except Exception as e:
            logger.error(f"Unexpected error loading checkpoint: {e}", exc_info=True)
            return None
    
    def delete_checkpoint(self) -> None:
        """Delete the checkpoint file."""
        try:
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
                logger.debug(f"Checkpoint deleted: {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}", exc_info=True)
    
    def is_resumable(self, pdf_path: str) -> Tuple[bool, Optional[Checkpoint], str]:
        """Check if conversion can be resumed.
        
        Args:
            pdf_path: Path to the input PDF file.
            
        Returns:
            Tuple of (can_resume, checkpoint, message)
        """
        checkpoint = self.load_checkpoint(pdf_path)
        if not checkpoint:
            return False, None, "No checkpoint found"
        
        # Check if PDF file has changed
        try:
            current_hash = self._calculate_file_hash(pdf_path)
            if current_hash != checkpoint.metadata["pdf_hash"]:
                return False, checkpoint, "PDF file has changed since last checkpoint"
        except Exception as e:
            return False, checkpoint, f"Error checking PDF file: {str(e)}"
        
        # Check if output files are accessible
        output_path = Path(checkpoint.metadata["output_md_path"])
        if not output_path.parent.exists():
            return False, checkpoint, f"Output directory does not exist: {output_path.parent}"
        
        return True, checkpoint, "Resumable checkpoint found"
    
    def cleanup(self) -> None:
        """Clean up checkpoint files."""
        self.delete_checkpoint()

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # If no exception occurred, clean up the checkpoint
            self.cleanup()
        return False  # Don't suppress exceptions