"""
Checkpointing system for PDF to Markdown conversion.

This module provides functionality to save and load conversion progress,
allowing the process to be resumed after interruptions.
"""

import json
import os
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, TypedDict, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

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
        self.checkpoint_dir = self.output_md_path.parent / '.pdf2md_checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
        image_output_dir: str,
        chunks: List[Dict[str, Any]],
        processed_chunks: int,
        total_chunks: int,
        completed: bool = False
    ) -> Checkpoint:
        """Create a new checkpoint.
        
        Args:
            pdf_path: Path to the input PDF file.
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
            pdf_hash = self._calculate_file_hash(pdf_path)
            
            # Ensure paths are absolute
            abs_pdf_path = os.path.abspath(pdf_path)
            abs_image_dir = os.path.abspath(image_output_dir)
            
            # Validate chunks data
            if not isinstance(chunks, list):
                raise ValueError("Chunks must be a list")
            
            # Create metadata
            metadata = CheckpointMetadata(
                pdf_path=abs_pdf_path,
                pdf_hash=pdf_hash,
                total_chunks=total_chunks,
                processed_chunks=processed_chunks,
                completed=completed,
                output_md_path=str(self.output_md_path),
                image_output_dir=abs_image_dir,
                timestamp=datetime.now().isoformat(),
                version="1.0"
            )
            
            # Create checkpoint
            checkpoint = Checkpoint(metadata=metadata, chunks=chunks)
            
            # Save checkpoint
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error creating checkpoint: {str(e)}")
            raise
    
    def load_checkpoint(self, pdf_path: str) -> Optional[Checkpoint]:
        """Load an existing checkpoint.
        
        Args:
            pdf_path: Path to the PDF being converted
            
        Returns:
            Checkpoint object if found, None otherwise.
        """
        try:
            if not self.checkpoint_path.exists():
                return None
                
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
                checkpoint = Checkpoint.from_dict(data)
                
            # Check if PDF file has changed
            try:
                current_hash = self._calculate_file_hash(pdf_path)
                if current_hash != checkpoint.metadata.pdf_hash:
                    logger.warning("PDF file has changed since last checkpoint")
                    return None
            except Exception as e:
                logger.error(f"Error checking PDF file: {str(e)}")
                return None
                
            return checkpoint
            
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
            if current_hash != checkpoint.metadata.pdf_hash:
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