#!/usr/bin/env python3
"""
PDF to Markdown Converter

A command-line tool to convert PDF files to well-formatted Markdown using Groq's LLM.
"""

import argparse
import logging
import os
import sys
from typing import Optional

from conversion_utils.convert import convert_pdf_to_markdown
from conversion_utils.groq_client import APIClientError, RateLimitError
from conversion_utils.checkpoint import CheckpointManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf2md.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown with inline images using Groq API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "pdf",
        help="Path to input PDF file"
    )
    
    parser.add_argument(
        "output",
        help="Path to output Markdown file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--images",
        default="extracted_images",
        help="Directory to save extracted images"
    )
    
    # Checkpointing options
    checkpoint_group = parser.add_argument_group('checkpointing options')
    checkpoint_group.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of chunks to process in each batch"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retry attempts for API calls"
    )
    
    parser.add_argument(
        "--backoff-factor",
        type=float,
        default=2.0,
        help="Base multiplier for exponential backoff between retries"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process chunks in parallel (faster but uses more API tokens)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (if --parallel is enabled)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def validate_arguments(args) -> bool:
    """Validate command line arguments."""
    if not os.path.isfile(args.pdf):
        logger.error(f"Error: PDF file '{args.pdf}' does not exist.")
        return False
        
    if not args.pdf.lower().endswith('.pdf'):
        logger.error(f"Error: Input file must be a PDF with .pdf extension.")
        return False
        
    if not args.output.lower().endswith(('.md', '.markdown')):
        logger.warning("Output file does not have a .md or .markdown extension.")
    
    if args.max_retries < 0:
        logger.error("Error: --max-retries must be a non-negative integer.")
        return False
        
    if args.backoff_factor <= 0:
        logger.error("Error: --backoff-factor must be a positive number.")
        return False
        
    if args.timeout <= 0:
        logger.error("Error: --timeout must be a positive integer.")
        return False
        
    if args.parallel and args.max_workers is not None and args.max_workers < 1:
        logger.error("Error: --max-workers must be at least 1 when using --parallel.")
        return False
        
    return True

def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Set log level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().setLevel(log_level)
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        logger.error(
            "Error: GROQ_API_KEY environment variable is not set. "
            "Please set it before running this script.\n"
            "Example: export GROQ_API_KEY='your-api-key'"
        )
        sys.exit(1)
    
    try:
        logger.info(f"Starting conversion of {args.pdf} to {args.output}")
        
        # Initialize checkpoint manager with output file path
        checkpoint_manager = CheckpointManager(output_md_path=args.output)
        
        try:
            convert_pdf_to_markdown(
                pdf_path=args.pdf,
                output_md_path=args.output,
                image_output_dir=args.images,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
                backoff_factor=args.backoff_factor,
                timeout=args.timeout,
                parallel=args.parallel,
                max_workers=args.max_workers,
                checkpoint_manager=checkpoint_manager
            )
        except APIClientError as e:
            logger.error(f"API client error: {str(e)}")
            sys.exit(1)
        except RateLimitError as e:
            logger.error(f"Rate limit error: {str(e)}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            sys.exit(1)
        
        logger.info("Conversion completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("\nConversion interrupted by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C
        logger.exception("An unexpected error  occurred during conversion.")
        sys.exit(1)

if __name__ == "__main__":
    main()
