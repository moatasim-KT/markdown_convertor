# PDF to Markdown Converter Framework

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust and efficient framework for converting PDF files to well-structured Markdown using the Groq API. This tool extracts both text and images, placing image references inline in the Markdown output with proper formatting and structure.

## ✨ Features

- **High-Quality Conversion**: Utilizes Groq's compound-beta LLM for superior Markdown conversion
- **Image Handling**: Extracts and embeds images with proper references
- **Math Support**: Detects and formats mathematical expressions using LaTeX
- **Robust Error Handling**: Comprehensive error handling with automatic retries
- **Parallel Processing**: Optional parallel processing for faster conversions
- **Logging**: Detailed logging for debugging and monitoring
- **Configurable**: Customize conversion parameters via command-line arguments
- **Checkpointing**: Resume interrupted conversions from the last successful point

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Groq API key (sign up at [Groq Cloud](https://groq.com/))

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/markdown-convertor.git
   cd markdown-convertor
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Set your Groq API key as an environment variable:
   ```sh
   # On Unix/Linux/macOS
   export GROQ_API_KEY='your-api-key-here'
   
   # On Windows
   set GROQ_API_KEY=your-api-key-here
   ```

## 🛠 Usage

### Basic Conversion

Convert a PDF to Markdown with default settings:
```sh
python pdf2md.py input.pdf output.md
```

### Advanced Options

```sh
python pdf2md.py input.pdf output.md \
    --images extracted_images \
    --max-retries 5 \
    --backoff-factor 2.0 \
    --timeout 30 \
    --parallel \
    --max-workers 4 \
    --debug
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `pdf` | Path to input PDF file | Required |
| `output` | Path to output Markdown file | Required |
| `--images` | Directory to save extracted images | `extracted_images` |
| `--max-retries` | Maximum number of retry attempts | `5` |
| `--backoff-factor` | Base multiplier for exponential backoff | `2.0` |
| `--timeout` | Request timeout in seconds | `30` |
| `--parallel` | Enable parallel processing | `False` |
| `--max-workers` | Maximum parallel workers (if --parallel) | `None` (auto) |
| `--debug` | Enable debug logging | `False` |
| `--no-resume` | Disable resuming from previous checkpoint | `False` |
| `--batch-size` | Number of chunks to process in each batch | `5` |

## 🏗 Project Structure

```
markdown_convertor/
├── conversion_utils/
│   ├── pdf_parser.py    # PDF text and image extraction
│   ├── chunker.py        # Document chunking logic
│   ├── groq_client.py    # Groq API client with rate limiting
│   ├── markdown_assembler.py  # Markdown generation
│   └── convert.py        # Core conversion logic
├── pdf2md.py            # Command-line interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔄 Checkpointing and Resuming

The tool includes a checkpointing system that allows you to resume interrupted conversions from the last successful point. This is particularly useful for large PDFs or when dealing with rate limits.

### How It Works

1. The tool periodically saves progress to a checkpoint file in the `.checkpoints` directory.
2. If the process is interrupted, it will automatically detect the checkpoint on the next run and resume from where it left off.
3. Checkpoints include the state of all processed chunks, so you won't lose any progress.

### Commands

- **Resume interrupted conversion** (default behavior):
  ```sh
  python pdf2md.py input.pdf output.md
  ```

- **Start a fresh conversion** (ignore checkpoints):
  ```sh
  python pdf2md.py --no-resume input.pdf output.md
  ```

- **Adjust batch size** for processing:
  ```sh
  python pdf2md.py --batch-size 10 input.pdf output.md
  ```

### Checkpoint Location

Checkpoints are stored in the `.checkpoints` directory within the same directory as the output Markdown file. They are automatically cleaned up when the conversion completes successfully.

## 🔧 Error Handling

The tool includes comprehensive error handling with the following features:

- **Rate Limiting**: Automatic retries with exponential backoff
- **Input Validation**: Checks for valid PDF files and output paths
- **Cleanup**: Removes partially created files on failure
- **Detailed Logging**: Logs are saved to `pdf2md.log`

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 Changelog

### [1.1.0] - 2025-05-15
- Added robust rate limiting with exponential backoff and jitter
- Improved error handling and logging
- Added parallel processing support
- Enhanced command-line interface
- Added input validation
- Improved documentation
