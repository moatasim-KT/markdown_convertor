import os
import random
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "mixtral-8x7b-32768"  # Updated to a more reliable model
DEFAULT_MAX_RETRIES = 10  # Increased from 5 to 10
DEFAULT_BACKOFF_FACTOR = 1.5  # Reduced from 2.0 for more frequent retries
DEFAULT_TIMEOUT = 60  # Increased from 30 seconds
MAX_RETRY_DELAY = 300  # 5 minutes maximum delay between retries
RATE_LIMIT_WINDOW = 60  # 1 minute window for rate limiting

# Constants
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Configure session with custom retry strategy
class CustomRetry(Retry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rate_limit_reset = 0  # Timestamp when rate limit resets

    def get_backoff_time(self):
        # If we have a rate limit reset time, wait until then
        now = time.time()
        if now < self._rate_limit_reset:
            wait_time = self._rate_limit_reset - now
            return min(wait_time, MAX_RETRY_DELAY)
        return super().get_backoff_time()

# Configure session with custom retry strategy and connection pooling
session = requests.Session()
retry_strategy = CustomRetry(
    total=DEFAULT_MAX_RETRIES,
    backoff_factor=DEFAULT_BACKOFF_FACTOR,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"],
    respect_retry_after_header=True,
    raise_on_status=False  # Don't raise immediately on failed status
)
adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=10,  # Number of connections to save
    pool_maxsize=10,     # Max number of connections to save
    pool_block=True      # Block when no connections are available
)
session.mount("https://", adapter)

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "PDF-to-Markdown-Converter/1.0"
}

# LLM_PROMPT definition
# Note: Literal curly braces in LaTeX examples (e.g., for environments) are escaped with a single backslash: \{ \}
# The {{text}} placeholder is for the actual input data, which is handled by .format_map()
LLM_PROMPT = r"""
Convert the following text and images to well-formatted Markdown. Follow these guidelines meticulously:

1. For mathematical expressions:
   - Inline math: Wrap in $...$ (e.g., $E = mc^2$).
   - Display math: Wrap in $$...$$ on separate lines (e.g., $$x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$$).
   - LaTeX Environments: The input may contain text pre-formatted with LaTeX delimiters (e.g., $...$, $$...$$, \(...\), \[...\]) or LaTeX environments (e.g., \begin{align}...\end{align}, \begin{equation}...\end{equation}). Transcribe these LaTeX segments *exactly* as provided into the final Markdown. Do not attempt to re-interpret or convert them to plain text. Ensure correct escaping for Markdown where necessary.
   - Do not convert any LaTeX to plain text or re-render it. Preserve it.

2. For images:
   - If an image clearly represents an equation, transcribe it into the appropriate LaTeX math format (inline or display).
   - For other images (charts, diagrams, photos), use the Markdown syntax: ![description](path_to_image.ext). Provide a concise, relevant description.

3. Content and Structure Preservation:
   - Preserve headings (using #, ##, etc.), lists (bulleted using -, *, or +; numbered using 1., 2., etc.), tables (using Markdown table syntax), code blocks (using triple backticks ```language ... ```), and blockquotes (using >).
   - Maintain the original meaning, logical flow, and paragraph structure of the text.

4. Formatting and Cleanliness:
   - Use proper Markdown syntax throughout.
   - Ensure consistent spacing and avoid excessive blank lines.
   - Input Text Quality: The input text is extracted from a PDF and may occasionally contain minor OCR artifacts (e.g., misspellings, strange characters, misplaced spaces). Produce clean, well-structured Markdown, correcting obvious OCR errors when confident it improves readability and accuracy without altering meaning. If a word or phrase is unclear, transcribe it as best as possible.
   - Repetition Handling: Critically examine the input text for any repeated words or phrases that seem erroneous. Ensure that such repetitions, if they appear to be transcription errors from the source, are corrected or removed in the output. Do not introduce new repetitions.

5. Output Requirements:
   - Generate only the Markdown content. Do not include any introductory phrases, explanations, or summaries before or after the Markdown output.

Input Text (potentially with OCR imperfections and pre-formatted LaTeX):
{{text}}

Output Markdown:
"""

class RateLimitError(Exception):
    """Custom exception for rate limit errors"""
    pass

class APIClientError(Exception):
    """Custom exception for API client errors"""
    pass

def _exponential_backoff_with_jitter(attempt: int, max_delay: float = MAX_RETRY_DELAY) -> float:
    """
    Calculate delay with exponential backoff and jitter.
    
    Args:
        attempt: Current attempt number (0-based)
        max_delay: Maximum delay in seconds
        
    Returns:
        float: Delay in seconds with jitter
    """
    # Cap the attempt to prevent overflow
    attempt = min(attempt, 10)
    
    # Exponential backoff with full jitter
    base_delay = min((DEFAULT_BACKOFF_FACTOR ** attempt), max_delay)
    jitter = random.uniform(0, base_delay)  # Full jitter
    
    # Add some randomness to spread out retries
    jitter = jitter * (0.8 + 0.4 * random.random())
    
    return min(jitter, max_delay)

def _handle_api_error(response: requests.Response) -> None:
    """
    Handle API errors and raise appropriate exceptions with rate limit handling.
    
    Args:
        response: The failed response object
        
    Raises:
        RateLimitError: For rate limiting errors with retry information
        APIClientError: For other API errors
    """
    status_code = response.status_code
    retry_after = None
    
    try:
        error_data = response.json()
        error_msg = error_data.get('error', {}).get('message', response.text)
        
        # Check for rate limit reset time in the response
        if status_code == 429:
            retry_after = int(response.headers.get('Retry-After', '1'))
            reset_time = int(time.time()) + retry_after
            
            # Update the session's retry strategy with the reset time
            if hasattr(session.adapters['https://'], 'max_retries'):
                session.adapters['https://'].max_retries._rate_limit_reset = reset_time
                
    except (json.JSONDecodeError, ValueError):
        error_msg = response.text or 'Unknown error'
    
    if status_code == 429:
        raise RateLimitError(
            f"Rate limit exceeded. Please wait {retry_after} seconds before retrying. "
            f"Error: {error_msg}"
        )
    else:
        raise APIClientError(
            f"API request failed with status {status_code}. "
            f"Error: {error_msg}"
        )

def _clean_markdown_response(markdown: str) -> str:
    """
    Clean up markdown response by removing any assistant messages or unwanted formatting.
    
    Args:
        markdown: Raw markdown string from the API
        
    Returns:
        str: Cleaned markdown content
    """
    if not markdown:
        return ""
    
    # Remove common assistant message patterns
    assistant_phrases = [
        "Here's the markdown conversion:",
        "Here's the converted markdown:",
        "Here's your markdown:",
        "```markdown",
        "```\n",
        "\n```"
    ]
    
    # Remove each phrase if it exists at the start of the string
    for phrase in assistant_phrases:
        if markdown.startswith(phrase):
            markdown = markdown[len(phrase):].lstrip('\n')
    
    # Remove any trailing code block markers
    markdown = markdown.rstrip('\n` ')
    
    # Ensure proper newlines at the end
    return markdown.strip() + '\n'

def convert_chunk_to_markdown(
    chunk: str, 
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    timeout: int = DEFAULT_TIMEOUT,
    chunk_index: Optional[int] = None
) -> str:
    """
    Sends a chunk of text to Groq API and returns the Markdown response.
    
    Args:
        chunk: The text chunk to convert to Markdown
        max_retries: Maximum number of retry attempts
        backoff_factor: Base multiplier for exponential backoff
        timeout: Request timeout in seconds
        chunk_index: Optional index of the chunk for logging purposes
        
    Returns:
        str: Converted Markdown content with assistant messages removed
        
    Raises:
        RateLimitError: If rate limited and max retries exceeded
        APIClientError: For other API errors
        requests.RequestException: For network-related errors
    """
    chunk_info = f"chunk {chunk_index + 1} " if chunk_index is not None else ""
    
    if not GROQ_API_KEY:
        error_msg = "GROQ_API_KEY environment variable is not set"
        logger.error(f"{chunk_info}{error_msg}")
        raise ValueError(error_msg)
    
    try:
        # First escape any existing curly braces in the chunk
        # by doubling them up to prevent string formatting issues
        escaped_chunk = chunk.replace('{', '{{').replace('}', '}}')
        
        # Now format the prompt with the escaped chunk
        # Using format_map with a custom dict to prevent any further formatting
        prompt = LLM_PROMPT.format_map({'text': escaped_chunk})
        
        # Log the size of the chunk being processed
        chunk_size = len(chunk.encode('utf-8')) / 1024  # Size in KB
        logger.info(f"Processing {chunk_info}({chunk_size:.1f} KB), Prompt length: {len(prompt)} chars")
        
        try:
            # Make the API request with a timeout
            response = session.post(
                GROQ_API_URL,
                headers=HEADERS,
                json={
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a highly accurate assistant specializing in converting documents to clean, well-formatted Markdown with precise LaTeX for math. Your goal is to produce the most accurate and error-free Markdown representation of the input, correcting minor OCR imperfections where appropriate. Do not include any introductory or explanatory text - just return the converted markdown content."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.05,
                    "max_tokens": 4000,
                    "top_p": 0.9
                },
                timeout=(5, timeout)  # 5s connection timeout, timeout read timeout
            )
            response.raise_for_status()
            
            # Process successful response
            response_data = response.json()
            if not response_data.get('choices'):
                error_msg = "Invalid response format from API - no choices in response"
                logger.error(f"{chunk_info}{error_msg}")
                raise APIClientError(error_msg)
            
            # Clean up the response before returning
            raw_content = response_data['choices'][0]['message']['content']
            if not raw_content or not raw_content.strip():
                error_msg = "Empty response from API"
                logger.error(f"{chunk_info}{error_msg}")
                raise APIClientError(error_msg)
                
            cleaned_content = _clean_markdown_response(raw_content)
            logger.info(f"Successfully processed {chunk_info}in {response_data['usage']['completion_tokens']} tokens")
            return cleaned_content
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            logger.error(f"{chunk_info}{error_msg}", exc_info=logger.isEnabledFor(logging.DEBUG))
            raise APIClientError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Error preparing request for {chunk_info}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise APIClientError(error_msg) from e
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        wait_time = _exponential_backoff_with_jitter(attempt)
        
        try:
            if attempt > 0:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} for {chunk_info}after waiting {wait_time:.2f}s. "
                    f"Sending request to Groq API..."
                )
                time.sleep(wait_time)
            else:
                logger.info(f"Sending initial request for {chunk_info}to Groq API...")
            
            # Make the API request with a timeout
            start_time = time.time()
            logger.debug(f"Sending request to {GROQ_API_URL} with timeout={timeout}s")
            
            try:
                logger.info(f"Sending request to {GROQ_API_URL} with timeout={timeout}s...")
                start_time = time.time()
                try:
                    # Use a connection timeout and read timeout
                    response = session.post(
                        GROQ_API_URL,
                        headers=HEADERS,
                        json=data,
                        timeout=(5, timeout)  # 5s connection timeout, timeout read timeout
                    )
                    response.raise_for_status()  # Raise for HTTP errors
                except requests.exceptions.Timeout:
                    logger.error(f"Request timed out after {timeout}s")
                    raise
                except requests.exceptions.ConnectionError:
                    logger.error("Connection error occurred")
                    raise
                except requests.exceptions.HTTPError as e:
                    logger.error(f"HTTP error occurred: {e.response.status_code}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                    raise
                response_time = time.time() - start_time
                logger.info(f"API request completed in {response_time:.2f}s with status {response.status_code}")
            except requests.exceptions.Timeout:
                logger.error(f"Request timed out after {timeout}s")
                raise
            except requests.exceptions.ConnectionError:
                logger.error("Connection error occurred")
                raise
            except Exception as e:
                logger.error(f"Request failed: {str(e)}", exc_info=True)
                raise
                
            request_duration = time.time() - start_time
            
            # Log request details
            logger.debug(
                f"API request completed in {request_duration:.2f}s with status {response.status_code}"
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', wait_time * 2))
                retry_after = min(retry_after, MAX_RETRY_DELAY)
                
                logger.warning(
                    f"Rate limited on {chunk_info}(attempt {attempt + 1}/{max_retries}). "
                    f"Retrying after {retry_after}s..."
                )
                
                # Update rate limit reset time
                reset_time = int(time.time()) + retry_after
                if hasattr(session.adapters['https://'], 'max_retries'):
                    session.adapters['https://'].max_retries._rate_limit_reset = reset_time
                
                time.sleep(retry_after)
                continue
                
            # Check for other errors
            response.raise_for_status()
            
            # Process successful response
            try:
                result = response.json()
                logger.debug(f"API response: {json.dumps(result, indent=2)[:500]}...")  # Log first 500 chars
                
                if "choices" not in result or not result["choices"]:
                    error_msg = "Invalid response format from API - no choices in response"
                    logger.error(f"{chunk_info}{error_msg}")
                    raise APIClientError(error_msg)
                
                # Clean up the response before returning
                raw_content = result["choices"][0]["message"]["content"]
                if not raw_content or not raw_content.strip():
                    error_msg = "Empty response from API"
                    logger.error(f"{chunk_info}{error_msg}")
                    raise APIClientError(error_msg)
                    
                cleaned_content = _clean_markdown_response(raw_content)
                logger.info(f"Successfully processed {chunk_info}in {request_duration:.2f}s")
                return cleaned_content
                
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                error_msg = f"Error parsing API response: {str(e)}"
                logger.error(f"{chunk_info}{error_msg}", exc_info=True)
                raise APIClientError(error_msg) from e
            
        except requests.exceptions.HTTPError as e:
            last_exception = e
            if isinstance(e.response, requests.Response):
                try:
                    _handle_api_error(e.response)
                except RateLimitError as rle:
                    logger.warning(f"Rate limited on {chunk_info}: {str(rle)}")
                    if attempt == max_retries - 1:  # Last attempt
                        logger.error(f"Max retries ({max_retries}) exceeded for {chunk_info}")
                        raise
                    continue
                    
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f"Max retries ({max_retries}) exceeded for {chunk_info}")
                break
                
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            last_exception = e
            logger.error(f"Request error on attempt {attempt + 1}/{max_retries} for {chunk_info}: {str(e)}", 
                        exc_info=logger.isEnabledFor(logging.DEBUG))
            
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f"Max retries ({max_retries}) exceeded for {chunk_info}")
                break
            
            # Add a small delay before retrying on network errors
            time.sleep(1 + random.random())
    
    # If we get here, all retries failed
    error_msg = f"Failed after {max_retries} attempts for {chunk_info}"
    if last_exception:
        error_msg += f": {str(last_exception)}"
    logger.error(error_msg)
    raise APIClientError(error_msg) from last_exception

def _process_chunks_in_parallel(
    chunks: List[str],
    process_func: callable,
    max_workers: Optional[int] = None
) -> List[str]:
    """
    Process chunks in parallel using ThreadPoolExecutor.
    
    Args:
        chunks: List of chunks to process
        process_func: Function to process each chunk
        max_workers: Maximum number of worker threads
        
    Returns:
        List of processed results with preserved order
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    # Create a thread-local storage for rate limit tracking
    local = threading.local()
    local.last_request_time = 0
    
    def process_with_rate_limit(chunk_data: Tuple[int, str]) -> Tuple[int, str]:
        idx, chunk = chunk_data
        chunk_info = f"chunk {idx + 1}"
        try:
            # Validate chunk data
            if not isinstance(chunk, str) or not chunk.strip():
                logger.warning(f"Skipping empty or invalid {chunk_info}")
                return idx, "[Error: Empty or invalid chunk]"
                
            # Add jitter to spread out requests
            time_since_last = time.time() - local.last_request_time
            min_delay = 1.0  # At least 1 second between requests
            if time_since_last < min_delay:
                delay = min_delay - time_since_last + random.random() * 0.5
                logger.debug(f"Rate limiting: Waiting {delay:.2f}s before processing {chunk_info}")
                time.sleep(delay)
            
            local.last_request_time = time.time()
            logger.info(f"Starting processing of {chunk_info}")
            result = process_func((idx, chunk))  # Pass both index and chunk
            logger.info(f"Completed processing of {chunk_info}")
            return idx, result
            
        except Exception as e:
            error_msg = f"Error in parallel processing {chunk_info}: {str(e)}"
            logger.error(error_msg, exc_info=logger.isEnabledFor(logging.DEBUG))
            return idx, f"[Error in {chunk_info}: {str(e)}]"
    
    # Initialize results with placeholders
    results = ["[Error: Not processed]"] * len(chunks)
    
    try:
        # Determine number of workers (default to number of chunks, but not more than 4)
        num_chunks = len(chunks)
        workers = min(max_workers or min(4, num_chunks), num_chunks)
        
        logger.info(f"Starting parallel processing of {num_chunks} chunks with {workers} workers")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all chunks for processing
            future_to_index = {
                executor.submit(process_with_rate_limit, (i, chunk)): i
                for i, chunk in enumerate(chunks)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    chunk_idx, result = future.result()
                    if 0 <= chunk_idx < len(results):
                        results[chunk_idx] = result
                    else:
                        logger.error(f"Invalid chunk index {chunk_idx} returned from worker")
                except Exception as e:
                    logger.error(f"Unexpected error processing chunk {idx}: {str(e)}", 
                                exc_info=logger.isEnabledFor(logging.DEBUG))
                    if 0 <= idx < len(results):
                        results[idx] = f"[Error: {str(e)}]"
    except Exception as e:
        logger.error(f"Fatal error in parallel processing: {str(e)}", 
                    exc_info=logger.isEnabledFor(logging.DEBUG))
        # Ensure we return something for each chunk even on fatal error
        results = [f"[Error: {str(e)}]"] * len(chunks)
    
    return results

def convert_chunks_to_markdown(
    chunks: List[str],
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    timeout: int = DEFAULT_TIMEOUT,
    parallel: bool = False,
    max_workers: Optional[int] = None
) -> List[str]:
    """
    Converts a list of text chunks to Markdown using Groq API.
    
    Args:
        chunks: List of text chunks to convert
        max_retries: Maximum number of retry attempts per chunk
        backoff_factor: Base multiplier for exponential backoff
        timeout: Request timeout in seconds
        parallel: Whether to process chunks in parallel
        max_workers: Maximum number of worker threads for parallel processing
        
    Returns:
        List[str]: List of converted Markdown chunks with assistant messages removed
        
    Note:
        When parallel=True, the max_workers parameter controls the number of concurrent
        API requests. Be mindful of API rate limits when increasing this value.
    """
    if not chunks:
        logger.warning("No chunks provided for processing")
        return []
    
    # Filter out empty chunks and log a warning
    valid_chunks = []
    invalid_indices = []
    for i, chunk in enumerate(chunks):
        if chunk and isinstance(chunk, str) and chunk.strip():
            valid_chunks.append(chunk)
        else:
            invalid_indices.append(i)
            logger.warning(f"Skipping empty or invalid chunk at index {i}")
    
    if not valid_chunks:
        logger.error("No valid chunks to process")
        return ["[Error: No valid content to process]"] * len(chunks)
    
    # Calculate rate limit window (requests per minute)
    rate_limit = 50  # Conservative default rate limit (requests per minute)
    min_delay = max(60.0 / rate_limit, 1.0)  # At least 1 second between requests
    
    def process_chunk(chunk_data: Tuple[int, str]) -> str:
        idx, chunk = chunk_data
        chunk_info = f"chunk {idx + 1}/{len(chunks)}"
        try:
            logger.info(f"Starting processing of {chunk_info}")
            result = convert_chunk_to_markdown(
                chunk,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                timeout=timeout,
                chunk_index=idx
            )
            logger.info(f"Successfully processed {chunk_info}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing {chunk_info}: {str(e)}"
            logger.error(error_msg, exc_info=logger.isEnabledFor(logging.DEBUG))
            return f"[Error in {chunk_info}: {str(e)}]"
    
    try:
        if parallel:
            logger.info(f"Starting parallel processing of {len(valid_chunks)} chunks")
            # Process valid chunks in parallel with rate limiting
            processed_chunks = _process_chunks_in_parallel(
                valid_chunks, 
                process_chunk,
                max_workers=max_workers or min(4, len(valid_chunks))
            )
        else:
            # Process chunks sequentially with rate limiting
            logger.info(f"Starting sequential processing of {len(valid_chunks)} chunks")
            processed_chunks = []
            last_request_time = 0
            
            for i, chunk in enumerate(valid_chunks):
                try:
                    # Enforce rate limiting
                    time_since_last = time.time() - last_request_time
                    if time_since_last < min_delay:
                        delay = min_delay - time_since_last + random.random() * 0.5
                        logger.debug(f"Rate limiting: Waiting {delay:.2f}s before next chunk")
                        time.sleep(delay)
                    
                    last_request_time = time.time()
                    result = process_chunk((i, chunk))
                    processed_chunks.append(result)
                    
                except Exception as e:
                    error_msg = f"Unexpected error in sequential processing chunk {i}: {str(e)}"
                    logger.error(error_msg, exc_info=logger.isEnabledFor(logging.DEBUG))
                    processed_chunks.append(f"[Error: {str(e)}]")
        
        # Reconstruct the full results list with original indices
        results = []
        valid_idx = 0
        for i in range(len(chunks)):
            if i in invalid_indices:
                results.append("[Error: Empty or invalid chunk]")
            else:
                if valid_idx < len(processed_chunks):
                    results.append(processed_chunks[valid_idx])
                    valid_idx += 1
                else:
                    results.append("[Error: Chunk not processed]")
        
        return results
        
    except Exception as e:
        error_msg = f"Fatal error in chunk processing: {str(e)}"
        logger.error(error_msg, exc_info=logger.isEnabledFor(logging.DEBUG))
        return [f"[Fatal Error: {str(e)}]"] * len(chunks)
