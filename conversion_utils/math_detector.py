import re
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MathDetector:
    """
    Detects and processes mathematical expressions in text using regex patterns.
    Handles both inline and display math expressions in LaTeX format.
    """
    
    def __init__(self):
        # Pre-compile regex patterns for better performance
        # Inline math: $...$ or \(...\)
        self.inline_math = re.compile(
            r'(?<!\\)\$([^$]+)\$'  # Simple $..$ inline math
            r'|'
            r'\\\\\\(([^()]+)\\\\\\)'  # Escaped inline math
        )
        
        # Display math: $$...$$ or \[...\]
        self.display_math = re.compile(
            r'(?<![\\])(?:\$\$)(.+?)(?<!\\)(?:\$\$)'  # $$...$$
            r'|'
            r'\\\[(?:[^\[\]]|\\\[|\\\])*\\\]'  # \[...\]
        )
        
        # Common LaTeX math environments that should be preserved
        self.math_environments = [
            'align', 'align*', 'equation', 'equation*', 'gather', 'gather*',
            'multline', 'multline*', 'split', 'split*', 'array', 'matrix',
            'pmatrix', 'bmatrix', 'Bmatrix', 'vmatrix', 'Vmatrix', 'smallmatrix',
            'cases', 'dcases', 'rcases', 'drcases', 'aligned', 'gathered', 'alignedat'
        ]
        
        # Pattern for math environments: \begin{env}...\end{env}
        env_pattern = '|'.join(re.escape(env) for env in self.math_environments)
        self.math_env_pattern = re.compile(
            r'\\begin\{(' + env_pattern + r'\*?)\}'  # \begin{env} or \begin{env*}
            r'(.*?)'  # content (non-greedy)
            r'\\end\{\1\}',  # \end{env} or \end{env*}
            re.DOTALL  # allow . to match newlines
        )
        
    def is_math_expression(self, text: str) -> bool:
        """Check if the given text contains any math expressions."""
        if not text or not isinstance(text, str):
            return False
            
        return bool(
            self.inline_math.search(text) or
            self.display_math.search(text) or
            self.math_env_pattern.search(text)
        )
    
    def wrap_math(self, text: str) -> str:
        """
        Wrap math expressions in appropriate LaTeX delimiters.
        Preserves existing math expressions and wraps unmarked math expressions.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return text or ""
            
        # First, protect already properly formatted math environments
        protected = []
        
        def protect_math(match):
            protected.append(match.group(0))
            return f"__MATH_PROTECTED_{len(protected)-1}__"
            
        # Protect math environments first (most specific)
        text = self.math_env_pattern.sub(protect_math, text)
        
        # Then protect display math ($$...$$ or \[...\])
        def protect_display_math(match):
            content = match.group(1) or match.group(2)
            if content.startswith('\\['):
                return match.group(0)  # Already properly formatted
            protected.append(content.strip())
            return f"__MATH_PROTECTED_{len(protected)-1}__"
            
        text = self.display_math.sub(protect_display_math, text)
        
        # Finally, protect inline math ($...$ or \(...\))
        def protect_inline_math(match):
            content = match.group(2) or match.group(3)
            if content.startswith('\\('):
                return match.group(0)  # Already properly formatted
            protected.append(content.strip())
            return f"__MATH_PROTECTED_{len(protected)-1}__"
            
        text = self.inline_math.sub(protect_inline_math, text)
        
        # Now process the text to find and wrap unmarked math expressions
        # This is a simplified version - in practice, you might want to use a more sophisticated approach
        
        # Restore protected math expressions
        for i, math_expr in enumerate(protected):
            text = text.replace(f"__MATH_PROTECTED_{i}__", math_expr, 1)
            
        return text
    
    def clean_ocr_artifacts(self, text: str) -> str:
        """Clean common OCR artifacts and fix repeating characters."""
        if not text or not isinstance(text, str):
            return text or ""
            
        # Split into lines for line-by-line processing
        lines = text.split('\n')
        cleaned_lines = []
        
        # Keep track of the last line to detect and merge partial lines
        last_line = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                if last_line:  # Only add empty line if we have content before it
                    cleaned_lines.append(last_line)
                    last_line = ""
                continue
                
            # Skip lines that are just a single repeated character
            if len(set(line)) == 1 and len(line) > 2:
                continue
                
            # Skip lines that are just a single word repeated multiple times
            words = line.split()
            if len(words) > 1 and all(w == words[0] for w in words):
                line = words[0]  # Keep just one instance
                
            # Fix common OCR artifacts
            line = re.sub(r'(\w)\1{3,}', r'\1\1', line)  # Fix 4+ repeating chars -> 2
            
            # Fix broken words with repeating patterns (like 'ararar' -> 'ar')
            line = re.sub(r'\b((\w{1,3}?)\2{2,})\b', lambda m: m.group(2), line)
            
            # Check if this line is a continuation of the previous line
            if (last_line and last_line.endswith('-') and 
                line and line[0].islower()):
                # Merge with previous line (remove the hyphen and join)
                last_line = last_line[:-1].strip() + ' ' + line
            elif last_line and last_line[-1].isalpha() and line and line[0].islower():
                # Merge words split across lines without a hyphen
                last_line = last_line + ' ' + line
            else:
                if last_line:  # Add the previous line before starting a new one
                    cleaned_lines.append(last_line)
                last_line = line
        
        # Add the last line if it exists
        if last_line:
            cleaned_lines.append(last_line)
            
        # Join lines with proper spacing
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Fix any remaining word fragments
        cleaned_text = re.sub(r'\b(\w{1,2})\s+\1\b', r'\1', cleaned_text)  # Remove duplicate short words
        
        # Fix any remaining hyphenated words
        cleaned_text = re.sub(r'(\w)-\s+\n\s*(\w)', r'\1\2', cleaned_text)
        cleaned_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned_text)
        
        # Remove any remaining single-character lines
        cleaned_lines = [line for line in cleaned_text.split('\n') 
                        if len(line.strip()) > 1 or not line.strip().isalnum()]
        
        return '\n'.join(cleaned_lines)

    def process_text(self, text: str) -> str:
        """
        Process text to ensure proper math expression formatting.
        
        Args:
            text: Input text potentially containing math expressions
            
        Returns:
            str: Text with properly formatted math expressions and cleaned OCR artifacts
        """
        if not text or not isinstance(text, str):
            return text or ""
            
        # Clean OCR artifacts first
        text = self.clean_ocr_artifacts(text)
            
        # Split into paragraphs and process each one
        paragraphs = text.split('\n\n')
        processed_paragraphs = []
        
        for para in paragraphs:
            if not para.strip():
                processed_paragraphs.append(para)
                continue
                
            # Check if the entire paragraph is a math expression
            para_stripped = para.strip()
            if (self.inline_math.fullmatch(para_stripped) or
                self.display_math.fullmatch(para_stripped) or
                self.math_env_pattern.fullmatch(para_stripped)):
                processed_paragraphs.append(para)
                continue
                
            # Process inline math within the paragraph
            processed_paragraph = self.wrap_math(para)
            
            # Clean up any remaining artifacts after math processing
            processed_paragraph = self.clean_ocr_artifacts(processed_paragraph)
            processed_paragraphs.append(processed_paragraph)
            
        return '\n\n'.join(processed_paragraphs)

# Global instance for convenience
math_detector = MathDetector()

def detect_math(text: str) -> bool:
    """Check if the text contains any math expressions."""
    return math_detector.is_math_expression(text)

def process_math(text: str) -> str:
    """Process text to ensure proper math expression formatting."""
    return math_detector.process_text(text)