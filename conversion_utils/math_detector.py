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
        
        # New patterns for suspected math
        # Order can be important: more specific patterns first, or patterns for typical function calls
        self.suspected_inline_patterns = [
            # Common functions like f(x), sin(x), log(n), etc.
            # Ensure it's not like object.log() or a filename.png(something)
            # Matches: func(args), sin(x), log(value), My_Var(a,b)
            # Does not match: object.log(value), file.png(description)
            re.compile(r'(?<!\w\.)\b(?:[a-zA-Z_][\w_]*|sin|cos|tan|csc|sec|cot|sinh|cosh|tanh|log|ln|exp|sqrt|det|dim|gcd|lcm|min|max|inf|sup|lim|sum|prod|abs|arg|deg|hom|ker|Pr|Re|Im)\s*\((?:[^()]*|\([^()]*\))*\)(?!\s*\w)'),
            # Simple variable assignments: x = 5, x = y, var = value (not part of a sentence like "let x = 5")
            # Matches: x = 5, y = z, myVar = 12.3
            re.compile(r'(?<!\w\s)\b([a-zA-Z_][\w_]*)\s*=\s*([a-zA-Z_][\w_]*|[0-9]+(?:\.[0-9]+)?)\b(?!\s\w)'),
            # Simple arithmetic: a + b, c - 10, var * const. Allow greek letters as variables.
            # Matches: a + b, count - 1, val * 0.5, α - β
            re.compile(r'\b([\w\u0370-\u03FF]+)\s*[\+\-\*\/]\s*([\w\u0370-\u03FF]+|[0-9]+(?:\.[0-9]+)?)\b'),
            # Comparisons: a > b, c <= 10, x != y
            # Matches: items > 5, value <= 100, result != expected, temp ≥ 0
            re.compile(r'\b([\w\u0370-\u03FF]+)\s*(?:[=<>≤≥≠≈]|[<>]=?|!=)\s*([\w\u0370-\u03FF]+|[0-9]+(?:\.[0-9]+)?)\b'),
            # Subscripts/Superscripts: x_1, p^2, T_{max}, var_sub, var^sup
            # Matches: x_1, p^2, T_{max}, H_2O, F^{test}
            re.compile(r'\b(\w+)(?:_\{?([\w\d,]+)\}?|\^\{?([\w\d,]+)\}?){1,2}\b'),
            # Fractions presented as text: a/b, (a+b)/(c+d)
            # Matches: x/y, (a+1)/(b-2)
            # Avoids matching URLs by requiring word boundaries and no preceding dot.
            re.compile(r'(?<!\w\.)\b([\w\u0370-\u03FF\(\)\[\]\{\}]+)\s*\/\s*([\w\u0370-\u03FF\(\)\[\]\{\}]+)\b(?!\.\w)'),
            # Greek letters in simple equations (e.g. alpha = beta + gamma)
            # Matches: α = β + γ, Δ = b^2 - 4ac
            re.compile(r'\b[\u0370-\u03FF]+\s*=\s*[\w\d\.\s\+\-\*\/\[\]\(\)\u0370-\u03FF^_{}]+\b'),
            # Pattern for series of numbers and operators: e.g. "1 + 2 - 3.4 * 5"
            # Matches: 1 + 2, 3.0 * 4.5 / 2, 10 - 5 + 3
            re.compile(r'\b(?:[0-9]+(?:\.[0-9]+)?\s*[\+\-\*\/^]\s*)+[0-9]+(?:\.[0-9]+)?\b'),
            # Isolated terms with subscripts/superscripts or math functions that might not be caught by others
            # Matches: x_i, y^n, log(k) - useful if they are standalone.
            re.compile(r'\b(?:[a-zA-Z_][\w_]*_(?:\{[\w,]+\}|\w+)|\w+\^(?:\{[\w,]+\}|\w+)|(?:sin|cos|log|ln|exp)\([\w,]+\))\b'),
        ]

    def _is_likely_display_math(self, line: str, min_math_char_ratio: float = 0.35, min_len: int = 3, max_len: int = 350) -> bool:
        """
        Heuristic check if a line is likely display math.
        A line is a candidate if it's not too short/long, has a decent ratio of math-like characters,
        or follows common equation patterns.
        """
        stripped_line = line.strip()
        if not min_len < len(stripped_line) < max_len:
            return False

        # Avoid lines that are clearly code comments, section headers, or just plain text ending in a period.
        if stripped_line.startswith(('#', '//', '/*')):
            return False
        if re.match(r'^\s*[0-9]+(?:\.[0-9]+)*\s+[A-Za-z\s]+$', stripped_line) and not re.search(r'[=<>≤≥≠≈\+\-\*\/^\[\]\{\}\(\)\u0370-\u03FF]', stripped_line): # e.g. "2.1 Introduction to..."
            return False
        if stripped_line.endswith('.') and stripped_line.count(' ') > 3 and not re.search(r'[=<>≤≥≠≈\+\-\*\/^\[\]\{\}\(\)\u0370-\u03FF]', stripped_line): # Ends with period, multiple spaces (sentence like) and no math symbols
            return False
        # Avoid lines that are just a few words without clear math operators
        if len(stripped_line.split()) < 10 and not re.search(r'[=<>≤≥≠≈\+\-\*\/^\[\]\{\}\(\)_]', stripped_line): # Few words and no math operators
             if len(re.findall(r'[a-zA-Z]', stripped_line)) > 0.8 * len(stripped_line.replace(" ","")): # if very high alpha ratio
                return False


        # Count math-like characters
        math_chars = re.findall(r'[0-9=<>≤≥≠≈\+\-\*\/^\[\]\{\}\(\)\u0370-\u03FF\u2211\u220F\u222B\u2202\u221A\u2190-\u21FF]', stripped_line) # Added unicode arrows
        
        non_space_chars = len(re.sub(r'\s', '', stripped_line))
        if non_space_chars == 0:
            return False

        ratio = len(math_chars) / non_space_chars
        
        has_operator = re.search(r'[=<>≤≥≠≈\+\-\*\/^]', stripped_line)
        has_sum_prod_integral = re.search(r'[\u2211\u220F\u222B]', stripped_line) # Σ Π ∫
        has_greek = re.search(r'[\u0370-\u03FF]', stripped_line) # Greek letters
        
        # Common equation structures (e.g., var = expression, or expression involving operators)
        is_equation_like = (
            re.match(r'^\s*([\w\u0370-\u03FF\(\)\[\]\{\}\^_]+\s*)+=\s*.+', stripped_line) or 
            re.match(r'^\s*((?:[\w\u0370-\u03FF\(\)\[\]\{\}\^_]+|[0-9\.]+)\s*[\+\-\*\/^]\s*)+(?:[\w\u0370-\u03FF\(\)\[\]\{\}\^_]+|[0-9\.]+).*', stripped_line) or
            (has_sum_prod_integral and has_operator) or
            (len(re.findall(r'[a-zA-Z]', stripped_line)) < 5 and has_operator and has_greek) # Few letters, but has operator and greek
        )
                           
        if ratio > min_math_char_ratio and (has_operator or is_equation_like or has_sum_prod_integral or (has_greek and len(math_chars) > 2)):
            # Avoid if it's a list item that isn't clearly an equation.
            if re.match(r'^\s*[\*\-\+]?\s*[a-zA-Z0-9][\.\)]\s+', stripped_line) and not is_equation_like: # e.g. "1. text" or "a) text"
                # Check if what follows the list marker looks like an equation
                if not re.search(r'[=<>≤≥≠≈\+\-\*\/^]', stripped_line.split('.',1)[-1].split(')',1)[-1]):
                    return False
            logger.debug(f"Identified as likely display math (ratio {ratio:.2f}): '{stripped_line}'")
            return True
            
        if ratio > (min_math_char_ratio * 0.6) and is_equation_like: # Lower ratio acceptable if structure is very equation-like
            logger.debug(f"Identified as likely display math (equation-like, ratio {ratio:.2f}): '{stripped_line}'")
            return True
            
        return False

    def _should_wrap_inline(self, match: re.Match, context_text: str) -> bool:
        """
        Check if a suspected inline math match should actually be wrapped.
        Helps avoid false positives like object.method(args) or parts of URLs.
        """
        match_str = match.group(0).strip()
        start_char_idx = match.start()
        end_char_idx = match.end()

        # 1. Avoid if it looks like a method call: object.method(args)
        if match_str[0].isalpha() and '(' in match_str and start_char_idx > 0 and context_text[start_char_idx - 1] == '.':
            pre_dot_context = context_text[max(0, start_char_idx - 30):start_char_idx-1]
            if re.search(r'\b\w+$', pre_dot_context):
                logger.debug(f"Skipping suspected inline math (method call pattern): {match_str}")
                return False

        # 2. Avoid if it's part of a URL, filename, or common non-math technical terms.
        if (match_str.startswith("http") or match_str.startswith("www.") or ".com" in match_str or
            ".org" in match_str or ".net" in match_str or ".edu" in match_str or ".gov" in match_str or
            match_str.endswith((".png", ".jpg", ".jpeg", ".gif", ".pdf", ".txt", ".csv", ".html", ".js", ".css"))):
            logger.debug(f"Skipping suspected inline math (URL/filename pattern): {match_str}")
            return False
        
        # Check for path-like structures, e.g. word/word or word/word/word
        # The fraction regex `(?<!\w\.)\b(...)\s*\/\s*(...)\b(?!\.\w)` tries to avoid this.
        # If a match contains multiple slashes and is not within parens/brackets, it might be a path.
        if match_str.count('/') > 1 and not (match_str.startswith('(') and match_str.endswith(')')):
             # check if parts between slashes are simple words
            path_parts = match_str.split('/')
            if all(re.match(r'^\w+$', part) for part in path_parts):
                logger.debug(f"Skipping suspected inline math (path-like with multiple slashes): {match_str}")
                return False

        # 3. Avoid wrapping if the match is a common word or very short and ambiguous.
        if match_str.lower() in ["is", "as", "if", "of", "or", "to", "be", "in", "on", "at", "it", "and", "the", "a", "an", "var", "val", "sum"]: # common short words
             if not re.search(r'[\d=<>≤≥≠≈\+\-\*\/^_{}\[\]\(\)\u0370-\u03FF]', match_str): # If no math symbols
                logger.debug(f"Skipping suspected inline math (common word): {match_str}")
                return False
        
        if len(match_str) < 2 and not re.search(r'[\u0370-\u03FF]', match_str): # Single char, not greek
            logger.debug(f"Skipping suspected inline math (too short): {match_str}")
            return False

        # 4. Avoid if it's likely part of an identifier like ClassName_METHOD_NAME or a constant.
        if re.fullmatch(r'[A-Z][A-Za-z0-9_]*_[A-Z_0-9]+', match_str):
            logger.debug(f"Skipping suspected inline math (constant-like): {match_str}")
            return False
            
        # 5. Check for surrounding quotes that might indicate a string literal rather than math
        # This is tricky; "$x=5$" is valid math. But "'x = y'" in text might be a literal.
        # Only skip if the content itself doesn't strongly suggest math.
        prev_char = context_text[start_char_idx-1] if start_char_idx > 0 else " "
        next_char = context_text[end_char_idx] if end_char_idx < len(context_text) else " "
        if (prev_char == '"' and next_char == '"' and match.group(0).count('"') == 0) or \
           (prev_char == "'" and next_char == "'" and match.group(0).count("'") == 0) :
            if not re.search(r'[=<>≤≥≠≈\+\-\*\/^_{}\[\]\(\)\u0370-\u03FF]', match_str) and match_str.isalpha(): # if quoted and just text
                logger.debug(f"Skipping suspected inline math (quoted text): {match_str}")
                return False

        # 6. If the pattern is for `f(x)` style, ensure it's not like `ClassName.FunctionName(...)`
        # This is partly handled by `(?<!\w\.)` in the regex itself.

        logger.debug(f"Match '{match_str}' passed _should_wrap_inline checks.")
        return True

    def _inline_wrapper_callback_factory(self, protected_list: List[str], segment_text: str):
        """
        Factory to create a callback for re.sub that can access the protected_list
        and the current text segment for context checks.
        """
        def callback(match: re.Match) -> str:
            original_math = match.group(0)
            
            # Check if the match is already inside a protected block (e.g. from display math wrapping)
            # This is a safeguard, ideally segments are already non-protected text.
            if "__MATH_PROTECTED_" in original_math:
                return original_math

            if self._should_wrap_inline(match, segment_text):
                # Avoid re-wrapping if it's already explicitly wrapped
                # This check is primarily for $...$ or \(...\) forms.
                # The main protection for existing math is done before this stage.
                # This is a fallback for suspected math that might coincidentally include delimiters.
                if (original_math.startswith('$') and original_math.endswith('$') and original_math.count('$') % 2 == 0 and len(original_math) > 2) or \
                   (original_math.startswith('\\(') and original_math.endswith('\\)')):
                    # If it's already wrapped, ensure it gets into the protected list if not yet there
                    # This situation should be rare due to pre-protection.
                    # For safety, we can add it to protected list if it's not a placeholder.
                    if not re.match(r'^__MATH_PROTECTED_\d+__$', original_math):
                        logger.debug(f"Suspected inline math '{original_math}' seems already wrapped, ensuring protection.")
                        protected_list.append(original_math)
                        return f"__MATH_PROTECTED_{len(protected_list)-1}__"
                    return original_math # It's already a placeholder or perfectly wrapped

                logger.info(f"Wrapping suspected inline math: '{original_math}'")
                wrapped_math = f"${original_math.strip()}$"
                protected_list.append(wrapped_math)
                return f"__MATH_PROTECTED_{len(protected_list)-1}__"
            return original_math
        return callback

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
            
        protected: List[str] = []
        
        # Inner function to protect matches by adding them to the `protected` list 
        # and returning a placeholder.
        def protect_replacement_callback(match: re.Match) -> str:
            original_text = match.group(0)
            # Check if already protected to avoid nested protection of same content
            if re.fullmatch(r"__MATH_PROTECTED_\d+__", original_text):
                 return original_text
            # Avoid adding duplicates if the same string is matched by different top-level rules
            # This is a simple check; complex overlaps are harder.
            if original_text not in protected:
                 protected.append(original_text)
                 logger.debug(f"Protecting existing math: '{original_text}' with placeholder __MATH_PROTECTED_{len(protected)-1}__")
                 return f"__MATH_PROTECTED_{len(protected)-1}__"
            else:
                # If it's already in protected, find its index
                try:
                    idx = protected.index(original_text)
                    logger.debug(f"Existing math '{original_text}' found again, reusing placeholder __MATH_PROTECTED_{idx}__")
                    return f"__MATH_PROTECTED_{idx}__"
                except ValueError: # Should not happen if "in protected" is true
                     # Fallback, should be rare
                     protected.append(original_text)
                     return f"__MATH_PROTECTED_{len(protected)-1}__"


        # 1. Protect existing, well-defined math environments (most specific)
        text = self.math_env_pattern.sub(protect_replacement_callback, text)
        
        # 2. Protect existing display math ($$...$$ or \[...\])
        text = self.display_math.sub(protect_replacement_callback, text)
        
        # 3. Protect existing inline math ($...$ or \(...\))
        text = self.inline_math.sub(protect_replacement_callback, text)
        
        # 4. NEW LOGIC: Iterate through the 'text' (which now has explicit math protected).
        #    Use new regexes/heuristics to find suspected math.
        #    Split text by protected blocks, process intermediate segments
        
        segments = re.split(r'(__MATH_PROTECTED_\d+__)', text)
        processed_segments = []
        
        for idx, segment in enumerate(segments):
            if re.fullmatch(r'__MATH_PROTECTED_\d+__', segment): # If it's a placeholder
                processed_segments.append(segment)
                continue

            if not segment.strip(): # Skip empty segments
                processed_segments.append(segment)
                continue

            current_segment_text = segment
            
            # A. Attempt to identify and wrap likely display math (entire lines)
            # This needs to be done carefully to avoid breaking inline math finding.
            # We will process lines, and if a line is wrapped, we replace it in a temporary
            # list of lines for the current segment.
            
            temp_lines = current_segment_text.split('\n')
            potentially_new_display_math_lines = []
            modified_line_in_segment = False

            for line_num, line_content in enumerate(temp_lines):
                line_placeholder_or_original = line_content
                # Check if the line itself is not already a placeholder from a previous step (unlikely here)
                if re.fullmatch(r'__MATH_PROTECTED_\d+__', line_content.strip()):
                    potentially_new_display_math_lines.append(line_content)
                    continue

                if self._is_likely_display_math(line_content):
                    # Check if the line ALREADY contains protected inline math.
                    # If so, it's too complex to simply wrap the whole line as display math.
                    # Example: "This is an equation $x=1$ and another $y=2$." - shouldn't become display.
                    # However, if the protected math IS the whole line, it's fine.
                    
                    # A simple check: if the line contains placeholders but isn't just one placeholder.
                    contains_placeholders = "__MATH_PROTECTED_" in line_content
                    is_single_placeholder = re.fullmatch(r'\s*__MATH_PROTECTED_\d+__\s*', line_content)

                    if contains_placeholders and not is_single_placeholder:
                        logger.debug(f"Line '{line_content[:50]}...' contains inline math, skipping display math wrapping for the whole line.")
                        potentially_new_display_math_lines.append(line_content)
                    else:
                        logger.info(f"Wrapping suspected display math line: '{line_content[:100]}'")
                        # Important: the content added to `protected` must be the *final* desired string.
                        wrapped_display_line = f"$$ {line_content.strip()} $$"
                        protected.append(wrapped_display_line)
                        line_placeholder_or_original = f"__MATH_PROTECTED_{len(protected)-1}__"
                        modified_line_in_segment = True
                
                potentially_new_display_math_lines.append(line_placeholder_or_original)

            if modified_line_in_segment:
                 current_segment_text = "\n".join(potentially_new_display_math_lines)
            
            # B. Attempt to identify and wrap suspected inline math from the remaining segment parts
            # (either original segment, or segment after some lines became display math)
            
            # Split the current_segment_text by already protected blocks again,
            # in case display math wrapping introduced new placeholders.
            # This ensures inline search only on "raw" text parts.
            
            inline_search_segments = re.split(r'(__MATH_PROTECTED_\d+__)', current_segment_text)
            final_inline_processed_parts = []

            for sub_segment_idx, sub_segment in enumerate(inline_search_segments):
                if re.fullmatch(r'__MATH_PROTECTED_\d+__', sub_segment):
                    final_inline_processed_parts.append(sub_segment)
                    continue
                if not sub_segment.strip():
                    final_inline_processed_parts.append(sub_segment)
                    continue

                processed_sub_segment = sub_segment
                # Create the callback for re.sub for this specific sub_segment context
                # The factory pattern allows `_should_wrap_inline` to use `sub_segment` for context.
                inline_cb = self._inline_wrapper_callback_factory(protected, sub_segment)
                for pattern in self.suspected_inline_patterns:
                    processed_sub_segment = pattern.sub(inline_cb, processed_sub_segment)
                final_inline_processed_parts.append(processed_sub_segment)
            
            processed_segments.append("".join(final_inline_processed_parts))
            
        text = "".join(processed_segments)
        
        # 5. Restore all protected math expressions
        # Sort placeholders by index numerically (descending) to avoid issues with nested placeholders
        # e.g., __MATH_PROTECTED_10__ inside __MATH_PROTECTED_1__ if not handled carefully.
        # However, simple sequential replacement should work if placeholders are unique.
        
        # Standard restoration loop
        for i, math_expr in enumerate(protected):
            placeholder = f"__MATH_PROTECTED_{i}__"
            # Ensure we only replace actual placeholders and not parts of other text
            # Using regex to replace ensures we replace the whole placeholder.
            # text = text.replace(placeholder, math_expr, 1) # original way
            # Make sure the replacement is robust, especially if math_expr contains $
            escaped_math_expr = math_expr.replace('\\', '\\\\').replace('$', '\\$')
            try:
                text = re.sub(re.escape(placeholder), lambda _: math_expr, text, 1) # Slower but safer for special chars in math_expr
            except re.error as e:
                logger.error(f"Regex error while restoring placeholder {placeholder} with content {math_expr[:50]}...: {e}")
                # Fallback to string replace if regex fails (e.g. due to complex content in math_expr)
                text = text.replace(placeholder, math_expr, 1)

            
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
            
            # Ligature replacements
            line = line.replace("ﬁ", "fi")
            line = line.replace("ﬂ", "fl")
            line = line.replace("ﬀ", "ff")
            line = line.replace("ﬆ", "st")
            line = line.replace("ﬅ", "ft")
            line = line.replace("æ", "ae")
            line = line.replace("œ", "oe")
            # Add more common ligatures if needed

            # Rule 1: Improve repeating character rule (3+ to 2)
            # General rule: reduce 3+ consecutive identical word characters to 2
            line = re.sub(r'(\w)\1{2,}', r'\1\1', line)
            # Specific common cases that should be single if repeated (e.g. 'helooo' -> 'helo')
            # This can be tricky; for now, the general rule is safer.
            # Example: line = re.sub(r'([lL])\1{2,}', r'\1', line) # lll -> l
            # line = re.sub(r'([oO])\1{2,}', r'\1', line) # ooo -> o (careful with "cool")

            # Rule 2: Word-level repetition (e.g., "the the" -> "the")
            # Handles different capitalization by keeping the first instance's case.
            line = re.sub(r'\b(\w+)(?:\s+\1)+\b', r'\1', line, flags=re.IGNORECASE)

            # Rule 3: Punctuation spacing
            # Remove space before common punctuation
            line = re.sub(r'\s+([,.!?;:])', r'\1', line)
            # Ensure space after common punctuation (if not end of line or followed by other punctuation)
            line = re.sub(r'([,.!?;:])(?=\w)', r'\1 ', line)
            # Correct space around hyphens in compound words (e.g., "word - word" to "word-word")
            # This should be applied carefully to avoid joining math operators like " - "
            # For now, only join if hyphen is clearly part of a word split by spaces
            line = re.sub(r'(\b\w+)\s+-\s+(\w+\b)', r'\1-\2', line)

            # Fix broken words with repeating patterns (like 'ararar' -> 'ar')
            line = re.sub(r'\b((\w{1,3}?)\2{2,})\b', lambda m: m.group(2), line)

            # Noise removal: Skip lines that are only punctuation/symbols (conservative)
            if re.fullmatch(r'^[^\w\sa-zA-Z0-9\(\)\[\]\{\}\$\+\=\-\*\/\\^%<>~_]+$', line.strip()): # allow common math symbols
                if not last_line: # if this is the first line and it's noise, skip
                    continue 
                # if last_line exists, this might be a separator, keep it for now
                # but if it's very short, it's likely noise
                if len(line.strip()) < 3:
                    continue


            # Check if this line is a continuation of the previous line
            if (last_line and last_line.endswith('-') and 
                line and (line[0].islower() or line[0].isdigit())): # Allow continuation if next word starts with digit
                # Merge with previous line (remove the hyphen and join)
                last_line = last_line[:-1].strip() + line # No extra space if hyphenated
            elif last_line and last_line[-1].isalpha() and line and (line[0].islower() or line[0].isdigit()):
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
        
        # Post-joining cleaning steps
        
        # Fix any remaining word fragments or short duplicates
        # e.g. "a a" -> "a", but be careful with "..." or similar constructs.
        # This rule is tricky, the previous one `\b(\w{1,2})\s+\1\b` was a bit aggressive.
        # Let's refine it: only remove if the short word is exactly the same.
        cleaned_text = re.sub(r'\b(\w{1,3})(?:\s+\1)+\b', r'\1', cleaned_text, flags=re.IGNORECASE)


        # Fix any remaining hyphenated words broken across lines (more aggressively after line joining)
        # This specifically targets hyphen, newline, word.
        cleaned_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned_text)
        
        # Remove lines that became empty after cleaning or are just noise
        final_cleaned_lines = []
        for l_idx, l_content in enumerate(cleaned_text.split('\n')):
            stripped_line = l_content.strip()
            if not stripped_line: # remove empty lines
                continue
            # Remove lines that are just a single character (unless it's a valid symbol like '%')
            # or very short non-alphanumeric noise
            if len(stripped_line) == 1 and not stripped_line[0].isalnum() and stripped_line[0] not in '%()[]${}': # common symbols
                 continue
            if len(stripped_line) < 2 and not stripped_line.isalnum(): # e.g. single '*' or '&'
                 continue
            
            # More aggressive noise removal: if a line is very short and mostly non-alphanumeric
            # This is heuristic and needs care.
            if len(stripped_line) < 5 and not re.search(r'[a-zA-Z]{2,}', stripped_line) and not re.search(r'[0-9]{2,}', stripped_line):
                 # If it has some alphanumeric but not much, and it's short, it might be noise
                 # e.g. "a * b", "c - d" (could be math, so be careful)
                 # Count non-alphanumeric chars (excluding spaces)
                 non_alnum_chars = len(re.findall(r'[^\w\s]', stripped_line))
                 alnum_chars = len(re.findall(r'\w', stripped_line))
                 if non_alnum_chars > alnum_chars and alnum_chars < 3 : # e.g. "**a*", "-b-", "c&d"
                     # Check if it's a potential list item or something structured
                     if not stripped_line.startswith(("- ", "* ", "+ ")) and not re.match(r'^\w\.\s', stripped_line):
                         # logger.debug(f"Skipping potentially noisy short line: {stripped_line}")
                         continue
            
            final_cleaned_lines.append(l_content)
        
        return '\n'.join(final_cleaned_lines)

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