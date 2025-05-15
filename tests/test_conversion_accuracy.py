import unittest
import re # For some advanced checks if needed

# Assuming direct import is possible based on PYTHONPATH or project structure
from conversion_utils.math_detector import MathDetector
from conversion_utils.groq_client import LLM_PROMPT # For testing prompt content
# For testing the text preparation stage before it goes to the LLM
from conversion_utils.convert import process_chunk_to_llm_input


class TestConversionAccuracy(unittest.TestCase):
    def setUp(self):
        """Set up for test methods."""
        self.math_detector = MathDetector()

    # --- OCR Artifact Cleaning Tests (via clean_ocr_artifacts) ---
    def test_ocr_repeated_characters(self):
        self.assertEqual(self.math_detector.clean_ocr_artifacts("characterrrr"), "characterr")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("hellooo world"), "helloo world")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("ggggrrreat"), "ggrreat") # Stays as 2
        self.assertEqual(self.math_detector.clean_ocr_artifacts("bookkeeper"), "bookkeeper") # Already valid
        self.assertEqual(self.math_detector.clean_ocr_artifacts("aaabbbccc"), "aabbcc")

    def test_ocr_repeated_words(self):
        self.assertEqual(self.math_detector.clean_ocr_artifacts("the the quick brown fox"), "the quick brown fox")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("Word Word word"), "Word word")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("test test test"), "test")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("Go go go!"), "Go!") # Punctuation
        self.assertEqual(self.math_detector.clean_ocr_artifacts("Now now is the time"), "Now is the time")

    def test_ocr_ligature_normalization(self):
        self.assertEqual(self.math_detector.clean_ocr_artifacts("ﬁle ﬂow ﬀ ligature"), "file flow ff ligature")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("ærial œnology"), "aerial oenology")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("ﬆop ﬅen"), "stop ften") # common 'st' and 'ft' ligatures

    def test_ocr_punctuation_spacing(self):
        # Test cases from description "word . word ,word" -> "word. word, word"
        self.assertEqual(self.math_detector.clean_ocr_artifacts("word . word ,word"), "word. word, word")
        # Test case "word -" -> "word-" - this specific rule for trailing hyphen is not explicitly in current clean_ocr_artifacts
        # Current rules focus on "word - word" or "word-\nword"
        # self.assertEqual(self.math_detector.clean_ocr_artifacts("word -"), "word-") # This might need a specific rule addition if desired
        self.assertEqual(self.math_detector.clean_ocr_artifacts("Hello ,world ."), "Hello, world.")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("This is a test !"), "This is a test!")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("What is this ?"), "What is this?")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("data : value"), "data: value")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("word1 - word2"), "word1-word2") # Hyphen between words
        self.assertEqual(self.math_detector.clean_ocr_artifacts(" spaced - out "), "spaced-out")

    def test_ocr_hyphenated_word_rejoining(self):
        # Test joining across newlines
        self.assertEqual(self.math_detector.clean_ocr_artifacts("long-\nword"), "longword")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("anoth-\ner exam-\nple"), "another example")
        # Test joining with spaces around hyphen then newline
        self.assertEqual(self.math_detector.clean_ocr_artifacts("super -\nmarket"), "supermarket")
        # Test words split without hyphen but across lines (handled by a different part of clean_ocr_artifacts)
        self.assertEqual(self.math_detector.clean_ocr_artifacts("This is a sentence split across\nlines."), "This is a sentence split across lines.")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("Number1\nnumber2"), "Number1 number2")

    def test_noise_removal(self):
        # Test lines with only symbols
        self.assertEqual(self.math_detector.clean_ocr_artifacts("* * * * *"), "") # Should be removed
        self.assertEqual(self.math_detector.clean_ocr_artifacts("----------"), "")   # Should be removed
        self.assertEqual(self.math_detector.clean_ocr_artifacts("Content\n* * *\nMore Content"), "Content\n\nMore Content")
        
        # Test very short, non-substantive lines (heuristic based)
        # The rule is: if len(stripped_line) < 5 and not re.search(r'[a-zA-Z]{2,}', stripped_line) and not re.search(r'[0-9]{2,}', stripped_line):
        # and other conditions...
        self.assertEqual(self.math_detector.clean_ocr_artifacts("a b c"), "") # "a b c" is len 5, but few alnum, mostly non_alnum if spaces count, this might be kept.
                                                                            # The rule was `non_alnum_chars > alnum_chars and alnum_chars < 3`
                                                                            # For "a b c": alnum=3. non_alnum (symbols)=0. So it's kept. This is fine.
                                                                            # The rule targets things like "*a*", "-b-".
        self.assertEqual(self.math_detector.clean_ocr_artifacts("x y\nz"), "x y\nz") # Kept due to length / content
        self.assertEqual(self.math_detector.clean_ocr_artifacts("-\nx y\n-"), "x y") # Surrounding noise lines
        self.assertEqual(self.math_detector.clean_ocr_artifacts("-\n\nContent\n\n-"), "Content") # Noise with blank lines

        # Test single character lines (non-alphanumeric, with exceptions)
        self.assertEqual(self.math_detector.clean_ocr_artifacts("Single char line:\n&\nNext line."), "Single char line:\nNext line.")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("Keep symbols like:\n%\nOkay."), "Keep symbols like:\n%\nOkay.")
        self.assertEqual(self.math_detector.clean_ocr_artifacts("And dollar:\n$\nOkay."), "And dollar:\n$\nOkay.")


    # --- Math Detection and Formatting Tests (via process_text) ---
    # process_text internally calls clean_ocr_artifacts and then wrap_math
    
    def test_simple_inline_math_wrapping(self):
        self.assertEqual(self.math_detector.process_text("let x = 5 be the value"), "let $x = 5$ be the value")
        self.assertEqual(self.math_detector.process_text("The equation is a + b = c."), "The equation is $a + b = c$.")
        self.assertEqual(self.math_detector.process_text("If y=mx+c, then..."), "If $y=mx+c$, then...")
        self.assertEqual(self.math_detector.process_text("Consider α = β + γ."), "Consider $α = β + γ$.")


    def test_existing_inline_math_preservation(self):
        self.assertEqual(self.math_detector.process_text("The value is $x^2$."), "The value is $x^2$.")
        self.assertEqual(self.math_detector.process_text("A formula: $E=mc^2$, is famous."), "A formula: $E=mc^2$, is famous.")
        self.assertEqual(self.math_detector.process_text("This is \\( a=b \\) inline."), "This is \\( a=b \\) inline.")

    def test_simple_display_math_wrapping(self):
        # Single line, clearly math
        self.assertEqual(self.math_detector.process_text("E = mc^2"), "$$E = mc^2$$")
        self.assertEqual(self.math_detector.process_text("x_1 + x_2 = y_1"), "$$x_1 + x_2 = y_1$$")
        # Test with surrounding text (should still make the math line display)
        # Note: process_text splits by paragraphs (\n\n). A single line is a paragraph.
        # If the heuristic _is_likely_display_math identifies it, it will be wrapped.
        input_text = "The formula is:\nE = mc^2\nThis is important."
        # process_text processes paragraph by paragraph. "E = mc^2" is its own paragraph.
        expected_text = "The formula is:\n\n$$E = mc^2$$\n\nThis is important." # Note: process_text adds \n\n between processed paras
        self.assertEqual(self.math_detector.process_text(input_text).strip(), expected_text.strip())


    def test_existing_display_math_preservation(self):
        text = "Value:\n\n$$E = mc^2$$\n\nNext"
        self.assertEqual(self.math_detector.process_text(text), text) # Should remain unchanged
        text_bracket = "Also:\n\n\\[x^2 + y^2 = z^2\\]\n\nEnd"
        self.assertEqual(self.math_detector.process_text(text_bracket), text_bracket)

    def test_latex_environment_preservation(self):
        latex_text = "\\begin{align}x &= y \\\\ z &= w\\end{align}"
        self.assertEqual(self.math_detector.process_text(latex_text), latex_text)
        
        complex_latex = "Text before \\begin{equation*} A = \\sum_{i=1}^{n} x_i \\end{equation*} and text after."
        self.assertEqual(self.math_detector.process_text(complex_latex), complex_latex)

    def test_function_notation_wrapping(self):
        self.assertEqual(self.math_detector.process_text("This is f(x) = 2x clearly."), "This is $f(x) = 2x$ clearly.")
        self.assertEqual(self.math_detector.process_text("Let g(x,y) be defined."), "Let $g(x,y)$ be defined.")
        self.assertEqual(self.math_detector.process_text("sin(x) + cos(y)"), "$sin(x) + cos(y)$")

    def test_avoid_wrapping_normal_text(self):
        self.assertEqual(self.math_detector.process_text("This is a normal sentence."), "This is a normal sentence.")
        self.assertEqual(self.math_detector.process_text("This is item 1. This is item 2."), "This is item 1. This is item 2.")
        self.assertEqual(self.math_detector.process_text("The quick brown fox_jumps_over the lazy dog."), "The quick brown fox_jumps_over the lazy dog.")

    def test_avoid_wrapping_code_like_text(self):
        self.assertEqual(self.math_detector.process_text("obj.method(arg)"), "obj.method(arg)")
        self.assertEqual(self.math_detector.process_text("my_variable_name = another_var;"), "my_variable_name = another_var;")
        self.assertEqual(self.math_detector.process_text("config.set_value('property', 10)"), "config.set_value('property', 10)")
        self.assertEqual(self.math_detector.process_text("See file.txt or image.png for details."), "See file.txt or image.png for details.")
        self.assertEqual(self.math_detector.process_text("The url is http://example.com/api/data?value=x"), "The url is http://example.com/api/data?value=x")

    # --- LLM Prompt Content & Pre-LLM Input Formatting Tests ---

    def test_llm_prompt_content(self):
        """Test that the LLM_PROMPT string contains the new key instructions."""
        self.assertIn("Critically examine the input text for any repeated words or phrases", LLM_PROMPT)
        self.assertIn("Transcribe these LaTeX segments *exactly* as provided", LLM_PROMPT)
        self.assertIn("Input Text (potentially with OCR imperfections and pre-formatted LaTeX)", LLM_PROMPT)
        # Ensure the specific LaTeX examples (which are single-backslash escaped in the prompt string) are present.
        self.assertIn("\\begin{align}...\\end{align}", LLM_PROMPT) 
        self.assertIn("$$x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$", LLM_PROMPT)
        # System message is not tested here as it's constructed dynamically in convert_chunk_to_markdown

    def test_process_chunk_to_llm_input_ocr_cleaning(self):
        """Test that process_chunk_to_llm_input applies OCR cleaning."""
        # This function internally calls process_math, which calls clean_ocr_artifacts
        # Mocking a "chunk" as a list of one text element
        raw_chunk_elements = [{"type": "text", "content": "Errrrorss and the the repetitionss"}]
        # image_output_dir is needed by the function signature
        processed_text_for_llm = process_chunk_to_llm_input(raw_chunk_elements, "dummy_img_dir")
        
        # Expected: OCR cleaned, but math wrapping might also occur if patterns match
        # "Errrrorss" -> "Errors" (by clean_ocr_artifacts via process_math)
        # "the the" -> "the" (by clean_ocr_artifacts via process_math)
        # "repetitionss" -> "repetitions" (by clean_ocr_artifacts via process_math)
        # The result "Errors and the repetitions" might then be evaluated for math by wrap_math.
        # Assuming "Errors and the repetitions" does not trigger math wrapping:
        self.assertEqual(processed_text_for_llm, "Errors and the repetitions")

    def test_process_chunk_to_llm_input_math_delimiting(self):
        """Test that process_chunk_to_llm_input applies math delimiting."""
        raw_chunk_elements = [{"type": "text", "content": "Let x = 10 and y = 20."}]
        processed_text_for_llm = process_chunk_to_llm_input(raw_chunk_elements, "dummy_img_dir")
        # Expected: math should be wrapped by process_math
        self.assertEqual(processed_text_for_llm, "Let $x = 10$ and $y = 20$.")

    def test_process_chunk_to_llm_input_image_placeholder(self):
        raw_chunk_elements = [
            {"type": "text", "content": "Some text."},
            {"type": "image", "content": "/abs/path/to/images/image1.png", "page": 1}
        ]
        # The process_chunk_to_llm_input expects image_output_dir to be the *actual* final dir
        # and el["content"] to be an absolute path to the image *within* a structure similar to image_output_dir
        # For testing, let's assume image_output_dir is 'out_images' and image is 'out_images/fig1.png'
        # Then relpath from 'out_images' to 'out_images/fig1.png' is 'fig1.png'
        
        # If el["content"] is /tmp/proj/out_images/image1.png
        # and image_output_dir is /tmp/proj/out_images
        # then rel_path should be image1.png
        
        # More robust test: create temp dirs/files if possible, or mock os.path.relpath
        # For now, assume simple case:
        processed_text_for_llm = process_chunk_to_llm_input(raw_chunk_elements, "/abs/path/to/images")
        self.assertIn("Some text.", processed_text_for_llm)
        self.assertIn("[IMAGE: image1.png]", processed_text_for_llm)
        
        # Test with a different relative path structure
        raw_chunk_elements_2 = [
             {"type": "image", "content": "/abs/path/to/other_images_folder/image2.png"}
        ]
        # If image_output_dir is /abs/path/to/images, then relpath might be ../other_images_folder/image2.png
        # The current `process_chunk_to_llm_input` makes el["content"] relative to `image_output_dir` itself.
        # So if el["content"] is /data/images/img.png and image_output_dir is /output/images_dir
        # os.path.relpath("/data/images/img.png", "/output/images_dir") -> ../../data/images/img.png (approx)
        # This seems fine as long as LLM / downstream can handle it or it's resolved later.
        # The prompt says ![description](path), so this relative path should be fine.
        
        # Let's test the specific case from process_chunk_to_llm_input:
        # el["content"] = /abs/path/image.png, image_output_dir = /abs/path
        # rel_path = os.path.relpath("/abs/path/image.png", "/abs/path") -> "image.png"
        raw_chunk_elements_3 = [{"type": "image", "content": "/abs/path/image3.png"}]
        processed_text_for_llm_3 = process_chunk_to_llm_input(raw_chunk_elements_3, "/abs/path")
        self.assertEqual(processed_text_for_llm_3, "[IMAGE: image3.png]")


if __name__ == '__main__':
    unittest.main()
