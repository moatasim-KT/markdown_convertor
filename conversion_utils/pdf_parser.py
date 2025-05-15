import fitz  # PyMuPDF
import os
from typing import List, Dict, Any
from PIL import Image
import pytesseract
import io

def save_image_from_pixmap(pix, page_num, image_output_dir, idx=0):
    image_filename = f"page{page_num+1}_ocr_img{idx+1}.png"
    image_path = os.path.join(image_output_dir, image_filename)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save(image_path)
    return image_path

def robust_block_to_element(block, page, page_num, page_width, page_height, image_output_dir, ocr_img_idx):
    # BBox recovery
    bbox = []
    for i in range(4):
        try:
            bbox.append(float(block[i]))
        except Exception:
            bbox.append(0.0 if i < 2 else (page_width if i == 2 else page_height))
    bbox = bbox[:4]
    # Text recovery
    text = ""
    if len(block) > 4 and isinstance(block[4], str) and block[4].strip():
        text = block[4].strip()
        return {
            "type": "text",
            "content": text,
            "bbox": bbox,
            "page": page_num + 1
        }
    else:
        # Try OCR on the region
        try:
            pix = page.get_pixmap(clip=fitz.Rect(*bbox))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                return {
                    "type": "text",
                    "content": ocr_text,
                    "bbox": bbox,
                    "page": page_num + 1
                }
            else:
                # Save as image if OCR fails
                image_path = save_image_from_pixmap(pix, page_num, image_output_dir, ocr_img_idx)
                return {
                    "type": "image",
                    "content": image_path,
                    "bbox": bbox,
                    "page": page_num + 1
                }
        except Exception:
            # If pixmap or OCR fails, fallback to page bbox and save as image
            pix = page.get_pixmap()
            image_path = save_image_from_pixmap(pix, page_num, image_output_dir, ocr_img_idx)
            return {
                "type": "image",
                "content": image_path,
                "bbox": bbox,
                "page": page_num + 1
            }

def extract_elements(pdf_path: str, image_output_dir: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    os.makedirs(image_output_dir, exist_ok=True)
    elements = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_width, page_height = page.rect.width, page.rect.height
        blocks = page.get_text("blocks")
        ocr_img_idx = 0
        for block in blocks:
            el = robust_block_to_element(block, page, page_num, page_width, page_height, image_output_dir, ocr_img_idx)
            if el["type"] == "image":
                ocr_img_idx += 1
            elements.append(el)
        # Extract images as before
        img_list = page.get_images(full=True)
        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            image_filename = f"page{page_num+1}_img{img_index+1}.{ext}"
            image_path = os.path.join(image_output_dir, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            bbox = [float(x) if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit() else 0.0 for x in img[5]]
            elements.append({
                "type": "image",
                "content": image_path,
                "bbox": bbox,
                "page": page_num + 1
            })
        # Sort elements by vertical position (y0)
        elements_on_page = [el for el in elements if el["page"] == page_num + 1]
        elements_on_page.sort(key=lambda el: el["bbox"][1])
        elements = [el for el in elements if el["page"] != page_num + 1] + elements_on_page
    return elements
