import pymupdf
import pytesseract
from PIL import Image

# Tesseract ka path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# PDF open karo
pdf_path = "documents/Industrial_training_fixed.pdf"
doc = pymupdf.open(pdf_path)

print("Total Pages:", len(doc))

all_text = []

# Har page ka OCR
for page_number, page in enumerate(doc, start=1):

    print(f"OCR processing: Page {page_number}/{len(doc)}")

    # PDF page ko high-quality image me convert karo
    pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3))

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    # OCR
    text = pytesseract.image_to_string(
        image,
        config="--psm 6",
        lang="eng"
    )

    all_text.append(text)

# Saara text ek file me save karo
output_path = "documents/industrial_training_ocr.txt"

with open(output_path, "w", encoding="utf-8") as file:
    for page_number, text in enumerate(all_text, start=1):
        file.write(f"\n\n===== PAGE {page_number} =====\n\n")
        file.write(text)

print("\nOCR completed successfully!")
print("Saved file:", output_path)