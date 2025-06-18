import fitz  # apparently pymupdf is faster and better than pypdf so im using it
from PIL import Image
import pytesseract
import io  # to treat png bytes like file bytestream


def ocr(pdf_path, zoom=2.0):  # zoom is basically resolution higher values increased page damage errors
    doc = fitz.open(pdf_path)
    all_text = ""

    for i in range(len(doc)):
        page = doc.load_page(i)
        mat = fitz.Matrix(zoom, zoom)  # zoom scale to increase resolution
        pix = page.get_pixmap(matrix=mat)

        # converting the pixmap to png bytes and then create file buffer using bytesio for opening using pillow
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # ocr
        text = pytesseract.image_to_string(img)  # default config itself was good enough
        all_text += f"\n\n--- Page {i+1} ---\n\n{text}"

    doc.close()
    return all_text


pdf_text = ocr("the_pdf.pdf")
pdf_text = pdf_text.strip()
print(pdf_text)

with open("the_pdf.txt", "w") as f:
    f.write(pdf_text)
