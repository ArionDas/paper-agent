import os
import fitz
import pdfplumber
from PIL import Image
import pytesseract

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
UPLOAD_FOLDER = 'pdf_uploads'
IMAGE_FOLDER = 'extracted_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

def extract_tables(pdf_path):
    """
    Extracts tables from the PDF and returns as list of strings.
    """
    texts = []
    try:
        print(f"Extracting tables from {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    rows = ["\t".join([cell or '' for cell in row]) for row in table]
                    table_text = "\n".join(rows)
                    print(table_text)
                    if table_text.strip():
                        texts.append(table_text)
        print("Table extraction complete.")
    except Exception:
        pass
    return texts


def extract_images_and_ocr(pdf_path):
    """
    Extracts images from the PDF, saves them locally, OCRs text, and returns list of OCR'd strings.
    """
    image_texts = []
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = base_image.get("ext", "png")
            image_name = f"page{page_index+1}_img{img_index+1}.{ext}"
            image_path = os.path.join(IMAGE_FOLDER, image_name)
            with open(image_path, 'wb') as img_file:
                img_file.write(img_bytes)
            # OCR the image
            try:
                pil_img = Image.open(image_path)
                text = pytesseract.image_to_string(pil_img)
                if text.strip():
                    image_texts.append(text)
            except Exception:
                pass
    return image_texts

def split_recursive_text(resume_path):
    loader = PyPDFLoader(resume_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size = 1000,
        chunk_overlap=500,
    )

    texts = text_splitter.split_documents(documents)
    
    # Append table text
    texts += extract_tables(resume_path)

    # Extract images and OCR text
    image_texts = extract_images_and_ocr(resume_path)
    texts += image_texts

    #texts = re.sub(r'<.*?>', '', texts)

    return texts