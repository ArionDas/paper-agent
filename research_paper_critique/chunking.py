import os
import fitz
import pdfplumber
from PIL import Image
import pytesseract

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configuration
UPLOAD_FOLDER = 'pdf_uploads'
IMAGE_FOLDER = 'extracted_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

def extract_tables(pdf_path):
    """
    Extracts tables from the PDF and returns as list of Document objects.
    """
    documents = []
    try:
        print(f"Extracting tables from {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_num, table in enumerate(tables):
                    rows = ["\t".join([cell or '' for cell in row]) for row in table]
                    table_text = "\n".join(rows)
                    print(table_text)
                    if table_text.strip():
                        # Create Document object with metadata
                        doc = Document(
                            page_content=table_text, 
                            metadata={"source": pdf_path, "type": "table", "page": page_num, "table": table_num}
                        )
                        documents.append(doc)
        print("Table extraction complete.")
    except Exception as e:
        print(f"Error extracting tables: {e}")
    return documents


def extract_images_and_ocr(pdf_path):
    """
    Extracts images from the PDF, saves them locally, OCRs text, and returns list of Document objects.
    """
    documents = []
    try:
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
                        # Create Document object with metadata
                        doc_obj = Document(
                            page_content=text, 
                            metadata={"source": pdf_path, "type": "image", "page": page_index, "image": image_name}
                        )
                        documents.append(doc_obj)
                except Exception as e:
                    print(f"Error OCRing image {image_path}: {e}")
    except Exception as e:
        print(f"Error extracting images: {e}")
    return documents


def split_recursive_text(resume_path):
    loader = PyPDFLoader(resume_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size = 1000,
        chunk_overlap=500,
    )

    texts = text_splitter.split_documents(documents)
    
    # Appending text from tables
    texts += extract_tables(resume_path)

    # Extracting images and OCR text
    image_texts = extract_images_and_ocr(resume_path)
    texts += image_texts

    return texts