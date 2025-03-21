from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path):
    
    print("Loading PDF...")
    pdf_loader = PyPDFLoader(pdf_path)
    print("PDF successfully loaded!!")
    
    pages = pdf_loader.load()
    print("Extracted {} pages from PDF".format(len(pages)))
    
    return pages
