from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_recursive_text(resume_path):
  loader = PyPDFLoader(resume_path)
  documents = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
      separators=["\n"],
      chunk_size = 1000,
      chunk_overlap=300,
  )

  texts = text_splitter.split_documents(documents)

  #texts = re.sub(r'<.*?>', '', texts)

  return texts