from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator=''
)

chunk = splitter.split_documents(docs)
print(len(chunk))
# print(chunk[0].page_content) 