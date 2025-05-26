"""
Performs Recursive operation until it reaches the required chunk size
first -> \n\n (for Paragraph),
second -> \n (for line),
third -> ' ' (for word),
fourth -> '' (for character)
"""


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0
)

chunks = splitter.split_documents(docs)
# print(len(chunks))
print(chunks)