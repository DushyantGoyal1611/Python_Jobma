from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
parser = StrOutputParser()

# prompt = PromptTemplate(
#     template = 'Pick a random page and give a summary about that topic \n {topic}',
#     input_variables=['topic']
# )

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# chain = prompt | model | parser

docs = loader.load()
print(docs[1].page_content)
print(docs[1].metadata)
# print(chain.invoke({'topic':docs.page_content}))