from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
parser = StrOutputParser()
prompt = PromptTemplate(
    template= 'Read the Text and find the smell of what \n {poem}',
    input_variables=['poem']
)

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'poem':docs[0].page_content})
print(result)