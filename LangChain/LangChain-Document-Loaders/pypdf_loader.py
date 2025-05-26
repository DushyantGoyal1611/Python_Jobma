from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
parser = StrOutputParser()

prompt = PromptTemplate(
    template='Generate a summary on the given topic \n {topic}',
    input_variables=['topic']
)

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()
# Just to combine all pages into a single page
text = '\n'.join(doc.page_content for doc in docs)

chain = prompt | model | parser
result = chain.invoke({'topic':text})
print(result)
