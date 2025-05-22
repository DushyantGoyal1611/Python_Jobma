# Using Output Parser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# 1st prompt -> Detailed Report
template1 = PromptTemplate(
    template='Create a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> Summary 
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

# Output Parser
parser = StrOutputParser()

# Chaining
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)
