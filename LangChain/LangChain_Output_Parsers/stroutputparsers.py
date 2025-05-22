#  Without using Output Parser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
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

prompt1 = template1.invoke({'topic':'black hole'})
result1 = model.invoke(prompt1 )

prompt2 = template2.invoke({'text':result1.content})
result2 = model.invoke(prompt2)

# Result
print(result2.content)