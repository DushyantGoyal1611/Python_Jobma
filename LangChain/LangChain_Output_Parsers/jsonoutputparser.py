# Using Output Parser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give me the name, age and city of a random fictional character \n {format_instruction}',
    input_variables= [],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})
print(result)

"""
Limitation:-
Doesn't enforce Schema, means it always doesn't generate json schema as user expected.
This problem is solved in structured output parser.
"""