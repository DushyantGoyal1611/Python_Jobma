# Using Output Parser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# Pydantic Class for Data Validation
class Person(BaseModel):
    name : str = Field(description='Name of the Person')
    age : int = Field(ge=18, description='Age of the Person')
    city : str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Generate the name, age and city of a fictional person who lives in {place} \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place':'india'})

print(result)