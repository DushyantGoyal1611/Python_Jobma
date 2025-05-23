# Using Output Parser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the Topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the Topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the Topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template= "Give 3 facts about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})
print(result)

"""
Limitations:-
Doesn't provide Data Validation

In LangChain’s StructuredOutputParser, the data validation problem refers to how strictly and reliably the model-generated output matches the schema you define.

When the output:

Doesn’t follow the exact JSON format

Misses required fields

Has wrong types (e.g., string instead of int)
LangChain raises a OutputParserException.
"""