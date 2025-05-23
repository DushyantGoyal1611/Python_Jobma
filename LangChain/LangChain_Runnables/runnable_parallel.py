from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a Tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Create a Linkeding Post about {topic}",
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

print(parallel_chain.invoke({'topic' : 'Hiring'}))