from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain this {text}',
    input_variables=['text']
)

joke_generator_chain = prompt1 | model | parser

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser, prompt2, model, parser)
})

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

print(final_chain.invoke({'topic': 'Chandler Bing'}))