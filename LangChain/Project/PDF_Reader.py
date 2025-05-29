from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.agents import Tool, initialize_agent, AgentType
from dotenv import load_dotenv
# import streamlit as st

load_dotenv()

# Document Loader
loader = PyPDFLoader('llm-tutorial.pdf')
docs = loader.load()

# Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embeddings Model and Vector Stores
embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
vector_store = FAISS.from_documents(chunks, embedding_model)

# Retriever
retriever = vector_store.as_retriever(search_type='similarity', search_tool={'k':4})

# Prompt
prompt = PromptTemplate(
    template="""
You are a helpful AI assistant.

Use the provided context below to answer the question.

If the context is missing or does not contain enough information to answer, respond exactly with:
INSUFFICIENT CONTEXT

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# LLM 
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.2)

# Output Parser (Requesting Output as a string)
parser = StrOutputParser()

# Query
question = "Can you summarize it"
retrieved_doc = retriever.invoke(question)

def format_docs(retrieved_doc):
  context_text = " ".join(doc.page_content for doc in retrieved_doc)
  return context_text

# Tool
search_tool = Tool(
    name='Search',
    func= lambda q: DuckDuckGoSearchRun().invoke(q),
    description="Search the web for current or missing information."
)

# AI Agent
agent = initialize_agent(
    tools = [search_tool],
    llm = llm,   
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
    handle_parsing_errors = True  # This allows the agent to retry if parsing fails.
)

# Chains
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()    
})

main_chain = parallel_chain | prompt | llm | parser

# Invokes
main_chain.invoke('Introduction to LLMs')


response = main_chain.invoke("Introduction to LLMs")

if "insufficient_context" in response.lower() or "don't have enough information" or "i don't know" in response.lower():
    print("Fallback Triggered: Using Available AI Agent for external info... ")
    final_response = agent.invoke("Introduction to LLMs")
else:
    final_response = response

print(final_response)