import os
import warnings
from dotenv import load_dotenv

# LangChain related libraries
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Code starts from here

warnings.filterwarnings('ignore')
load_dotenv()

# Document Loader
def extract_document(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")

        docs = loader.load()
        print(f'Loading {file_path} ......')
        print('Loading Successful')
        return docs

    except FileNotFoundError as fe:
        print(f"File not found: {fe}")
    except Exception as e:
        print(f"Error loading document: {e}")
    
    return []

docs = extract_document('llm-tutorial.pdf')

# Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embeddings Model and Vector Stores
embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
vector_store = FAISS.from_documents(chunks, embedding_model)

# Retriever
retriever = vector_store.as_retriever(search_type='similarity', search_tool={'k':4})

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
# Output Parser
parser = StrOutputParser()

def format_docs(retrieved_doc):
    context_text = " ".join(doc.page_content for doc in retrieved_doc)
    return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()    
})

main_chain = parallel_chain | prompt | llm | parser

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
    verbose = False,
    handle_parsing_errors = True  # This allows the agent to retry if parsing fails.
)

# Chatbot

chat_history = [SystemMessage(content="You are a helpful AI assistant.")]

while True:
    user_input = input('You:')
    if user_input.lower()  == 'exit':
        break

    response = main_chain.invoke(user_input)
    fallback_triggers = ["insufficient", "context", "not sure", "i don't know", "not enough information"]
    if any(kw in response.lower() for kw in fallback_triggers):
        print("Fallback Triggered: Using Available AI Agent for external info... ")
        final_response = agent.invoke(HumanMessage(content=user_input))
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=final_response["output"]))
        print(f"AI (Fallback): {final_response['output']}")
    else:
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print(f"AI: {response}")

print("Chat History:")
print(chat_history)
