import os
import re
import warnings
from dotenv import load_dotenv

# LangChain related libraries
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Gemini
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.agents import Tool, initialize_agent, AgentType

# Memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Code starts from here

warnings.filterwarnings('ignore')
load_dotenv()


# Tool
@tool
def calculation(a:int):
    """This function will solve any mathematical query"""
    return a**2

# Prompt
prompt = PromptTemplate(
    template="""
You are a highly capable AI assistant trained to read and extract information from various document types, including .pdf, .docx, and .txt formats.

The documents may contain:
- Headings and subheadings (especially in PDFs and DOCX)
- Bullet points, numbered lists, and paragraphs
- Tables or structured content (DOCX, PDF)
- Plain text with minimal formatting (.txt)

Use the context provided below to answer the question **as accurately and concisely as possible**.

If the context is irrelevant, insufficient, or does not contain the answer, respond only with:
INSUFFICIENT CONTEXT
And the question from user should only be related to document, if not then reply with:
"SORRY: This question is irrelavant."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# Document Loader
def extract_document(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding="utf-8")
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

# RAG WorkFlow
def rag_flow(doc, prompt):
    # Document Loader
    docs = extract_document(doc)
    if not docs:
        raise ValueError("Document could not be loaded.")
    # Text Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # Model Embeddings and Vector Store
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(chunks, embedding_model)
    # Retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_tool={'k':4})
    # Model
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    # Parser
    parser = StrOutputParser()

    def format_docs(retrieved_docs):
        return "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()    
    })

    main_chain = parallel_chain | prompt | llm | parser
    return main_chain

# The Chatbot
def ask_ai():
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    chain = rag_flow('formatted_QA.txt', prompt)
    
    # Memory
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(llm=llm, memory=memory)

    while True:
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break

        response = chain.invoke(user_input)

        fallback_triggers = r"(insufficient|not (sure|enough|understand)|i don't know|no context)"
        if re.search(fallback_triggers, response, re.IGNORECASE):
            print("Fallback Triggered: Using Available AI Agent for external info... ")
            final_response = conversation.predict(input=user_input)
            print(f"AI (Fallback): {final_response}")
        else:
            print(f"AI: {response}")

    print("Chat History:")
    for msg in memory.buffer:
        role = "Human" if msg.type == "human" else "AI"
        print(f"{role}: {msg.content}")


# Calling the Chatbot
ask_ai()