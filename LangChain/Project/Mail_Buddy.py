import email.utils
import os, warnings, base64, email
from email.message import EmailMessage
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dotenv import load_dotenv

# LangChain
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


# Code starts from here --------------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# Prompt
prompt = PromptTemplate(
    template="""
    You are an email-writing assistant.
    Write a reply in the **same tone** as the original e-mail.
    Keep it short, polite, and actionable.

    Original Email:
    {context}

    Draft Reply:
""",
input_variables=["context"]
)

# The RAG Workflow
def rag_chain(email_txt:str, prompt):
    # Document Loader
    loader = UnstructuredEmailLoader(email_txt)
    docs = loader.load()
    # Text Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # Model Embedding and Vector Store
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    # Retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})
    # LLM
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    # Parser
    parser = StrOutputParser()

    def format_docs(retrieved_docs):
        return "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Chaining
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()    
    })

    main_chain = parallel_chain | prompt | llm | parser

    return main_chain

def create_message(reply_to: str, subject: str, body: str):
    msg = EmailMessage()
    msg['To'] = reply_to
    msg['Subject'] = "Re: " + subject
    msg.set_content(body)
    raw_bytes = msg.as_bytes()  # raw email bytes
    raw_b64 = base64.urlsafe_b64encode(raw_bytes).decode('utf-8')  # encode bytes to base64 string
    return {"raw": raw_b64}
