import os
import re
import warnings
from dotenv import load_dotenv
# LangChain related libraries
# RAG Libraries
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# Memory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# LLM
# llm = Ollama(model="llama3.2")

# Prompt
prompt = PromptTemplate(
    template="""
You are an intelligent assistant that only answers questions based on the provided document content.

The document may include:
- Headings, paragraphs, subheadings
- Lists or bullet points
- Tables or structured data
- Text from PDF, DOCX, or TXT formats

Your responsibilities:
1. Use ONLY the content in the document to answer.
2. If the question is clearly related to the document topic but the content is insufficient, respond with: INSUFFICIENT CONTEXT.
3. If the question is completely unrelated to the document, respond with: SORRY: This question is irrelevant.
4. If the user greets (e.g., "hi", "hello"), respond with a friendly greeting.
5. Otherwise, provide a concise and accurate answer using only the document content.

Document Content:
{context}

User Question:
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
        print(f'Loading {file_path} .....')
        print("Loading Successful")

        return docs
    except FileNotFoundError as fe:
        print(f"File not found: {fe}")
    except Exception as e:
        print(f"Error loading document: {e}")

    return []

# RAG WorkFlow
def create_rag_chain(doc, prompt):
    # Document Loader
    docs = extract_document(doc)
    if not docs:
        raise ValueError("Document could not be loaded.")
    # Text Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # Model Embeddings and Vector Store
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_documents(chunks, embedding_model)
    # LLM
    llm = Ollama(model="llama3.2")
    # Retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})
    # Output Parser
    parser = StrOutputParser()

    def format_docs(retrieved_docs):
        return "\n\n".join([doc.page_content for doc in retrieved_docs])
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | parser
    return main_chain

# Chatbot
def ask_ai():
    # LLM
    llm = Ollama(model="llama3.2")

    chat_history = [SystemMessage(content="You are a helpful AI Assistant")]
    chain = create_rag_chain('formatted_QA.txt', prompt)
    fallback_triggers = r"(insufficient|not (sure|enough|understand)|i don't know|no context)"

    while True:
        user_input = input("You: ")
        if user_input in ['exit', 'quit']:
            print('Exiting the Chat.\nGoodbye!')
            break
        
        if chain:
             response = chain.invoke(user_input)
        else:
            response = "I don't have access to the knowledge base. Please ask general questions or schedule an interview."

        if re.search(fallback_triggers, response, re.IGNORECASE):
            print("Fallback Triggered: Using AI for external info... ")
            chat_history.append(HumanMessage(content=user_input))
            final_response = llm.invoke(chat_history)

            chat_history.append(final_response)
            print(f"AI (Fallback): {final_response.content}")
        else:
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))
            print(f"AI: {response}")

    # Chat History
    print("\n--- Chat History ---")
    for chat in chat_history:
        if isinstance(chat, HumanMessage):
            role = "User"
        elif isinstance(chat, AIMessage):
            role = "AI"
        else:
            role = "System"
        print(f"{role}: {chat.content}")

ask_ai()