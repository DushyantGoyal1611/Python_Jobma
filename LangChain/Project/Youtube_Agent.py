from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st
import re

load_dotenv()

# Function to extract video id from the youtube video link
def extract_video_id(input_text):
    # If it's a raw video ID
    if re.fullmatch(r'[0-9A-Za-z_-]{11}', input_text):
        return input_text

    # Try to extract from standard YouTube URLs
    regex_patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",                 # Matches standard watch URLs and embedded
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",             # Short links
        r"(?:youtube\.com\/shorts\/)([0-9A-Za-z_-]{11})"   # Shorts
    ]

    for pattern in regex_patterns:
        match = re.search(pattern, input_text)
        if match:
            return match.group(1)

    return None  # Return None if no match found


st.set_page_config(page_title="Youtube Agent", layout='centered')
st.title("Transcript Chatter")

# Document Loader
video_input = st.text_input("Enter YouTube video URL or ID")
video_id = extract_video_id(video_input)

if video_id:
    try:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            if not transcript_list:
                raise ValueError("Transcript is empty or not available.")
        except NoTranscriptFound:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            for t in transcripts:
                if t.is_generated and t.is_translatable:
                    transcript_list = t.translate('en').fetch()
                    break
        if transcript_list is None:
            st.error("No usable transcript available, and auto-translation to English failed.")
        else:
            # Flatten the script 
            # transcript = " ".join(chunk['text'] for chunk in transcript_list)
            transcript = " ".join(chunk.text for chunk in transcript_list)

        # Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )

        chunks = text_splitter.create_documents([transcript])

        # Embeddings and Vector Stores
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(chunks, embedding_model)

        # Retriever
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

        # Prompt
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
            """,
            input_variables= {'context','question'}
        )

        # LLM
        llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.2)

        # Query
        # retrieved_docs = retriever.invoke(user_question)  # Bcz in flowchart (query -> retriever)

        def format_docs(retrieved_docs):
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text

        parallel_chain = RunnableParallel({
            'context' : retriever | RunnableLambda(format_docs),
            'question' : RunnablePassthrough()
        })

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

        st.success("Transcript Loaded, Now you can ask the question")

        user_question = st.text_input('Ask a question')
        if user_question:
            with st.spinner("Generating answer..."):
                answer = main_chain.invoke(user_question)
                st.markdown('### Answer')
                st.write(answer)

    except TranscriptsDisabled as e:
        st.error("No captions available for this video.")
    except Exception as e:
        st.error(f'Error in Transcript: {e}')
else:
    st.info("Please enter a YouTube video URL or ID")
