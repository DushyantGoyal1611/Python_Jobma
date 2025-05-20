from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

documents = [
    "Virat Kohli is a former captain of the Indian cricket team and one of the world's top batsmen.",
    "Rohit Sharma is the current captain of the Indian cricket team and known for his explosive batting.",
    "Jasprit Bumrah is a fast bowler known for his deadly yorkers and consistency in all formats.",
    "Ravindra Jadeja is an all-rounder who contributes with both bat and ball, and is a brilliant fielder.",
    "KL Rahul is a stylish right-handed batsman who has played in all formats for India."
]

query = "Tell me about jasprit bumrah"

document_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

# Calculating Similarity between Document and Query
print(cosine_similarity([query_embeddings], document_embeddings))