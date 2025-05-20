# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceHub(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 100}
)

# model = ChatHuggingFace(llm=llm)

result = llm.invoke("What is the Capital of India")
print(result)