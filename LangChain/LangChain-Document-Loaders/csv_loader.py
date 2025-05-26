from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
parser = StrOutputParser()

loader = CSVLoader('Social_Network_Ads.csv')

docs = loader.load()

print(docs[0])
print('===============================================================================')
print(docs[1])