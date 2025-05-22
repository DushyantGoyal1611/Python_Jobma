from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Annotated

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash") 

class Review(BaseModel):
    summary: Annotated[str, 'A brief summary of the review']
    sentiment: Annotated[str, 'Return sentiment of the review either positive, negative or neutral']

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The Hardware is good, but the software feels bloated, there are too many pre-installed apps that I cant't remove. Also the UI looks outdated.
                      Hoping for a software update to fix this""")

result1 = structured_model.invoke("""The Hardware is good, but the software feels ok, there are too less pre-installed apps.
                                   Also the UI looks quite impressive.""")


print(result1)