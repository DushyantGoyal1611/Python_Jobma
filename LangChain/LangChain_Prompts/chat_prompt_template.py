from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain {topic} in simple terms')
])

prompt = chat_template.invoke({'domain':'cricket', 'topic':'LBW'})

print(prompt)