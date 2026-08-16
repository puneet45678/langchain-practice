from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

messages = [
    SystemMessage(content="You are a helpful assistant that replies concisely."),
    HumanMessage(content="What is LangChain in one sentence?"),
]

response = llm.invoke(messages)
print(response.content)
