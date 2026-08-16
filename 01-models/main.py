from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ============================================================
# 1. CHAT MODEL — message list in, AIMessage out
# ============================================================
print("--- Chat Model ---")

llm = ChatOpenAI(model="gpt-4o-mini")

messages = [
    SystemMessage(content="You are a helpful assistant that replies concisely."),
    HumanMessage(content="What is LangChain in one sentence?"),
]

response = llm.invoke(messages)
print(response.content)
print(f"Response type: {type(response)}\n")  # AIMessage


# ============================================================
# 2. MULTI-TURN — memory via message list (Video 3 key concept)
# ============================================================
print("--- Multi-turn (memory via message list) ---")

history = [SystemMessage(content="You are a helpful assistant.")]

# Turn 1
history.append(HumanMessage(content="My name is Puneet."))
reply = llm.invoke(history)
history.append(reply)  # add AIMessage back so next call has context
print(f"Turn 1: {reply.content}")

# Turn 2 — model remembers
history.append(HumanMessage(content="What is my name?"))
reply = llm.invoke(history)
print(f"Turn 2: {reply.content}\n")


# ============================================================
# 3. EMBEDDING MODEL — text in, vector out
# ============================================================
print("--- Embedding Model ---")

embedder = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embedder.embed_query("What is LangChain?")
print(f"Vector dimensions: {len(vector)}")
print(f"First 5 values: {vector[:5]}")
