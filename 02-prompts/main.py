from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.example_selectors import LengthBasedExampleSelector

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


# ============================================================
# 1. RAW MESSAGES — no template, manual list (you already know this)
# ============================================================
print("=" * 50)
print("1. Raw Messages")
print("=" * 50)

messages = [
    SystemMessage(content="You reply in exactly one sentence."),
    HumanMessage(content="What is LangChain?"),
]
response = llm.invoke(messages)
print(response.content)


# ============================================================
# 2. PromptTemplate — simple string template (legacy / non-chat LLMs)
# ============================================================
print("\n" + "=" * 50)
print("2. PromptTemplate (simple string)")
print("=" * 50)

template = PromptTemplate.from_template(
    "Tell me a {adjective} fact about {topic}."
)
prompt = template.invoke({"adjective": "surprising", "topic": "Python"})
print("Rendered prompt:", prompt.text)

# pass as a single human message to chat model
response = llm.invoke(prompt.text)
print("Response:", response.content)


# ============================================================
# 3. ChatPromptTemplate — structured messages with variables
# ============================================================
print("\n" + "=" * 50)
print("3. ChatPromptTemplate")
print("=" * 50)

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Reply in 2 sentences max."),
    ("human", "{question}")
])

messages = chat_template.invoke({
    "domain": "distributed systems",
    "question": "What is eventual consistency?"
})

response = llm.invoke(messages)
print(response.content)


# ============================================================
# 4. MessagesPlaceholder — inject chat history (memory pattern)
# ============================================================
print("\n" + "=" * 50)
print("4. MessagesPlaceholder (chat history / memory)")
print("=" * 50)

memory_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),   # dynamic history injected here
    ("human", "{question}")
])

# Simulate a conversation history
history = [
    HumanMessage(content="My favourite language is Python."),
    AIMessage(content="Great choice! Python is very versatile."),
]

messages = memory_template.invoke({
    "history": history,
    "question": "What is my favourite language?"
})

response = llm.invoke(messages)
print(response.content)  # should mention Python


# ============================================================
# 5. FewShotChatMessagePromptTemplate — guide with examples
# ============================================================
print("\n" + "=" * 50)
print("5. FewShot Prompting")
print("=" * 50)

examples = [
    {"input": "happy",    "output": "sad"},
    {"input": "tall",     "output": "short"},
    {"input": "fast",     "output": "slow"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai",    "{output}")
])

few_shot = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

final_template = ChatPromptTemplate.from_messages([
    ("system", "Give the opposite of the word the user provides. Reply with just one word."),
    few_shot,       # examples injected here
    ("human", "{word}")
])

messages = final_template.invoke({"word": "bright"})
response = llm.invoke(messages)
print(f"Opposite of 'bright': {response.content}")


# ============================================================
# 6. PromptTemplate CLASS (explicit) — declare variables upfront
# ============================================================
print("\n" + "=" * 50)
print("6. PromptTemplate class (explicit)")
print("=" * 50)

explicit_template = PromptTemplate(
    template="Write a {tone} summary of {topic} in {language}.",
    input_variables=["tone", "topic", "language"]
)

prompt = explicit_template.invoke({
    "tone": "simple",
    "topic": "how LangChain works",
    "language": "English"
})
print("Rendered:", prompt.text)
response = llm.invoke(prompt.text)
print("Response:", response.content)


# ============================================================
# 7. PARTIAL PROMPTS — pre-fill some variables, fill rest later
# ============================================================
print("\n" + "=" * 50)
print("7. Partial Prompts")
print("=" * 50)

# A) Fill some variables at "startup"
base_template = PromptTemplate(
    template="You are a {role}. Answer this {language} question: {question}",
    input_variables=["role", "language", "question"]
)

# Pre-fill role + language (fixed for this app)
partial = base_template.partial(role="senior Python developer", language="Python")

# At request time — only need to provide question
prompt = partial.invoke({"question": "What is a list comprehension?"})
print("Partial prompt:", prompt.text)
response = llm.invoke(prompt.text)
print("Response:", response.content)

# B) Callable partial — auto-injects dynamic value (e.g. current date)
print()
date_template = PromptTemplate(
    template="Today is {date}. In one sentence, answer: {question}",
    input_variables=["question"],
    partial_variables={"date": lambda: datetime.now().strftime("%Y-%m-%d")}
)

prompt = date_template.invoke({"question": "What is today's date?"})
print("With auto date:", prompt.text)
response = llm.invoke(prompt.text)
print("Response:", response.content)


# ============================================================
# 8. DYNAMIC FEW-SHOT — auto-select examples based on token budget
# ============================================================
print("\n" + "=" * 50)
print("8. Dynamic Few-Shot (LengthBasedExampleSelector)")
print("=" * 50)

examples = [
    {"input": "happy",    "output": "sad"},
    {"input": "tall",     "output": "short"},
    {"input": "fast",     "output": "slow"},
    {"input": "bright",   "output": "dim"},
    {"input": "loud",     "output": "quiet"},
    {"input": "ancient",  "output": "modern"},
]

# LengthBasedExampleSelector needs a simple PromptTemplate (not Chat)
selector_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="input: {input}\noutput: {output}"
)

chat_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai",    "{output}")
])

# Automatically picks as many examples as fit within max_length words
selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=selector_prompt,
    max_length=10
)

dynamic_few_shot = FewShotChatMessagePromptTemplate(
    example_selector=selector,
    example_prompt=chat_example_prompt,
)

dynamic_template = ChatPromptTemplate.from_messages([
    ("system", "Give the opposite of the word. Reply with one word only."),
    dynamic_few_shot,
    ("human", "{word}")
])

messages = dynamic_template.invoke({"word": "cold"})
response = llm.invoke(messages)
print(f"Opposite of 'cold': {response.content}")
