from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


# ============================================================
# 1. SEQUENTIAL CHAIN — chain1 output feeds chain2
# ============================================================
print("=" * 50)
print("1. Sequential Chain")
print("=" * 50)

# Step 1: summarise a topic
summarise = (
    ChatPromptTemplate.from_messages([
        ("human", "Summarise {topic} in exactly one sentence.")
    ])
    | llm
    | StrOutputParser()
)

# Step 2: translate that summary
translate = (
    ChatPromptTemplate.from_messages([
        ("human", "Translate this to {language}, just the translation, no extra text:\n{text}")
    ])
    | llm
    | StrOutputParser()
)

# Wire them: summary becomes {text} for the translate chain
sequential = (
    RunnableParallel({
        "text":     summarise,
        "language": RunnablePassthrough() | RunnableLambda(lambda x: x["language"]),
    })
    | translate
)

result = sequential.invoke({"topic": "machine learning", "language": "Hindi"})
print(f"Translated summary: {result}\n")


# ============================================================
# 2. PARALLEL CHAIN — multiple chains on same input, simultaneously
# ============================================================
print("=" * 50)
print("2. Parallel Chain (RunnableParallel)")
print("=" * 50)

pros_chain = (
    ChatPromptTemplate.from_messages([
        ("human", "List 2 pros of {topic} in bullet points.")
    ])
    | llm | StrOutputParser()
)

cons_chain = (
    ChatPromptTemplate.from_messages([
        ("human", "List 2 cons of {topic} in bullet points.")
    ])
    | llm | StrOutputParser()
)

summary_chain = (
    ChatPromptTemplate.from_messages([
        ("human", "Give a neutral one-sentence summary of {topic}.")
    ])
    | llm | StrOutputParser()
)

# All 3 run at the SAME TIME — merged into a dict
parallel = RunnableParallel({
    "pros":    pros_chain,
    "cons":    cons_chain,
    "summary": summary_chain,
})

result = parallel.invoke({"topic": "remote work"})
print("Summary:", result["summary"])
print("Pros:\n", result["pros"])
print("Cons:\n", result["cons"])


# ============================================================
# 3. RunnablePassthrough — carry original input alongside
# ============================================================
print("=" * 50)
print("3. RunnablePassthrough")
print("=" * 50)

answer_chain = (
    ChatPromptTemplate.from_messages([("human", "Answer briefly: {question}")])
    | llm | StrOutputParser()
)

# Pass original question through AND generate answer, both in output
chain = RunnableParallel({
    "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
    "answer":   answer_chain,
})

result = chain.invoke({"question": "What is a vector database?"})
print(f"Q: {result['question']}")
print(f"A: {result['answer']}\n")


# ============================================================
# 4. RunnableLambda — custom Python function in the chain
# ============================================================
print("=" * 50)
print("4. RunnableLambda (custom function)")
print("=" * 50)

def enrich_input(input: dict) -> dict:
    # pre-process: add a difficulty level based on topic length
    input["level"] = "advanced" if len(input["topic"]) > 10 else "beginner"
    return input

chain = (
    RunnableLambda(enrich_input)
    | ChatPromptTemplate.from_messages([
        ("human", "Explain {topic} for a {level} audience in one sentence.")
    ])
    | llm
    | StrOutputParser()
)

print(chain.invoke({"topic": "neural networks"}))   # advanced
print(chain.invoke({"topic": "AI"}))                 # beginner


# ============================================================
# 5. BRANCHING — route to different chains based on input
# ============================================================
print("\n" + "=" * 50)
print("5. Branching (RunnableBranch)")
print("=" * 50)

python_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a Python expert."),
        ("human", "Answer: {question}")
    ])
    | llm | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a general programming expert."),
        ("human", "Answer: {question}")
    ])
    | llm | StrOutputParser()
)

branch = RunnableBranch(
    # condition → chain to use
    (lambda x: "python" in x["question"].lower(), python_chain),
    # default fallback
    general_chain,
)

q1 = "What is a Python list comprehension?"
q2 = "What is a binary search tree?"

print(f"Q: {q1}")
print(f"A: {branch.invoke({'question': q1})}\n")

print(f"Q: {q2}")
print(f"A: {branch.invoke({'question': q2})}")


# ============================================================
# 6. ASCII GRAPH — visualise the chain structure
# ============================================================
print("\n" + "=" * 50)
print("6. ASCII Graph of each chain")
print("=" * 50)

print("\n--- Sequential chain ---")
sequential.get_graph().print_ascii()

print("\n--- Parallel chain ---")
parallel.get_graph().print_ascii()

print("\n--- Branch chain ---")
branch.get_graph().print_ascii()
