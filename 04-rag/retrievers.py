import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# ============================================================
# SETUP — build a small vector store to practice with
# (save/load pattern so we don't re-embed every run)
# ============================================================
INDEX_PATH = "04-rag/faiss_index"

docs = [
    Document(page_content="LangChain is an open-source framework for building applications powered by large language models.", metadata={"source": "intro"}),
    Document(page_content="LCEL stands for LangChain Expression Language. It uses the pipe operator | to compose chains.", metadata={"source": "lcel"}),
    Document(page_content="RAG stands for Retrieval Augmented Generation. It lets LLMs answer questions from your own documents.", metadata={"source": "rag"}),
    Document(page_content="Agents in LangChain use the ReAct pattern — they reason about which tool to use and act step by step.", metadata={"source": "agents"}),
    Document(page_content="FAISS is an in-memory vector store by Facebook. It enables fast similarity search over embeddings.", metadata={"source": "faiss"}),
    Document(page_content="Chroma is a persistent vector store that saves data to disk automatically, unlike FAISS.", metadata={"source": "chroma"}),
    Document(page_content="Embeddings convert text into vectors. Similar meaning produces similar vectors measured by cosine similarity.", metadata={"source": "embeddings"}),
    Document(page_content="Text splitters break documents into chunks. RecursiveCharacterTextSplitter is the most commonly used.", metadata={"source": "splitters"}),
    Document(page_content="Output parsers clean LLM responses. StrOutputParser gives plain text, PydanticOutputParser gives typed objects.", metadata={"source": "parsers"}),
    Document(page_content="Prompt templates are reusable structures. ChatPromptTemplate is used with modern chat models.", metadata={"source": "prompts"}),
]

if os.path.exists(INDEX_PATH):
    db = FAISS.load_local(INDEX_PATH, embedder, allow_dangerous_deserialization=True)
    print("Loaded existing FAISS index\n")
else:
    print("Building FAISS index (embedding docs)...")
    db = FAISS.from_documents(docs, embedder)
    db.save_local(INDEX_PATH)
    print("Index built and saved\n")


# ============================================================
# 1. VectorStoreRetriever — basic similarity search
# ============================================================
print("=" * 50)
print("1. VectorStoreRetriever (similarity, k=3)")
print("=" * 50)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

results = retriever.invoke("What is LangChain?")
print(f"Query: 'What is LangChain?'  →  {len(results)} results")
for i, doc in enumerate(results):
    print(f"  [{i+1}] ({doc.metadata['source']}) {doc.page_content[:80]}...")


# ============================================================
# 2. Similarity with Score — see how confident the match is
# ============================================================
print("\n" + "=" * 50)
print("2. similarity_search_with_score")
print("=" * 50)

results_with_score = db.similarity_search_with_score("How do I split documents?", k=3)
print(f"Query: 'How do I split documents?'")
for doc, score in results_with_score:
    print(f"  Score: {score:.4f} | ({doc.metadata['source']}) {doc.page_content[:70]}...")
print("  Note: lower score = more similar in FAISS (uses L2 distance)")


# ============================================================
# 3. MMR Retriever — relevance + diversity
# ============================================================
print("\n" + "=" * 50)
print("3. MMR Retriever (relevance + diversity)")
print("=" * 50)

mmr_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 8,       # pull 8 candidates, pick 3 most diverse
        "lambda_mult": 0.5  # 0 = max diversity, 1 = max relevance
    }
)

results = mmr_retriever.invoke("Tell me about LangChain storage")
print(f"Query: 'Tell me about LangChain storage'  →  {len(results)} results")
for i, doc in enumerate(results):
    print(f"  [{i+1}] ({doc.metadata['source']}) {doc.page_content[:80]}...")


# ============================================================
# 4. Score Threshold Retriever — only confident matches
# ============================================================
print("\n" + "=" * 50)
print("4. Score Threshold Retriever")
print("=" * 50)

threshold_retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.5,
        "k": 5
    }
)

results = threshold_retriever.invoke("What is a vector store?")
print(f"Query: 'What is a vector store?'  →  {len(results)} results above threshold")
for doc in results:
    print(f"  ({doc.metadata['source']}) {doc.page_content[:80]}...")

results_low = threshold_retriever.invoke("What is the best pizza recipe?")
print(f"\nQuery: 'What is the best pizza recipe?'  →  {len(results_low)} results (unrelated topic)")


# ============================================================
# 5. MultiQueryRetriever — LLM generates multiple query variations
# ============================================================
print("\n" + "=" * 50)
print("5. MultiQueryRetriever")
print("=" * 50)

import logging
# Uncomment below to see the generated queries in console
# logging.basicConfig()
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

base_retriever = db.as_retriever(search_kwargs={"k": 2})

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

results = multi_retriever.invoke("How does retrieval work in LangChain?")
print(f"Query: 'How does retrieval work in LangChain?'")
print(f"Got {len(results)} unique docs (LLM generated multiple query variations internally)")
for doc in results:
    print(f"  ({doc.metadata['source']}) {doc.page_content[:80]}...")


# ============================================================
# 6. ContextualCompressionRetriever — compress chunks to relevant parts
# ============================================================
print("\n" + "=" * 50)
print("6. ContextualCompressionRetriever")
print("=" * 50)

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=db.as_retriever(search_kwargs={"k": 3})
)

results = compression_retriever.invoke("What is FAISS?")
print(f"Query: 'What is FAISS?'  →  compressed results:")
for doc in results:
    print(f"  ({doc.metadata.get('source', '?')}) {doc.page_content}")


# ============================================================
# 7. FULL RAG CHAIN — retriever wired into a complete pipeline
# ============================================================
print("\n" + "=" * 50)
print("7. Full RAG Chain (retriever | prompt | llm | parser)")
print("=" * 50)

def format_docs(docs):
    """Join retrieved docs into a single context block."""
    return "\n\n".join(
        f"[{doc.metadata.get('source', '?')}]: {doc.page_content}"
        for doc in docs
    )

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}"""),
    ("human", "{question}")
])

final_retriever = db.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {
        "context":  final_retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

questions = [
    "What is RAG?",
    "How do embeddings work?",
    "What is the best football team?",   # not in our docs — should say I don't know
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {rag_chain.invoke(q)}")
