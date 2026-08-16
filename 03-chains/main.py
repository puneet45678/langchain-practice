from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


# ============================================================
# 1. StrOutputParser — strip AIMessage, get plain string
# ============================================================
print("=" * 50)
print("1. StrOutputParser")
print("=" * 50)

template = ChatPromptTemplate.from_messages([
    ("system", "You reply in exactly one sentence."),
    ("human", "What is {topic}?")
])

# Without parser — get AIMessage back
messages = template.invoke({"topic": "LangChain"})
raw = llm.invoke(messages)
print(f"Without parser type: {type(raw).__name__}")   # AIMessage

# With parser — get plain string
chain = template | llm | StrOutputParser()
result = chain.invoke({"topic": "LangChain"})
print(f"With parser type:    {type(result).__name__}")  # str
print(f"Result: {result}\n")


# ============================================================
# 2. JsonOutputParser — LLM returns JSON, parse to dict
# ============================================================
print("=" * 50)
print("2. JsonOutputParser")
print("=" * 50)

parser = JsonOutputParser()

template = ChatPromptTemplate.from_messages([
    ("system", "Reply only with valid JSON. No markdown fences, no extra text."),
    ("human", "Give me info about {country}. Keys: name, capital, population, language.")
])

chain = template | llm | parser

result = chain.invoke({"country": "Japan"})
print(f"Type: {type(result).__name__}")     # dict
print(f"Capital: {result['capital']}")
print(f"Full result: {result}\n")


# ============================================================
# 3. StructuredOutputParser (manual) — inject schema description into prompt
# NOTE: StructuredOutputParser was removed in LangChain 1.x.
# The same concept still works: you manually write format instructions
# in the system prompt and use JsonOutputParser to parse the result.
# ============================================================
print("=" * 50)
print("3. Structured Output (manual schema in prompt)")
print("=" * 50)

format_instructions = """Reply with a JSON object using exactly these keys:
- name: Name of the country
- capital: Capital city
- population: Approximate population as a plain integer
- language: Official language
Return only the JSON object, no extra text."""

template = ChatPromptTemplate.from_messages([
    ("system", "Answer the user query.\n{format_instructions}"),
    ("human", "Tell me about {country}.")
])

chain = template | llm | JsonOutputParser()

result = chain.invoke({
    "country": "France",
    "format_instructions": format_instructions
})

print(f"Type: {type(result).__name__}")     # dict
print(f"Capital:  {result['capital']}")
print(f"Language: {result['language']}\n")


# ============================================================
# 4. PydanticOutputParser — typed Python object with validation
# ============================================================
print("=" * 50)
print("4. PydanticOutputParser")
print("=" * 50)

class Country(BaseModel):
    name:       str = Field(description="Name of the country")
    capital:    str = Field(description="Capital city")
    population: int = Field(description="Population as a plain integer")
    language:   str = Field(description="Official language")

parser = PydanticOutputParser(pydantic_object=Country)

template = ChatPromptTemplate.from_messages([
    ("system", "Answer the user query.\n{format_instructions}"),
    ("human", "Tell me about {country}.")
])

chain = template | llm | parser

result = chain.invoke({
    "country": "Germany",
    "format_instructions": parser.get_format_instructions()
})

print(f"Type:       {type(result).__name__}")   # Country (Pydantic model)
print(f"Dot access: {result.capital}")          # "Berlin"  (not result["capital"])
print(f"Typed int:  {result.population} ({type(result.population).__name__})")
print(f"Full object: {result}\n")


# ============================================================
# 5. CHAINING RECAP — all 3 steps in one line
# ============================================================
print("=" * 50)
print("5. Full chain recap (prompt | llm | parser)")
print("=" * 50)

class Movie(BaseModel):
    title:    str = Field(description="Movie title")
    director: str = Field(description="Director name")
    year:     int = Field(description="Release year as integer")
    genre:    str = Field(description="Genre of the movie")

parser = PydanticOutputParser(pydantic_object=Movie)

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Answer in the requested format.\n{format_instructions}"),
        ("human", "Tell me about the movie: {movie}")
    ])
    | llm
    | parser
)

result = chain.invoke({
    "movie": "Inception",
    "format_instructions": parser.get_format_instructions()
})

print(f"Title:    {result.title}")
print(f"Director: {result.director}")
print(f"Year:     {result.year}")
print(f"Genre:    {result.genre}")
