import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# ------------------ APP ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ LOAD VECTORSTORE ------------------
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Always resolve path relative to this file
vectorstore_path = os.path.join(os.path.dirname(__file__), "src", "vectorstore")
if not os.path.exists(vectorstore_path):
    raise FileNotFoundError(
        f"Vectorstore not found at {vectorstore_path}. "
        "Please run ingest.py first: python src/ingest.py"
    )

print("Loading vectorstore...")
vectorstore = FAISS.load_local(
    vectorstore_path,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Vectorstore loaded successfully!")

# ------------------ LLM ------------------
print("Loading LLM...")

from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain_core.language_models.llms import LLM
from typing import Optional, List

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

class FlanT5LLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "flan-t5"

llm = FlanT5LLM()
print("LLM loaded!")

# ------------------ PROMPT ------------------
prompt = ChatPromptTemplate.from_template("""
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {input}

Answer:
""")

# ------------------ CHAIN ------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ------------------ API ROUTES ------------------
@app.get("/")
def home():
    return {"message": "College RAG API is running!"}

@app.get("/chat")
def chat(query: str):
    if not query.strip():
        return {"error": "Query cannot be empty"}
    try:
        answer = chain.invoke(query)
        return {"query": query, "answer": answer}
    except Exception as e:
        return {"error": str(e)}
@app.get("/debug")
def debug(query: str):
    docs = retriever.invoke(query)
    return {"chunks": [doc.page_content for doc in docs]}