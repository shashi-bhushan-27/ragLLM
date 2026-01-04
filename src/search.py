import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.vector_store import FaissVectorStore

load_dotenv()

class RAGSearch:
    def __init__(self, store: FaissVectorStore, llm_model: str = "llama-3.1-8b-instant"):
        self.store = store
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        if self.store.index is None:
            return "Vector store is empty. Please upload documents first."

        results = self.store.query(query, top_k=top_k)
        if not results:
            return "No relevant documents found."

        texts = [r["metadata"]["text"] for r in results if r.get("metadata")]
        context = "\n\n".join(texts)

        prompt = f"""Answer the question using the context below.

        Context:
        {context}

        Question:
        {query}

        Answer:"""

        response = self.llm.invoke([prompt])
        return response.content
