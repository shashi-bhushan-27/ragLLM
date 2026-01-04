from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = []
        skipped = 0
        for i, chunk in enumerate(chunks):
            text = getattr(chunk, "page_content", "")
            # Only accept plain strings
            if not isinstance(text, str):
                print(f"[WARN] Skipping non-string chunk at index {i} (type: {type(text).__name__})")
                skipped += 1
                continue
            # Strip whitespace
            text = text.strip()
            if not text:
                print(f"[WARN] Skipping empty chunk at index {i}")
                skipped += 1
                continue
            # Ensure the text is valid for encoding (replace problematic chars)
            try:
                # Test if the string can be encoded
                text.encode('utf-8')
            except UnicodeEncodeError as e:
                print(f"[WARN] Skipping chunk at index {i} (encoding error: {e})")
                skipped += 1
                continue
            texts.append(text)
        
        print(f"[INFO] Valid chunks: {len(texts)}, Skipped: {skipped}")
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        
        if not texts:
            print("[ERROR] No valid text chunks to embed.")
            return np.array([])
        
        # Encode with error handling
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            print(f"[INFO] Embeddings shape: {embeddings.shape}")
            return embeddings
        except TypeError as e:
            print(f"[ERROR] Encoding failed: {e}")
            print(f"[DEBUG] Sample texts to encode: {texts[:3]}")
            print(f"[DEBUG] Text types: {[type(t).__name__ for t in texts[:3]]}")
            raise

