from typing import List, Optional
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import numpy as np

class MilvusRAG:
    def __init__(self, db_path: str = "milvus_rag_db.db"):
        """Initialize Milvus RAG system"""
        self.client = MilvusClient(db_path)
        self.collection_name = "documents"
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

    def create_collection(self):
        if self.client.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists.")
            return
        print(f"Creating collection '{self.collection_name}'...")
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedding_dim,
            metric_type="COSINE",
            consistency_level="Strong"
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_type="HNSW",
            index_params={"M": 16, "efConstruction": 200}
        )
        print("Collection and index created.")

    def embed_text(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model.encode(texts, convert_to_numpy=True)

    def index_documents(self, docs: List[str], batch_size: int = 64):
        self.create_collection()
        print(f"Indexing {len(docs)} documents in batches of {batch_size}...")
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            embeddings = self.embed_text(batch_docs)
            data = []
            for j, (embedding, doc) in enumerate(zip(embeddings, batch_docs)):
                data.append({
                    "id": i + j,
                    "vector": embedding.tolist(),
                    "text": doc
                })
            self.client.insert(collection_name=self.collection_name, data=data)
        self.client.flush(collection_name=self.collection_name)
        print("Data insertion complete.")

    def seed_data(self):
        sample_docs = [
            "Photosynthesis is the process by which green plants use sunlight to synthesize foods from carbon dioxide and water.",
            "The mitochondrion is the powerhouse of the cell, generating ATP through cellular respiration.",
            "Newton's first law states that an object in motion stays in motion unless acted upon by an external force.",
            "Water boils at 100 degrees Celsius at sea level pressure.",
            "The capital of France is Paris, known for landmarks like the Eiffel Tower and Louvre Museum.",
            "The process of cell division in eukaryotes is called mitosis, resulting in two identical daughter cells.",
            "Gravity is the force that attracts two bodies toward each other, proportional to their masses.",
            "The human heart has four chambers: two atria and two ventricles.",
            "In chemistry, an acid is a substance that donates protons or hydrogen ions and accepts electrons.",
            "The Great Wall of China is a series of fortifications built to protect against invasions."
        ]
        self.index_documents(sample_docs)

    def retrieve(self, query: str, top_k: int = 5) -> Optional[List[dict]]:
        query_emb = self.embed_text([query])
        results = self.client.search(
            collection_name=self.collection_name,
            data=query_emb,
            limit=top_k,
            output_fields=["text"]
        )
        # Format results
        formatted_results = []
        for result in results[0]:  # results is a list of lists
            formatted_results.append(
                {
                    "id": result["id"],
                    "text": result["entity"]["text"],
                    "score": result["distance"],
                }
            )

        if not formatted_results:
            return None
        return formatted_results

    def generate_prompt(self, query: str, docs: List[dict]) -> str:
        context = "\n\n".join([doc['text'] for doc in docs])
        prompt = (
            "You are an expert assistant. Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        return prompt

    def generate_answer(self, query: str, llm_callable, top_k: int = 5) -> str:
        docs = self.retrieve(query, top_k)
        if not docs:
            return "Sorry, I could not find relevant information."
        prompt = self.generate_prompt(query, docs)
        response = llm_callable(prompt, max_tokens=512, temperature=0.3)
        return response["choices"][0]["text"].strip()

    def build_chat_prompt(self, conversation: List[dict], retrieved_docs: List[dict]) -> str:
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        prompt = "You are an AI assistant. Use the following context to answer the conversation.\n\n"
        prompt += f"Context:\n{context}\n\nConversation:\n"
        for turn in conversation:
            prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
        prompt += "Assistant:"
        return prompt

    def chat(self, user_message: str, conversation_history: List[dict], llm_callable, top_k: int = 5) -> str:
        docs = self.retrieve(user_message, top_k) or []
        prompt = self.build_chat_prompt(conversation_history + [{'role': 'user', 'content': user_message}], docs)
        response = llm_callable(prompt, max_tokens=512, temperature=0.3)
        return response["choices"][0]["text"].strip()


    