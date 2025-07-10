from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MilvusRAG:
    def __init__(self, db_path: str = "milvus_rag_db1.db"):
        """Initialize Milvus RAG system"""
        self.client = MilvusClient(db_path)
        self.collection_name = "documents"
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

    def create_collection(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            if self.client.has_collection(collection_name=self.collection_name):
                print(f"Collection '{self.collection_name}' already exists")
                return

            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.embedding_dim,
                metric_type="COSINE",
                consistency_level="Strong",
            )
            print(f"Collection '{self.collection_name}' created successfully")

        except Exception as e:
            print(f"Error creating collection: {e}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def ingest(self, documents: List[Dict[str, Any]]):
        """Insert documents into Milvus collection"""
        try:
            # Prepare data for insertion
            data = []
            for i, doc in enumerate(documents):
                # Generate embedding for the document text
                embedding = self.embed_text(doc["text"])

                data.append(
                    {
                        "id": i,
                        "text": doc["text"],
                        "vector": embedding,
                    }
                )

            # Insert data
            self.client.insert(collection_name=self.collection_name, data=data)
            print(f"Inserted {len(documents)} documents successfully")

        except Exception as e:
            print(f"Error inserting documents: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.embed_text(query)

            # Search in Milvus
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text"],
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

            return formatted_results

        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def create_demo_data(self) -> List[Dict[str, Any]]:
        """Create demo documents for testing"""
        return [
            {
                "text": "Fever and cough are common symptoms of flu and COVID-19."
            },
            {
                "text": "Diabetes is a chronic condition characterized by high blood sugar levels."
            },
            {
                "text": "Hypertension increases the risk of heart disease and stroke."
            },
            {
                "text": "Common cold symptoms include sneezing, runny nose, and sore throat."
            },
            {
                "text": "Asthma causes difficulty breathing due to inflamed airways."
            }
        ]

    def seed_db(self):
        try:
            existing_docs = self.search("", top_k=1)
            if not existing_docs:
                print("Loading demo data...")
                demo_docs = self.create_demo_data()
                self.ingest(demo_docs)
                print("Demo data loaded successfully!")
            else:
                print("Found existing data in database.")

        except:
            print("Loading demo data...")
            demo_docs = self.create_demo_data()
            self.ingest(demo_docs)
            print("Demo data loaded successfully!")
