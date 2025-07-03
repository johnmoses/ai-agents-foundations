import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict, Any

class MilvusRAG:
    def __init__(self, db_path: str = "milvus_rag_db.db"):
        """Initialize Milvus RAG system"""
        self.client = MilvusClient(db_path)
        self.collection_name = "documents"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
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
                consistency_level="Strong"
            )
            print(f"Collection '{self.collection_name}' created successfully")
            
        except Exception as e:
            print(f"Error creating collection: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def insert_documents(self, documents: List[Dict[str, Any]]):
        """Insert documents into Milvus collection"""
        try:
            # Prepare data for insertion
            data = []
            for i, doc in enumerate(documents):
                # Generate embedding for the document text
                embedding = self.embed_text(doc['text'])
                
                data.append({
                    "id": i,
                    "text": doc['text'],
                    "title": doc.get('title', ''),
                    "category": doc.get('category', ''),
                    "vector": embedding
                })
            
            # Insert data
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            print(f"Inserted {len(documents)} documents successfully")
            
        except Exception as e:
            print(f"Error inserting documents: {e}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.embed_text(query)
            
            # Search in Milvus
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text", "title", "category"]
            )
            
            # Format results
            formatted_results = []
            for result in results[0]:  # results is a list of lists
                formatted_results.append({
                    "id": result["id"],
                    "text": result["entity"]["text"],
                    "title": result["entity"]["title"],
                    "category": result["entity"]["category"],
                    "score": result["distance"]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def generate_response(self, query: str, top_k: int = 3) -> str:
        """Generate RAG response based on retrieved documents"""
        # Retrieve relevant documents
        relevant_docs = self.search_similar(query, top_k)
        
        if not relevant_docs:
            return "No relevant documents found."
        
        # Combine retrieved documents as context
        context = "\n\n".join([
            f"Document {i+1} (Score: {doc['score']:.3f}):\n"
            f"Title: {doc['title']}\n"
            f"Category: {doc['category']}\n"
            f"Content: {doc['text']}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Simple response generation (in practice, you'd use an LLM here)
        response = f"""Based on the retrieved documents, here's the information related to your query: "{query}"

Retrieved Context:
{context}

Summary: The most relevant document (score: {relevant_docs[0]['score']:.3f}) suggests that {relevant_docs[0]['text'][:200]}...

Note: This is a basic RAG implementation. For production use, integrate with an LLM for better response generation."""
        
        return response

def create_demo_data() -> List[Dict[str, Any]]:
    """Create demo documents for testing"""
    return [
        {
            "title": "Introduction to Machine Learning",
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns.",
            "category": "AI/ML"
        },
        {
            "title": "Deep Learning Fundamentals",
            "text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition.",
            "category": "AI/ML"
        },
        {
            "title": "Python Programming Basics",
            "text": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, artificial intelligence, and automation. Python's syntax is clean and intuitive, making it an excellent choice for beginners.",
            "category": "Programming"
        },
        {
            "title": "Data Structures and Algorithms",
            "text": "Data structures are ways of organizing and storing data in a computer so that it can be accessed and modified efficiently. Common data structures include arrays, linked lists, stacks, queues, trees, and graphs. Algorithms are step-by-step procedures for solving problems.",
            "category": "Computer Science"
        },
        {
            "title": "Web Development with Flask",
            "text": "Flask is a lightweight web framework for Python that provides the basic tools needed to build web applications. It's minimalist and flexible, allowing developers to choose their preferred tools and libraries for specific functionalities.",
            "category": "Web Development"
        },
        {
            "title": "Database Design Principles",
            "text": "Database design involves creating a detailed data model of a database. Good database design ensures data integrity, reduces redundancy, and optimizes performance. Key principles include normalization, proper indexing, and defining clear relationships between entities.",
            "category": "Database"
        },
        {
            "title": "Vector Databases Overview",
            "text": "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They're essential for applications involving machine learning, AI, and similarity search. Examples include Milvus, Pinecone, and Weaviate.",
            "category": "Database"
        },
        {
            "title": "Natural Language Processing",
            "text": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves tasks like text classification, sentiment analysis, machine translation, and question answering.",
            "category": "AI/ML"
        },
        {
            "title": "Cloud Computing Concepts",
            "text": "Cloud computing is the delivery of computing services over the internet. It provides scalable resources like servers, storage, databases, and software on demand. Major cloud providers include Amazon Web Services, Microsoft Azure, and Google Cloud Platform.",
            "category": "Cloud"
        },
        {
            "title": "Cybersecurity Best Practices",
            "text": "Cybersecurity involves protecting digital information, systems, and networks from threats. Best practices include using strong passwords, enabling two-factor authentication, keeping software updated, and being cautious with email attachments and links.",
            "category": "Security"
        }
    ]

def display_menu():
    """Display the main menu options"""
    print("\n" + "="*60)
    print("🤖 MILVUS RAG SYSTEM")
    print("="*60)
    print("1. 🔍 Search/Query Documents")
    print("2. 📄 View All Documents")
    print("3. ➕ Add New Document")
    print("4. 📊 Show Collection Statistics")
    print("5. 🧪 Run Demo Queries")
    print("6. 🗑️  Clear Database")
    print("7. ❌ Exit")
    print("="*60)

def get_user_input(prompt: str, input_type: str = "string") -> Any:
    """Get user input with validation"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                print("❌ Input cannot be empty. Please try again.")
                continue
            
            if input_type == "int":
                return int(user_input)
            elif input_type == "float":
                return float(user_input)
            else:
                return user_input
        except ValueError:
            print(f"❌ Please enter a valid {input_type}.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            exit(0)

def view_all_documents(rag: MilvusRAG):
    """Display all documents in the collection"""
    try:
        # Get all documents by searching with a broad query
        all_docs = rag.search_similar("", top_k=1000)  # Get many documents
        
        if not all_docs:
            print("📝 No documents found in the collection.")
            return
        
        print(f"\n📚 Found {len(all_docs)} documents:")
        print("-" * 80)
        
        for i, doc in enumerate(all_docs, 1):
            print(f"{i}. 📄 {doc['title']}")
            print(f"   📁 Category: {doc['category']}")
            print(f"   📝 Text: {doc['text'][:150]}...")
            print(f"   🆔 ID: {doc['id']}")
            print()
            
            # Pause every 5 documents for readability
            if i % 5 == 0 and i < len(all_docs):
                cont = input("Press Enter to continue or 'q' to stop: ")
                if cont.lower() == 'q':
                    break
                    
    except Exception as e:
        print(f"❌ Error viewing documents: {e}")

def add_new_document(rag: MilvusRAG):
    """Add a new document to the collection"""
    print("\n➕ Adding New Document")
    print("-" * 40)
    
    try:
        title = get_user_input("📄 Enter document title: ")
        category = get_user_input("📁 Enter document category: ")
        
        print("📝 Enter document text (press Enter twice to finish):")
        text_lines = []
        empty_line_count = 0
        
        while True:
            line = input()
            if line.strip() == "":
                empty_line_count += 1
                if empty_line_count >= 2:
                    break
            else:
                empty_line_count = 0
                text_lines.append(line)
        
        text = "\n".join(text_lines).strip()
        
        if not text:
            print("❌ Document text cannot be empty.")
            return
        
        # Create document and insert
        new_doc = {
            "title": title,
            "text": text,
            "category": category
        }
        
        rag.insert_documents([new_doc])
        print("✅ Document added successfully!")
        
    except Exception as e:
        print(f"❌ Error adding document: {e}")

def show_collection_stats(rag: MilvusRAG):
    """Display collection statistics"""
    try:
        # Get collection info
        stats = rag.client.describe_collection(collection_name=rag.collection_name)
        print(f"\n📊 Collection Statistics")
        print("-" * 40)
        print(f"📁 Collection Name: {rag.collection_name}")
        print(f"🔢 Vector Dimension: {rag.embedding_dim}")
        print(f"📏 Metric Type: COSINE")
        print(f"🎯 Embedding Model: all-MiniLM-L6-v2")
        
        # Try to get document count (this might vary based on Milvus version)
        try:
            # Simple way to estimate count
            docs = rag.search_similar("", top_k=10000)
            print(f"📄 Estimated Documents: {len(docs)}")
        except:
            print("📄 Document count: Unable to determine")
            
    except Exception as e:
        print(f"❌ Error getting collection stats: {e}")

def run_demo_queries(rag: MilvusRAG):
    """Run predefined demo queries"""
    demo_queries = [
        "What is machine learning?",
        "Tell me about Python programming",
        "How do vector databases work?",
        "What are data structures?",
        "Explain cloud computing"
    ]
    
    print("\n🧪 Running Demo Queries")
    print("=" * 50)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        # Get similar documents
        similar_docs = rag.search_similar(query, top_k=3)
        
        if similar_docs:
            print("🔍 Top 3 similar documents:")
            for j, doc in enumerate(similar_docs, 1):
                print(f"   {j}. {doc['title']} (Score: {doc['score']:.3f})")
                print(f"      📁 {doc['category']}")
                print(f"      📝 {doc['text'][:100]}...")
                print()
        else:
            print("❌ No similar documents found.")
        
        # Ask if user wants to continue
        if i < len(demo_queries):
            cont = input("Press Enter to continue to next query or 'q' to stop: ")
            if cont.lower() == 'q':
                break

def search_documents(rag: MilvusRAG):
    """Interactive document search"""
    print("\n🔍 Document Search")
    print("-" * 30)
    
    query = get_user_input("🔎 Enter your search query: ")
    
    # Get number of results
    try:
        top_k = int(get_user_input("📊 Number of results (default 5): ") or "5")
        top_k = max(1, min(top_k, 50))  # Limit between 1 and 50
    except:
        top_k = 5
    
    print(f"\n🔍 Searching for: '{query}'")
    print("-" * 50)
    
    # Search documents
    similar_docs = rag.search_similar(query, top_k=top_k)
    
    if not similar_docs:
        print("❌ No similar documents found.")
        return
    
    print(f"📄 Found {len(similar_docs)} similar documents:")
    print()
    
    for i, doc in enumerate(similar_docs, 1):
        print(f"{i}. 📄 {doc['title']}")
        print(f"   📁 Category: {doc['category']}")
        print(f"   🎯 Similarity Score: {doc['score']:.4f}")
        print(f"   📝 Text: {doc['text'][:200]}...")
        print()
    
    # Ask if user wants detailed response
    generate_response = input("\n🤖 Generate RAG response? (y/n): ").lower().strip()
    if generate_response == 'y':
        print("\n🤖 Generating response...")
        response = rag.generate_response(query, top_k=min(3, len(similar_docs)))
        print("\n" + "="*60)
        print("🤖 RAG RESPONSE:")
        print("="*60)
        print(response)
        print("="*60)

def clear_database(rag: MilvusRAG):
    """Clear the database"""
    print("\n🗑️  Clear Database")
    print("-" * 30)
    
    confirm = input("⚠️  Are you sure you want to clear all data? (type 'YES' to confirm): ")
    
    if confirm == "YES":
        try:
            rag.client.drop_collection(collection_name=rag.collection_name)
            print("✅ Database cleared successfully!")
            print("🔄 Recreating collection...")
            rag.create_collection()
            print("✅ Collection recreated. You can now add new documents.")
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
    else:
        print("❌ Operation cancelled.")

def initialize_system():
    """Initialize the RAG system with demo data"""
    print("🚀 Initializing Milvus RAG System...")
    
    # Initialize RAG system
    rag = MilvusRAG("milvus_rag_db.db")
    
    # Create collection
    rag.create_collection()
    
    # Check if we need to load demo data
    try:
        existing_docs = rag.search_similar("", top_k=1)
        if not existing_docs:
            print("📚 Loading demo data...")
            demo_docs = create_demo_data()
            rag.insert_documents(demo_docs)
            print("✅ Demo data loaded successfully!")
        else:
            print("📚 Found existing data in database.")
    except:
        print("📚 Loading demo data...")
        demo_docs = create_demo_data()
        rag.insert_documents(demo_docs)
        print("✅ Demo data loaded successfully!")
    
    return rag

def main():
    """Main CLI interface"""
    try:
        # Initialize system
        rag = initialize_system()
        
        print("\n🎉 Welcome to the Milvus RAG System!")
        print("Type your queries and get AI-powered responses based on your document collection.")
        
        while True:
            display_menu()
            
            try:
                choice = get_user_input("🎯 Select an option (1-7): ", "int")
                
                if choice == 1:
                    search_documents(rag)
                elif choice == 2:
                    view_all_documents(rag)
                elif choice == 3:
                    add_new_document(rag)
                elif choice == 4:
                    show_collection_stats(rag)
                elif choice == 5:
                    run_demo_queries(rag)
                elif choice == 6:
                    clear_database(rag)
                elif choice == 7:
                    print("\n👋 Thank you for using Milvus RAG System!")
                    break
                else:
                    print("❌ Invalid option. Please select 1-7.")
                    
            except ValueError:
                print("❌ Please enter a valid number (1-7).")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
                
            # Pause before showing menu again
            input("\nPress Enter to continue...")
                
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check your dependencies and try again.")

if __name__ == "__main__":
    main()