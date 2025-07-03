import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Connect to Milvus Lite embedded instance (local file storage)
connections.connect("default", uri="file://./milvus_lite_db")

def create_collection(name, dim):
    if Collection.exists(name):
        Collection.drop(name)
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
    ]
    schema = CollectionSchema(fields, description=f"{name} knowledge base")
    collection = Collection(name, schema)
    return collection

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dim = embedding_model.get_sentence_embedding_dimension()

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device)
gpt2_model.eval()

class Agent:
    def __init__(self, name):
        self.name = name
        self.collection = create_collection(name, dim)

    def embed_texts(self, texts):
        return embedding_model.encode(texts).astype(np.float32)

    def insert_knowledge(self, texts):
        embeddings = self.embed_texts(texts)
        self.collection.insert([[], embeddings, texts])
        self.collection.flush()

    def retrieve_context(self, query, top_k=3):
        query_emb = self.embed_texts([query])[0]
        results = self.collection.search(
            [query_emb],
            "embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text"]
        )
        hits = results[0]
        return [hit.entity.get("text") for hit in hits]

    def generate_text(self, prompt, max_length=150):
        input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_ids = gpt2_model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
        text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text[len(prompt):].strip()

    def respond(self, query, shared_context=""):
        retrieved = self.retrieve_context(query)
        combined_context = shared_context + "\n".join(retrieved)
        prompt = f"Context:\n{combined_context}\nQuestion: {query}\nAnswer:"
        return self.generate_text(prompt)

# Initialize agents and seed knowledge
flight_agent = Agent("flight_agent")
weather_agent = Agent("weather_agent")

flight_agent.insert_knowledge([
    "Flights to Paris depart daily at various times.",
    "Booking a flight requires destination and date."
])

weather_agent.insert_knowledge([
    "Paris weather is mild in spring.",
    "Weather forecasts help travelers plan their trips."
])

class Manager:
    def __init__(self, agents):
        self.agents = agents

    def handle_query(self, query):
        shared_context = ""
        agent_responses = {}

        # Each agent responds with context augmented by shared context
        for agent in self.agents:
            response = agent.respond(query, shared_context)
            agent_responses[agent.name] = response
            shared_context += f"\n[{agent.name}]: {response}"

        # Final combined prompt for GPT-2
        final_prompt = f"Combined context:\n{shared_context}\nUser question: {query}\nFinal answer:"
        input_ids = gpt2_tokenizer.encode(final_prompt, return_tensors="pt").to(device)
        output_ids = gpt2_model.generate(
            input_ids,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
        final_answer = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return final_answer[len(final_prompt):].strip()

def main():
    print("Welcome to the Multi-Agent CLI with Milvus Lite RAG!")
    print("Type your queries below. Type 'exit' or 'quit' to stop.\n")

    manager = Manager([flight_agent, weather_agent])

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            print("Please enter a valid query.")
            continue

        answer = manager.handle_query(user_input)
        print(f"AI: {answer}\n")

if __name__ == "__main__":
    main()
