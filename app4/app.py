from pymilvus import MilvusClient, Collection
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
from actions import get_response_time, get_weather, calculate_sum

# Initialize Milvus Lite local DB client
milvus_db_path = "milvus_rag_db.db"
client = MilvusClient(milvus_db_path)

# Load Milvus collection
collection_name = "rag_milvus_collection"
collection = Collection(collection_name, using="default")

# Initialize embedding model (all-MiniLM-L6-v2)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize GPT4All model
llm = GPT4All(
    model="/Users/johnmoses/.cache/lm-studio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf",
    n_threads=8,
)


# Retrieve top-k relevant docs from Milvus
def retrieve_docs(query, top_k=3):
    query_embedding = embedding_model.encode([query])[0].tolist()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=None,
    )
    docs = []
    for hits in results:
        for hit in hits:
            text = hit.entity.get("text", "")
            docs.append(text)
    return docs


# Prompt template for RAG
def rag_prompt(context, question):
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"


# External tools dispatcher
def call_tool(tool_name, argument):
    if tool_name == "weather":
        return get_weather(argument)
    elif tool_name == "response_time":
        return get_response_time(argument)
    elif tool_name == "sum":
        return calculate_sum(argument)
    else:
        return "Unknown tool"


# Query router to select agent/tool
def route_query(query):
    q = query.lower()
    if "weather" in q:
        return "weather", query
    elif "response time" in q:
        return "response_time", query
    elif "sum" in q:
        return "sum", query
    elif any(k in q for k in ["explain", "define", "what is", "who is"]):
        return "rag", query
    else:
        return "general", query


# General GPT4All response (no retrieval)
def general_response(query):
    return llm.generate(query)


# RAG response: retrieve docs, build prompt, generate answer
def rag_response(query):
    docs = retrieve_docs(query)
    context = "\n".join(docs)
    prompt = rag_prompt(context, query)
    return llm.generate(prompt)


def main():
    print("Multi-agent system (no LangChain). Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        agent, arg = route_query(user_input)
        if agent == "rag":
            answer = rag_response(arg)
            print("AI (RAG Agent):", answer)
        elif agent in ["weather", "response_time", "sum"]:
            answer = call_tool(agent, arg)
            print(f"AI ({agent.title()} Tool):", answer)
        else:
            answer = general_response(arg)
            print("AI (General Agent):", answer)

    client.close()


if __name__ == "__main__":
    main()
