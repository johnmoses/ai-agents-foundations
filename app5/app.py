import json
import threading
import time
import logging
import re

from flask import Flask, request, jsonify, render_template
import yfinance as yf

from llama_cpp import Llama
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from pymilvus import (
    MilvusClient,
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
# nltk.download('punkt')

MILVUS_DB_URI = "milvus_rag_db.db"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

milvus_client = MilvusClient(MILVUS_DB_URI)
connections.connect(alias="default", uri=MILVUS_DB_URI)


# --- Step 2: Create collection with primary key if not exists ---
def create_collection():
    if COLLECTION_NAME in milvus_client.list_collections():
        return Collection(COLLECTION_NAME, using="default")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    ]
    schema = CollectionSchema(fields, description="RAG collection")
    return Collection(name=COLLECTION_NAME, schema=schema, using="default")


collection = create_collection()

# --- Step 3: Prepare in-memory documents ---
documents = [
    "Milvus is an open-source vector database built for scalable similarity search.",
    "It supports embedding-based search for images, video, and text.",
    "You can use SentenceTransformers to generate embeddings for your documents.",
    "GPT-2 is an open-source language model suitable for text generation tasks.",
]

# --- Step 4: Embed documents ---
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)


# --- Step 5: Insert data programmatically ---
def insert_data(collection, embeddings, texts):
    entities = [
        embeddings.tolist(),  # embeddings
        texts,  # texts
    ]
    collection.insert(entities)
    collection.flush()


if collection.num_entities == 0:
    insert_data(collection, doc_embeddings, documents)
else:
    print(
        f"Collection already has {collection.num_entities} entities, skipping insert."
    )

# --- Step 5.1: Create index and load collection ---
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}

try:
    print("Creating index on embedding field...")
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index created.")
except Exception as e:
    print(f"Index creation skipped or failed: {e}")

print("Loading collection into memory...")
collection.load()
print("Collection loaded.")


def embed_text(text: str):
    # Dummy example: replace with your embedding model inference
    # For example, use sentence-transformers or OpenAI embeddings
    # Here we just return a fixed-size zero vector for demo purposes
    import numpy as np

    return np.random.rand(768).tolist()

# --- Initialize LLaMA model ---
MODEL_PATH = "/Users/johnmoses/.cache/lm-studio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"  # Update this path!
llm = Llama(model_path=MODEL_PATH)

# --- Logging setup ---
logger = logging.getLogger('financial_chatbot')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('financial_chatbot.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = Flask(__name__)

def truncate_prompt_by_chars(prompt, max_chars=1500):
    """
    Truncate the prompt string to max_chars characters from the end (keep recent context).
    """
    if len(prompt) > max_chars:
        prompt = prompt[-max_chars:]
    return prompt

# --- Generate response with LLaMA ---
def generate_response(prompt, max_tokens=128, temperature=0.7):
    prompt = truncate_prompt_by_chars(prompt, max_chars=1500)
    start_time = time.time()
    output = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["User:", "Assistant:"]
    )
    latency = time.time() - start_time
    response_text = output['choices'][0]['text'].strip()
    return response_text, latency


# --- Few-shot + CoT Intent Agent ---
class IntentAgent:
    def __init__(self, llm):
        self.llm = llm

    def detect_intent(self, query):
        prompt = self.build_prompt(query)
        output_text, _ = generate_response(prompt, max_tokens=50, temperature=0.0)
        valid_intents = {
            "get_stock_price", "get_historical_data", "calculate_interest",
            "get_compliance_docs", "compare_stock_prices", "general_chat"
        }
        for line in reversed(output_text.split('\n')):
            line_clean = line.strip().lower().replace(".", "")
            if line_clean in valid_intents:
                return line_clean
        return "general_chat"

    @staticmethod
    def build_prompt(user_query: str) -> str:
        system_prompt = """
        [INST] <<SYS>>
        You are a precise financial assistant. Your task is to classify the intent of a user query into one of these categories:
        - get_stock_price
        - get_historical_data
        - calculate_interest
        - get_compliance_docs
        - compare_stock_prices
        - general_chat

        For each query, first explain your reasoning step-by-step, then output ONLY the intent label exactly as above.

        If the query is ambiguous or does not fit any category, respond with "general_chat".

        Be concise and accurate.
        <</SYS>>
        """
        examples = """
        User query: "What is the current price of AAPL stock?"
        Reasoning: The user is asking about the current price of a specific stock ticker, so the intent is to get stock price.
        Intent: get_stock_price

        User query: "Show me the stock history of Tesla for the last month."
        Reasoning: The user requests historical stock data for Tesla, so the intent is to get historical data.
        Intent: get_historical_data

        User query: "Calculate interest on 1000 dollars at 5% for 2 years."
        Reasoning: The user wants to calculate interest based on principal, rate, and time, so the intent is calculate_interest.
        Intent: calculate_interest

        User query: "Where can I find GDPR compliance documents?"
        Reasoning: The user is asking for compliance documents related to GDPR, so the intent is get_compliance_docs.
        Intent: get_compliance_docs

        User query: "Compare the prices of MSFT and GOOG stocks."
        Reasoning: The user wants to compare prices of two stocks, so the intent is compare_stock_prices.
        Intent: compare_stock_prices

        User query: "Hello, how are you?"
        Reasoning: This is a general greeting not related to finance, so the intent is general_chat.
        Intent: general_chat

        User query: "{user_query}"
        Reasoning:"""
        return f"{system_prompt}{examples}".replace("{user_query}", user_query)

# --- Retrieval Agent ---
class RetrievalAgent:
    def __init__(self, milvus_client):
        self.milvus_client = milvus_client

    def retrieve(self, query_embedding, top_k=3):
        return self.milvus_client.search(query_embedding, top_k=top_k)

# --- Finance Agent ---
class FinanceAgent:
    def get_stock_price(self, ticker):
        try:
            ticker_obj = yf.Ticker(ticker)
            price = ticker_obj.history(period="1d")['Close'][-1]
            return f"The current price of {ticker.upper()} is ${price:.2f}"
        except Exception as e:
            return f"Error fetching stock price for {ticker}: {str(e)}"

    def calculate_interest(self, principal, rate, time):
        try:
            p = float(principal)
            r = float(rate)
            t = float(time)
            interest = (p * r * t) / 100
            return f"Calculated simple interest is ${interest:.2f}"
        except Exception as e:
            return f"Error calculating interest: {str(e)}"

    def compare_stock_prices(self, ticker1, ticker2):
        try:
            price1 = yf.Ticker(ticker1).history(period="1d")['Close'][-1]
            price2 = yf.Ticker(ticker2).history(period="1d")['Close'][-1]
            if price1 > price2:
                return f"{ticker1.upper()} (${price1:.2f}) is priced higher than {ticker2.upper()} (${price2:.2f})"
            elif price2 > price1:
                return f"{ticker2.upper()} (${price2:.2f}) is priced higher than {ticker1.upper()} (${price1:.2f})"
            else:
                return f"Both {ticker1.upper()} and {ticker2.upper()} have the same price of ${price1:.2f}"
        except Exception as e:
            return f"Error comparing stock prices: {str(e)}"

    def get_historical_data(self, ticker, period="1mo"):
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period)
            if hist.empty:
                return f"No historical data available for {ticker} for period {period}."
            summary = hist[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5).to_dict()
            return f"Last 5 days of {ticker.upper()} historical  {summary}"
        except Exception as e:
            return f"Error fetching historical data for {ticker}: {str(e)}"

# --- Compliance Agent ---
class ComplianceAgent:
    def get_docs(self, topic):
        docs = {
            "gdpr": "GDPR info: https://gdpr-info.eu/",
            "sox": "Sarbanes-Oxley Act info: https://www.soxlaw.com/",
            "basel": "Basel III framework: https://www.bis.org/bcbs/basel3.htm"
        }
        return docs.get(topic.lower(), "Compliance document not found for the specified topic.")

# --- Chat Agent ---
class ChatAgent:
    def __init__(self, llm):
        self.llm = llm

    def chat(self, user_message, context=""):
        prompt = f"Context:\n{context}\nUser: {user_message}\nAssistant:"
        response = generate_response(prompt, max_tokens=128, temperature=0.7)
        return response

# --- Coordinator Agent ---
class CoordinatorAgent:
    def __init__(self, intent_agent, retrieval_agent, finance_agent, compliance_agent, chat_agent, embedder):
        self.intent_agent = intent_agent
        self.retrieval_agent = retrieval_agent
        self.finance_agent = finance_agent
        self.compliance_agent = compliance_agent
        self.chat_agent = chat_agent
        self.embedder = embedder

    def handle_query(self, user_message):
        intent = self.intent_agent.detect_intent(user_message)
        tickers = re.findall(r'\b[A-Za-z]{1,5}\b', user_message.upper())
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", user_message)

        if intent == "get_stock_price":
            if tickers:
                return intent, self.finance_agent.get_stock_price(tickers[0])
            else:
                return intent, "Please specify a valid stock ticker symbol."
        elif intent == "get_historical_data":
            if tickers:
                return intent, self.finance_agent.get_historical_data(tickers[0])
            else:
                return intent, "Please specify a stock ticker symbol."
        elif intent == "compare_stock_prices":
            if len(tickers) >= 2:
                return intent, self.finance_agent.compare_stock_prices(tickers[0], tickers[1])
            else:
                return intent, "Please specify two stock ticker symbols to compare."
        elif intent == "calculate_interest":
            if len(numbers) >= 3:
                return intent, self.finance_agent.calculate_interest(numbers[0], numbers[1], numbers[2])
            else:
                return intent, "Please provide principal, rate, and time for interest calculation."
        elif intent == "get_compliance_docs":
            topics = ["gdpr", "sox", "basel"]
            topic = next((t for t in topics if t in user_message.lower()), None)
            if topic:
                return intent, self.compliance_agent.get_docs(topic)
            else:
                return intent, "Please specify a compliance topic like GDPR, SOX, or Basel."
        else:
            query_embedding = self.embedder(user_message)
            retrieved_docs = self.retrieval_agent.retrieve(query_embedding)
            context = "\n\n".join(retrieved_docs)
            response = self.chat_agent.chat(user_message, context)
            return intent, response

# --- Instantiate agents ---
intent_agent = IntentAgent(llm)
retrieval_agent = RetrievalAgent(milvus_client)
finance_agent = FinanceAgent()
compliance_agent = ComplianceAgent()
chat_agent = ChatAgent(llm)

coordinator = CoordinatorAgent(intent_agent, retrieval_agent, finance_agent, compliance_agent, chat_agent, embed_text)

# --- Flask API ---

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")
    intent, response = coordinator.handle_query(user_message)
    return jsonify({"intent": intent, "response": response})

@app.route('/')
def index():
    return render_template('index.html')  # Your chat UI file

# --- Run Flask app ---
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5001)
