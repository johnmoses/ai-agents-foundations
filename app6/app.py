import os
import logging
from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama
from rag import MilvusRAG

# --- Flask app setup ---
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG system
milvus_rag = MilvusRAG("milvus_rag_db.db")

# Create collection
milvus_rag.create_collection()

# Seed db
milvus_rag.seed_db()

# --- Load Llama 3B model ---
model_path = os.path.expanduser(
    "/Users/johnmoses/.cache/lm-studio/models/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
)  # Update path as needed
if not os.path.exists(model_path):
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

llm = Llama(model_path=model_path, n_ctx=2048, temperature=0.7)

SYSTEM_CHAT_PROMPT = "You are a helpful healthcare assistant."


class DiagnosisAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, symptoms, history="No significant history provided."):
        prompt = f"""
        You are a medical diagnostic assistant. Given symptoms and patient history, provide 2-3 possible diagnoses with explanations and next steps.

        Symptoms:
        {symptoms}

        History:
        {history}

        Diagnoses and recommendations:
        """
        response = self.llm(prompt=prompt, max_tokens=512)
        return response["choices"][0]["text"].strip()


class PrescriptionAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, diagnosis):
        prompt = f"""
        You are a medical assistant providing common prescription options based on diagnosis.

        Diagnosis:
        {diagnosis}

        Prescription suggestions (include disclaimer):
        """
        response = self.llm(prompt=prompt, max_tokens=512)
        return response["choices"][0]["text"].strip()


class EducationAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, term):
        prompt = f"""
        Explain the following medical term or diagnosis in simple language for a patient:

        Term:
        {term}

        Explanation:
        """
        response = self.llm(prompt=prompt, max_tokens=512)
        return response["choices"][0]["text"].strip()


class RAGAgent:
    def __init__(self, milvus_rag, llm):
        self.milvus_rag = milvus_rag
        self.llm = llm

    def run(self, query):
        docs = self.milvus_rag.search(query, top_k=3)
        context = "\n\n".join([doc["text"] for doc in docs])
        prompt = f"""
        You are a helpful healthcare assistant. Use the following context to answer the question.

        <context>
        {context}
        </context>

        <question>
        {query}
        </question>
        """
        response = self.llm(prompt=prompt, max_tokens=512)
        answer = response["choices"][0]["text"].strip()
        return {"answer": answer, "retrieved_docs": docs}


class OrchestratorAgent:
    def __init__(self, diagnosis_agent, prescription_agent, education_agent, rag_agent):
        self.diagnosis_agent = diagnosis_agent
        self.prescription_agent = prescription_agent
        self.education_agent = education_agent
        self.rag_agent = rag_agent

    def route(self, user_input):
        text = user_input.lower()

        # Simple keyword-based routing — extend with NLP intent detection as needed
        if any(kw in text for kw in ["symptom", "diagnose", "diagnosis"]):
            # For demo: treat entire input as symptoms, no history provided
            return self.diagnosis_agent.run(symptoms=user_input)
        elif any(kw in text for kw in ["prescribe", "medication", "medicine"]):
            return self.prescription_agent.run(diagnosis=user_input)
        elif any(kw in text for kw in ["explain", "what is", "define"]):
            return self.education_agent.run(term=user_input)
        else:
            return self.rag_agent.run(user_input)


diagnosis_agent = DiagnosisAgent(llm)
prescription_agent = PrescriptionAgent(llm)
education_agent = EducationAgent(llm)
rag_agent = RAGAgent(milvus_rag, llm)

orchestrator = OrchestratorAgent(
    diagnosis_agent, prescription_agent, education_agent, rag_agent
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/multiagent", methods=["POST"])
def multiagent():
    try:
        data = request.get_json(force=True)
        user_input = data.get("message")
        if not user_input:
            return jsonify({"error": "Message is required."}), 400

        result = orchestrator.route(user_input)

        # RAGAgent returns dict, others return string
        if isinstance(result, dict):
            return jsonify(result)
        else:
            return jsonify({"response": result})

    except Exception as e:
        logger.error(f"Error in /multiagent endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001)
