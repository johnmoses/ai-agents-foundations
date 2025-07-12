from llama_cpp import Llama
import json
import re
from .rag import MilvusRAG

rag = MilvusRAG("milvus_rag_db.db")

# Create collection
rag.create_collection()

# Seed db
rag.seed_data()

llm = Llama(
    model_path="/Users/johnmoses/.cache/lm-studio/models/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    n_ctx=2048,
    temperature=0.7,
)

def call_llm(prompt: str, max_tokens=512, temperature=0.3) -> dict:
    return llm(prompt, max_tokens=max_tokens, temperature=temperature)


def clean_json_string(text: str) -> str:
    text = re.sub(r"``````", "", text, flags=re.DOTALL)  # Remove code blocks
    text = text.replace("`", "")  # Remove backticks
    text = text.strip()
    text = re.sub(r",\s*([\]}])", r"\1", text)  # Remove trailing commas
    return text


def extract_json_array(text: str):
    """
    Extract and parse the first JSON array found in the text.
    Ignores any trailing text after the JSON array.
    """
    try:
        # Find the first '[' and last matching ']' that closes the array properly
        start = text.find("[")
        if start == -1:
            print("No opening '[' found in text.")
            return None

        # Use a stack to find the matching closing bracket for the first '['
        stack = []
        for i in range(start, len(text)):
            if text[i] == "[":
                stack.append("[")
            elif text[i] == "]":
                stack.pop()
                if not stack:
                    end = i
                    break
        else:
            print("No matching closing ']' found for JSON array.")
            return None

        json_str = text[start : end + 1]

        # Clean markdown artifacts
        json_str = json_str.strip("` \n\r\t")

        # Fix trailing commas before closing brackets if any
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

        return json.loads(json_str)

    except json.JSONDecodeError as e:
        print(f"JSON extraction error: {e}")
        print(f"Cleaned JSON string was:\n{json_str}")
        return None


def call_ai_model(prompt: str) -> str:
    response = llm(prompt, max_tokens=1024, temperature=0.3)
    return response["choices"][0]["text"].strip()


def generate_quiz_text(topic: str, num_questions: int = 5):
    prompt = f"""
    Generate a quiz on the topic "{topic}" with {num_questions} multiple-choice questions.
    Return ONLY a JSON array with objects containing these keys:
    - "question": string
    - "options": list of strings
    - "correct_answer": string (one of the options)

    Example format:
    [
    {{
        "question": "What is the capital of France?",
        "options": ["Paris", "London", "Berlin"],
        "correct_answer": "Paris"
    }},
    {{
        "question": "Which planet is known as the Red Planet?",
        "options": ["Earth", "Mars", "Jupiter"],
        "correct_answer": "Mars"
    }}
    ]

    Do NOT include any explanations, introductions, or markdown formatting.
    Output strictly the JSON array.
    """

    raw_response = call_ai_model(prompt)
    questions = extract_json_array(raw_response)

    if not questions:
        raise ValueError(
            "Failed to extract valid JSON quiz questions from model output."
        )

    # Validate questions
    for idx, q in enumerate(questions):
        if not isinstance(q, dict):
            raise ValueError(f"Question at index {idx} is not a JSON object.")
        if "question" not in q or "options" not in q or "correct_answer" not in q:
            raise ValueError(f"Question at index {idx} missing required fields.")
        if not isinstance(q["options"], list) or len(q["options"]) < 2:
            raise ValueError(f"Question at index {idx} has invalid options list.")
        if q["correct_answer"] not in q["options"]:
            raise ValueError(
                f"Question at index {idx} has correct_answer not in options."
            )

    return questions


def generate_homework_explanation(question: str) -> str:
    """
    Generate a clear explanation for a homework question.
    """
    prompt = f"Explain the following homework question clearly:\n{question}"
    response = llm(prompt, max_tokens=512)
    return response["choices"][0]["text"].strip()


def generate_summary(content: str) -> str:
    """
    Generate a concise summary of the given content.
    """
    prompt = f"Summarize the following content concisely:\n{content}"
    response = llm(prompt, max_tokens=256)
    return response["choices"][0]["text"].strip()


def generate_chat_response(history: list, message: str) -> str:
    """
    Generate a chat response continuing the conversation.

    Args:
        history (list): List of dicts with keys 'user' and 'ai' representing past turns.
        message (str): The latest user message.

    Returns:
        str: The AI assistant's response.
    """
    prompt = "You are a friendly, helpful tutor. Continue the conversation:\n"
    for turn in history:
        prompt += f"User: {turn['user']}\nAI: {turn['ai']}\n"
    prompt += f"User: {message}\nAI:"

    response = llm(prompt, max_tokens=256)
    return response["choices"][0]["text"].strip()
