""" 
A basic Agents AI written with Python Flask framework
"""

from flask import Flask, request, jsonify
from transformers import pipeline
import re

app = Flask(__name__)

# Initialize a text-generation pipeline with a small open-source model
# You can replace "gpt2" with any other compatible model you have locally or from Hugging Face Hub
generator = pipeline("text-generation", model="gpt2")


# External function the agent can call
def get_response_time(url):
    response_times = {
        "learnwithhasan.com": 0.5,
        "google.com": 0.3,
        "openai.com": 0.4,
    }
    return response_times.get(url, "Unknown URL")


def agent_think(user_input):
    """
    Simulate the agent's reasoning step.
    The prompt instructs the model to decide if it needs to call get_response_time.
    """
    prompt = (
        "You are an AI agent. When asked about website response times, "
        "respond with a function call like get_response_time('url') if needed. "
        "Otherwise, answer directly.\n\n"
        f"User: {user_input}\nAgent:"
    )
    # Generate a short response
    outputs = generator(
        prompt, max_length=len(prompt.split()) + 50, num_return_sequences=1
    )
    response = outputs[0]["generated_text"][len(prompt) :].strip()
    return response


@app.route("/agent", methods=["POST"])
def ai_agent():
    data = request.json
    user_input = data.get("input", "")

    # Agent thinks and decides what to do
    agent_response = agent_think(user_input)

    # Check if the agent wants to call the external function
    match = re.search(r"get_response_time\(['\"]([^'\"]+)['\"]\)", agent_response)
    if match:
        url = match.group(1)
        # Call the external function
        result = get_response_time(url)
        # Generate final response incorporating the function result
        final_prompt = (
            f"User: {user_input}\n"
            f"Agent: {agent_response}\n"
            f"Function get_response_time('{url}') returned: {result}\n"
            f"Agent final answer:"
        )
        outputs = generator(
            final_prompt,
            max_length=len(final_prompt.split()) + 50,
            num_return_sequences=1,
        )
        final_answer = outputs[0]["generated_text"][len(final_prompt) :].strip()
        return jsonify({"answer": final_answer})
    else:
        # Return the agent's direct response
        return jsonify({"answer": agent_response})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
