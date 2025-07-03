from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Enable evaluation mode and use GPU if available
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def get_response_time(url):
    response_times = {"learnwithhasan.com": 0.5, "google.com": 0.3, "openai.com": 0.4}
    return response_times.get(url, "Unknown URL")


def agent_response(user_input):
    if "response time" in user_input.lower():
        words = user_input.split()
        url = words[-1] if words else ""
        time = get_response_time(url)
        return f"The response time for {url} is {time} seconds."
    else:
        prompt = f"User: {user_input}\nAI:"
        response = generate_text(prompt)
        return response


if __name__ == "__main__":
    print("Welcome to the GPT-2 AI Agent! Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        response = agent_response(user_input)
        print(f"AI: {response}\n")
