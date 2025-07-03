import re
import torch
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# --- Setup NLU pipeline for intent classification (zero-shot) ---
# Using Hugging Face zero-shot-classification pipeline for intent detection
nlu_classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

# --- Setup local GPT-2 for generation ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_model.eval()


def generate_text(prompt, max_length=100, temperature=0.7):
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)
    output_ids = gpt2_model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        no_repeat_ngram_size=2,
        pad_token_id=gpt2_tokenizer.eos_token_id,
    )
    generated = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated[len(prompt) :].strip()


# --- Agents ---


class FlightAgent:
    def book_flight(self, destination, date):
        return f"Flight to {destination} on {date} has been booked."


class WeatherAgent:
    def get_weather(self, city):
        # In reality, integrate a weather API or knowledge base
        return f"The weather in {city} is sunny with a temperature of 25°C."


class BudgetAgent:
    def calculate_budget(self, amount, expenses):
        remaining = amount - sum(expenses)
        return f"Your remaining budget is ${remaining}."


# --- Manager with NLU and LLM integration ---


class TravelAssistantManager:
    def __init__(self):
        self.flight_agent = FlightAgent()
        self.weather_agent = WeatherAgent()
        self.budget_agent = BudgetAgent()

    def nlu_parse(self, text):
        # Define candidate intents
        candidate_labels = [
            "book flight",
            "check weather",
            "calculate budget",
            "small talk",
            "unknown",
        ]
        result = nlu_classifier(text, candidate_labels)
        top_intent = result["labels"][0]
        return top_intent

    def extract_entities(self, intent, text):
        # Simple regex-based entity extraction per intent
        if intent == "book flight":
            dest_match = re.search(r"to ([\w\s]+)", text, re.I)
            date_match = re.search(r"on ([\w\s\d]+)", text, re.I)
            destination = (
                dest_match.group(1).strip() if dest_match else "unknown destination"
            )
            date = date_match.group(1).strip() if date_match else "unknown date"
            return {"destination": destination, "date": date}

        elif intent == "check weather":
            city_match = re.search(r"weather in ([\w\s]+)", text, re.I)
            city = city_match.group(1).strip() if city_match else "unknown city"
            return {"city": city}

        elif intent == "calculate budget":
            amount_match = re.search(r"(\d+)", text)
            expenses_match = re.findall(r"(\d+)", text)
            if amount_match and expenses_match:
                amount = int(expenses_match[0])
                expenses = list(map(int, expenses_match[1:]))
                return {"amount": amount, "expenses": expenses}
            return {}

        return {}

    def handle_request(self, text):
        intent = self.nlu_parse(text)
        entities = self.extract_entities(intent, text)

        if intent == "book flight":
            return self.flight_agent.book_flight(
                entities.get("destination", "unknown"), entities.get("date", "unknown")
            )

        elif intent == "check weather":
            return self.weather_agent.get_weather(entities.get("city", "unknown"))

        elif intent == "calculate budget":
            amount = entities.get("amount", 0)
            expenses = entities.get("expenses", [])
            if amount and expenses:
                return self.budget_agent.calculate_budget(amount, expenses)
            else:
                return "Sorry, I couldn't extract budget details properly."

        elif intent == "small talk":
            # Use GPT-2 to generate a small talk response
            prompt = f"User: {text}\nAI:"
            return generate_text(prompt)

        else:
            # Unknown intent fallback with GPT-2
            prompt = f"User: {text}\nAI:"
            return generate_text(prompt)


# --- CLI loop ---


def main():
    print("Welcome to the NLU & Local LLM Multi-Agent Travel Assistant CLI!")
    print("Type your requests or 'exit' to quit.\n")

    manager = TravelAssistantManager()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            print("Please enter a valid request.")
            continue

        response = manager.handle_request(user_input)
        print(f"AI: {response}\n")


if __name__ == "__main__":
    main()
