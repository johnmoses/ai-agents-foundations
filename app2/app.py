import re

# --- Agents ---


class FlightAgent:
    def book_flight(self, destination, date):
        return f"Flight to {destination} on {date} has been booked."


class WeatherAgent:
    def get_weather(self, city):
        return f"The weather in {city} is sunny with a temperature of 25°C."


class BudgetAgent:
    def calculate_budget(self, amount, expenses):
        remaining = amount - sum(expenses)
        return f"Your remaining budget is ${remaining}."


# --- Manager ---


class TravelAssistantManager:
    def __init__(self):
        self.flight_agent = FlightAgent()
        self.weather_agent = WeatherAgent()
        self.budget_agent = BudgetAgent()

    def parse_request(self, request_text):
        tasks = {}

        # Flight booking extraction
        flight_match = re.search(
            r"book (?:a )?flight to ([\w\s]+) on ([\w\s\d]+)", request_text, re.I
        )
        if flight_match:
            tasks["flight"] = {
                "destination": flight_match.group(1).strip(),
                "date": flight_match.group(2).strip(),
            }

        # Weather extraction
        weather_match = re.search(r"weather in ([\w\s]+)", request_text, re.I)
        if weather_match:
            tasks["weather"] = {"city": weather_match.group(1).strip()}

        # Budget extraction
        budget_match = re.search(
            r"budget.*?(\d+).*?expenses.*?([\d,\s]+)", request_text, re.I
        )
        if budget_match:
            amount = int(budget_match.group(1))
            expenses_str = budget_match.group(2)
            expenses = [int(x.strip()) for x in expenses_str.split(",")]
            tasks["budget"] = {"amount": amount, "expenses": expenses}

        return tasks

    def handle_request(self, request_text):
        tasks = self.parse_request(request_text)
        responses = []

        if "flight" in tasks:
            dest = tasks["flight"]["destination"]
            date = tasks["flight"]["date"]
            responses.append(self.flight_agent.book_flight(dest, date))

        if "weather" in tasks:
            city = tasks["weather"]["city"]
            responses.append(self.weather_agent.get_weather(city))

        if "budget" in tasks:
            amount = tasks["budget"]["amount"]
            expenses = tasks["budget"]["expenses"]
            responses.append(self.budget_agent.calculate_budget(amount, expenses))

        if not responses:
            responses.append(
                "Sorry, I couldn't understand your request. Please try again."
            )

        return "\n".join(responses)


# --- Main CLI loop ---


def main():
    print("Welcome to the Multi-Agent Travel Assistant CLI!")
    print("Type your requests or 'exit' to quit.\n")
    manager = TravelAssistantManager()

    example_queries = [
        "Book a flight to Paris on December 20th, check the weather in Paris, and calculate my remaining budget if I have 2000 and expenses of 500, 300.",
        "Check the weather in Tokyo and book a flight to Tokyo on August 5th.",
        "Calculate my remaining budget if I have 1500 and expenses of 400, 200.",
    ]

    print("Example queries you can try:")
    for q in example_queries:
        print(f" - {q}")
    print("\n")

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
