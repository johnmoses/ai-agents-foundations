from gpt4all import GPT4All
from actions import get_response_time, get_weather, calculate_sum


class Agent:
    def __init__(self, name, model_path, system_prompt):
        self.name = name
        self.model = GPT4All(model_path)
        self.system_prompt = system_prompt
        self.history = []

    def build_prompt(self, user_input):
        conversation = self.system_prompt + "\n"
        for turn in self.history:
            conversation += f"User: {turn['user']}\n{self.name}: {turn['agent']}\n"
        conversation += f"User: {user_input}\n{self.name}:"
        return conversation

    def generate_response(self, user_input):
        prompt = self.build_prompt(user_input)
        response = self.model.generate(prompt, max_tokens=150).strip()
        self.history.append({"user": user_input, "agent": response})
        return response


def build_function_prompt(user_input, history):
    instructions = """
    You are an AI assistant that can answer questions or call external functions.
    Available functions:
    - get_response_time(url)
    - get_weather(city)
    - calculate_sum(numbers)

    If the user asks about website response times, respond with:
    FUNCTION_CALL: get_response_time("url")

    If the user asks about weather, respond with:
    FUNCTION_CALL: get_weather("city")

    If the user wants to calculate the sum of numbers, respond with:
    FUNCTION_CALL: calculate_sum("num1,num2,...")

    Otherwise, answer normally.
    """
    conversation = instructions + "\n"
    for turn in history:
        conversation += f"User: {turn['user']}\nAI: {turn['ai']}\n"
    conversation += f"User: {user_input}\nAI:"
    return conversation


def parse_function_call(response):
    if response.startswith("FUNCTION_CALL:"):
        call_str = response[len("FUNCTION_CALL:") :].strip()
        if call_str.startswith("get_response_time(") and call_str.endswith(")"):
            url = call_str[len("get_response_time(") : -1].strip().strip('"').strip("'")
            return ("get_response_time", url)
        elif call_str.startswith("get_weather(") and call_str.endswith(")"):
            city = call_str[len("get_weather(") : -1].strip().strip('"').strip("'")
            return ("get_weather", city)
        elif call_str.startswith("calculate_sum(") and call_str.endswith(")"):
            numbers = call_str[len("calculate_sum(") : -1].strip().strip('"').strip("'")
            return ("calculate_sum", numbers)
    return (None, None)


class MultiAgentCoordinator:
    def __init__(self, agents, model_path):
        self.agents = agents
        self.active_agent = agents["travel_advisor"]
        self.model = GPT4All(model_path)
        self.history = []

        self.function_map = {
            "get_response_time": get_response_time,
            "get_weather": get_weather,
            "calculate_sum": calculate_sum,
        }

    def route_query(self, user_input):
        lower_input = user_input.lower()
        # Simple keyword routing
        if "hotel" in lower_input:
            self.active_agent = self.agents["hotel_advisor"]
        elif "weather" in lower_input:
            self.active_agent = self.agents["weather_agent"]
        elif "travel" in lower_input or "destination" in lower_input:
            self.active_agent = self.agents["travel_advisor"]

        # Let the active agent respond
        response = self.active_agent.generate_response(user_input)

        # Check for function call in response
        func_name, arg = parse_function_call(response)
        if func_name in self.function_map:
            func_result = self.function_map[func_name](arg)
            # Build prompt with function result for final answer
            followup_prompt = build_function_prompt(user_input, self.history)
            followup_prompt += f"\nFUNCTION_RESULT: {func_result}\nAI:"
            final_response = self.model.generate(
                followup_prompt, max_tokens=150
            ).strip()
            self.history.append({"user": user_input, "ai": final_response})
            return final_response

        # Check if agent requests handoff
        if "please ask hoteladvisor" in response.lower():
            self.active_agent = self.agents["hotel_advisor"]
            response = self.active_agent.generate_response(user_input)
        elif "please ask traveladvisor" in response.lower():
            self.active_agent = self.agents["travel_advisor"]
            response = self.active_agent.generate_response(user_input)

        self.history.append({"user": user_input, "ai": response})
        return response


def main():
    model_path = "/Users/johnmoses/.cache/lm-studio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"  # or your preferred model

    travel_advisor_prompt = """
    You are TravelAdvisor, an expert in travel destinations.
    If the user asks about hotels, say: "Please ask HotelAdvisor for hotel recommendations."
    Answer travel-related questions clearly.
    """

    hotel_advisor_prompt = """
    You are HotelAdvisor, an expert in hotel recommendations.
    If the user asks about travel destinations, say: "Please ask TravelAdvisor for travel destination advice."
    Answer hotel-related questions clearly.
    """

    weather_agent_prompt = """
    You are WeatherAgent, an expert in weather information.
    Answer weather-related questions clearly.
    """

    agents = {
        "travel_advisor": Agent("TravelAdvisor", model_path, travel_advisor_prompt),
        "hotel_advisor": Agent("HotelAdvisor", model_path, hotel_advisor_prompt),
        "weather_agent": Agent("WeatherAgent", model_path, weather_agent_prompt),
    }

    coordinator = MultiAgentCoordinator(agents, model_path)

    print("Multi-agent AI system. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = coordinator.route_query(user_input)
        print(f"AI ({coordinator.active_agent.name}): {response}\n")


if __name__ == "__main__":
    main()
