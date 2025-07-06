from gpt4all import GPT4All
from actions import get_response_time, get_weather, calculate_sum


def build_prompt(user_input, history):
    system_instructions = """
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
    conversation = system_instructions + "\n"
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


function_map = {
    "get_response_time": get_response_time,
    "get_weather": get_weather,
    "calculate_sum": calculate_sum,
}


def main():
    model_path = "/Users/johnmoses/.cache/lm-studio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
    model = GPT4All(model_path)
    history = []

    print("Welcome to the AI agent. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        prompt = build_prompt(user_input, history)
        raw_response = model.generate(prompt, max_tokens=150).strip()

        func_name, arg = parse_function_call(raw_response)
        if func_name in function_map:
            func_result = function_map[func_name](arg)
            followup_prompt = f"{prompt}\nFUNCTION_RESULT: {func_result}\nAI:"
            final_response = model.generate(followup_prompt, max_tokens=150).strip()
            print("AI:", final_response)
            history.append({"user": user_input, "ai": final_response})
        else:
            print("AI:", raw_response)
            history.append({"user": user_input, "ai": raw_response})


if __name__ == "__main__":
    main()
