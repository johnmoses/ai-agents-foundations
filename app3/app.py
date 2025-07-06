from langchain_community.llms import GPT4All
from langchain.agents import initialize_agent, Tool
from actions import get_response_time, get_weather, calculate_sum

def main():
    # Path to your local GPT4All model file
    model_path = "/Users/johnmoses/.cache/lm-studio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"  # Adjust accordingly

    # Initialize GPT4All LLM with threading for performance
    llm = GPT4All(model=model_path, n_threads=8)

    # Wrap your external functions as LangChain Tools
    tools = [
        Tool(
            name="GetResponseTime",
            func=get_response_time,
            description="Get website response time for a given URL."
        ),
        Tool(
            name="GetWeather",
            func=get_weather,
            description="Get weather information for a given city."
        ),
        Tool(
            name="CalculateSum",
            func=calculate_sum,
            description="Calculate the sum of a list of numbers separated by commas."
        ),
    ]

    # Initialize a zero-shot agent that can reason and call tools
    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True,
    )

    print("Multi-agent LangChain AI system with GPT4All. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Run the agent on the user input
        response = agent.run(user_input)
        print(f"AI: {response}\n")

if __name__ == "__main__":
    main()
