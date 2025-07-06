# actions.py


def get_response_time(url):
    times = {
        "learnwithhasan.com": 0.5,
        "google.com": 0.3,
        "openai.com": 0.4,
    }
    return times.get(url.lower(), "Unknown URL")


def get_weather(city):
    weather_data = {
        "london": "Cloudy, 15°C",
        "new york": "Sunny, 22°C",
        "lagos": "Rainy, 28°C",
    }
    return weather_data.get(city.lower(), "Weather data not available")


def calculate_sum(numbers):
    try:
        nums = [float(n) for n in numbers.split(",")]
        return sum(nums)
    except Exception:
        return "Invalid input for sum calculation"
