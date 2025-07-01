# AI Agents Starter

## App 1

### 1.1. Install Requirements

```bash
pip install flask transformers torch
```

### 1.2. Start the Flask Server

```bash
python app_1.py
```

### 1.3. Send a request to the server

Send a POST request to `http://localhost:5001/agent` with JSON body like:

```json
{
  "input": "What is the response time for learnwithhasan.com?"
}
```

## App 2

### 2.1. Install Requirements

```bash
pip install flask transformers torch
```

### 2.2. Start the Flask Server

```bash
python app_2.py
```

### 2.3. Access dashboard

```.bash
http://localhost:5001
```

### 2.4. Create Agents

```bash
import requests

agent_config = {
    "name": "My Sensor Agent",
    "input_type": "sensor",
    "output_type": "alert",
    "inference_interval": 1.0,
    "max_memory_mb": 256,
    "priority": 7
  }

response = requests.post('http://localhost:5001/api/agents', json=agent_config)
```

### 2.5. Send data to agents

```bash
data = {"sensor_value": 85.2, "timestamp": "2025-06-27T10:30:00"}
requests.post('http://localhost:5001/api/agents/sensor_001/data', json=data)
```

## App 3

### 3.1. Install Requirements

```bash
pip install flask transformers torch
```

### 3.2. Start the Flask Server

```bash
python app_3.py
```

### 3.3. Access dashboard

```.bash
http://localhost:5001
```

### 3.4. Create custom LLM Agents

```bash
import requests

agent_config = {
    "name": "Security Monitor",
    "agent_type": "analysis",
    "model_name": "llama2:7b",
    "model_type": "ollama",
    "system_prompt": "You are a cybersecurity expert analyzing system data for threats.",
    "temperature": 0.3,
    "max_tokens": 1024,
    "context_size": 20
}

response = requests.post('http://localhost:5001/api/agents', json=agent_config)
```

### 3.5. Send data to agents for analysis

```bash
# Sensor data analysis
sensor_data = {
    "sensor_data": {
        "temperature": 85.7,
        "cpu_usage": 92.3,
        "memory_usage": 87.1,
        "disk_io": 156.2
    }
}
requests.post('http://localhost:5001/api/agents/analysis_001/data', json=sensor_data)

# Log analysis
log_data = {
    "log_entries": [
        "2025-06-27 10:30:15 ERROR: Database connection timeout",
        "2025-06-27 10:30:16 WARN: High memory usage detected",
        "2025-06-27 10:30:17 INFO: Attempting reconnection"
    ]
}
requests.post('http://localhost:5001/api/agents/log_001/data', json=log_data)
```

### 3.6. Chat

```bash
# Chat with agent
chat_message = {"message": "What's the current system status?"}
requests.post('http://localhost:5001/api/agents/chat_001/data', json=chat_message)

# Get conversation history
history = requests.get('http://localhost:5001/api/agents/chat_001/history').json()
```
