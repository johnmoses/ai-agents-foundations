import json
import time
import threading
import logging
import requests
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import psutil
import queue
import uuid
import os
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string
from werkzeug.serving import make_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    model_name: str
    model_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    context_window: int = 4096
    model_type: str = "local"  # 'local', 'api', 'ollama'

@dataclass
class AgentConfig:
    """Configuration for LLM-based AI agents"""
    agent_id: str
    name: str
    agent_type: str  # 'chat', 'analysis', 'summarization', 'classification', 'extraction'
    llm_config: LLMConfig
    system_prompt: str
    input_type: str  # 'text', 'sensor_data', 'log_analysis', 'document'
    output_type: str  # 'response', 'classification', 'summary', 'alert'
    inference_interval: float  # seconds
    max_memory_mb: int
    priority: int  # 1-10, higher is more important
    context_size: int = 10  # Number of previous interactions to remember
    enabled: bool = True

@dataclass
class AgentStatus:
    """Status information for an agent"""
    agent_id: str
    status: str  # 'running', 'stopped', 'error', 'initializing'
    last_inference: Optional[datetime]
    total_inferences: int
    memory_usage_mb: float
    cpu_usage_percent: float
    average_response_time: float
    context_length: int
    error_message: Optional[str] = None

@dataclass
class ConversationEntry:
    """Single conversation entry"""
    timestamp: datetime
    role: str  # 'user', 'assistant', 'system'
    content: str
    metadata: Optional[Dict[str, Any]] = None

class LLMInterface(ABC):
    """Abstract interface for LLM interactions"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM is available"""
        pass

class OllamaLLM(LLMInterface):
    """Ollama LLM implementation for local models"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.api_endpoint or "http://localhost:11434"
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            # Convert messages to Ollama format
            prompt = self._format_messages(messages)
            
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Ollama LLM error: {e}")
            return f"Error: {str(e)}"
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Ollama"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"Human: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        return "\n".join(formatted) + "\nAssistant:"
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

class OpenAILLM(LLMInterface):
    """OpenAI API LLM implementation"""
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
            
            response = requests.post(
                self.config.api_endpoint or "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"OpenAI LLM error: {e}")
            return f"Error: {str(e)}"
    
    def is_available(self) -> bool:
        return bool(self.config.api_key)

class MockLLM(LLMInterface):
    """Mock LLM for demonstration purposes"""
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        time.sleep(0.5)  # Simulate processing time
        
        last_message = messages[-1].get("content", "") if messages else ""
        
        # Simple rule-based responses for demonstration
        if "temperature" in last_message.lower():
            return "Based on the temperature data, I observe normal operating conditions. The readings are within acceptable parameters."
        elif "anomaly" in last_message.lower() or "alert" in last_message.lower():
            return "I've detected unusual patterns in the data. Recommend investigating the root cause and monitoring closely."
        elif "summarize" in last_message.lower():
            return "Summary: The system is operating normally with no critical issues detected. All parameters are within expected ranges."
        elif "classify" in last_message.lower():
            return "Classification: NORMAL - No immediate action required."
        else:
            return f"I understand you're asking about: '{last_message[:100]}...'. Based on my analysis, this appears to be a routine inquiry that requires standard monitoring protocols."
    
    def is_available(self) -> bool:
        return True

class BaseLLMAgent(ABC):
    """Base class for LLM-based AI agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.status = AgentStatus(
            agent_id=config.agent_id,
            status='initializing',
            last_inference=None,
            total_inferences=0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            average_response_time=0.0,
            context_length=0
        )
        self.running = False
        self.thread = None
        self.input_queue = queue.Queue(maxsize=100)
        self.conversation_history: List[ConversationEntry] = []
        self.llm: Optional[LLMInterface] = None
        self.response_times: List[float] = []
        
    def load_model(self):
        """Load the LLM model"""
        try:
            if self.config.llm_config.model_type == "ollama":
                self.llm = OllamaLLM(self.config.llm_config)
            elif self.config.llm_config.model_type == "api":
                self.llm = OpenAILLM(self.config.llm_config)
            else:
                self.llm = MockLLM(self.config.llm_config)
            
            if not self.llm.is_available():
                raise Exception("LLM is not available")
            
            # Add system prompt to conversation history
            if self.config.system_prompt:
                self.conversation_history.append(ConversationEntry(
                    timestamp=datetime.now(),
                    role="system",
                    content=self.config.system_prompt
                ))
            
            logger.info(f"Loaded LLM for agent {self.config.name}")
            
        except Exception as e:
            raise Exception(f"Failed to load LLM: {e}")
    
    @abstractmethod
    def preprocess(self, data: Any) -> str:
        """Preprocess input data to text format"""
        pass
    
    @abstractmethod
    def postprocess(self, response: str, original_data: Any) -> Any:
        """Postprocess LLM response"""
        pass
    
    def start(self):
        """Start the agent"""
        if self.running:
            return
        
        try:
            self.load_model()
            self.running = True
            self.status.status = 'running'
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            logger.info(f"Agent {self.config.name} started")
        except Exception as e:
            self.status.status = 'error'
            self.status.error_message = str(e)
            logger.error(f"Failed to start agent {self.config.name}: {e}")
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        self.status.status = 'stopped'
        if self.thread:
            self.thread.join(timeout=5)
        logger.info(f"Agent {self.config.name} stopped")
    
    def add_input(self, data: Any):
        """Add input data to the agent's queue"""
        try:
            self.input_queue.put_nowait(data)
        except queue.Full:
            logger.warning(f"Input queue full for agent {self.config.name}")
    
    def _run_loop(self):
        """Main execution loop for the agent"""
        while self.running:
            try:
                if not self.input_queue.empty():
                    data = self.input_queue.get_nowait()
                    
                    start_time = time.time()
                    
                    # Preprocess input
                    user_input = self.preprocess(data)
                    
                    # Add to conversation history
                    self.conversation_history.append(ConversationEntry(
                        timestamp=datetime.now(),
                        role="user",
                        content=user_input,
                        metadata={"original_data": data}
                    ))
                    
                    # Generate LLM response
                    messages = self._build_messages()
                    response = self.llm.generate_response(messages)
                    
                    # Add response to history
                    self.conversation_history.append(ConversationEntry(
                        timestamp=datetime.now(),
                        role="assistant",
                        content=response
                    ))
                    
                    # Maintain context size
                    self._maintain_context()
                    
                    # Postprocess response
                    output = self.postprocess(response, data)
                    
                    # Update statistics
                    inference_time = time.time() - start_time
                    self._update_statistics(inference_time)
                    
                    # Handle output
                    self._handle_output(output)
                    
                    logger.debug(f"Agent {self.config.name} processed input in {inference_time:.3f}s")
                
                time.sleep(self.config.inference_interval)
                
            except Exception as e:
                self.status.status = 'error'
                self.status.error_message = str(e)
                logger.error(f"Error in agent {self.config.name}: {e}")
                time.sleep(1)
    
    def _build_messages(self) -> List[Dict[str, str]]:
        """Build messages list for LLM"""
        messages = []
        for entry in self.conversation_history:
            messages.append({
                "role": entry.role,
                "content": entry.content
            })
        return messages
    
    def _maintain_context(self):
        """Maintain conversation context within limits"""
        # Keep system message + last N entries
        system_entries = [e for e in self.conversation_history if e.role == "system"]
        other_entries = [e for e in self.conversation_history if e.role != "system"]
        
        if len(other_entries) > self.config.context_size * 2:  # *2 for user+assistant pairs
            other_entries = other_entries[-(self.config.context_size * 2):]
        
        self.conversation_history = system_entries + other_entries
        self.status.context_length = len(self.conversation_history)
    
    def _update_statistics(self, inference_time: float):
        """Update agent statistics"""
        self.status.last_inference = datetime.now()
        self.status.total_inferences += 1
        
        # Update response time
        self.response_times.append(inference_time)
        if len(self.response_times) > 100:  # Keep last 100 times
            self.response_times = self.response_times[-100:]
        self.status.average_response_time = sum(self.response_times) / len(self.response_times)
        
        # Update resource usage
        process = psutil.Process()
        self.status.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        self.status.cpu_usage_percent = process.cpu_percent()
    
    def _handle_output(self, output: Any):
        """Handle agent output"""
        logger.info(f"Agent {self.config.name} output: {output}")

class ChatAgent(BaseLLMAgent):
    """General purpose chat agent"""
    
    def preprocess(self, data: Any) -> str:
        if isinstance(data, dict):
            return data.get("message", str(data))
        return str(data)
    
    def postprocess(self, response: str, original_data: Any) -> Any:
        return {"response": response, "type": "chat"}

class DataAnalysisAgent(BaseLLMAgent):
    """Agent for analyzing sensor/system data"""
    
    def preprocess(self, data: Any) -> str:
        if isinstance(data, dict):
            if "sensor_data" in data:
                sensor_data = data["sensor_data"]
                return f"Analyze this sensor data: {json.dumps(sensor_data, indent=2)}"
            elif "metrics" in data:
                metrics = data["metrics"]
                return f"Please analyze these system metrics: {json.dumps(metrics, indent=2)}"
        return f"Analyze this data: {str(data)}"
    
    def postprocess(self, response: str, original_data: Any) -> Any:
        # Extract key insights or alerts
        if any(word in response.lower() for word in ["alert", "warning", "critical", "anomaly"]):
            return {"analysis": response, "type": "alert", "priority": "high"}
        return {"analysis": response, "type": "normal"}

class LogAnalysisAgent(BaseLLMAgent):
    """Agent for analyzing log files and system events"""
    
    def preprocess(self, data: Any) -> str:
        if isinstance(data, dict):
            if "log_entries" in data:
                logs = data["log_entries"]
                log_text = "\n".join(logs) if isinstance(logs, list) else str(logs)
                return f"Analyze these log entries for issues or patterns:\n{log_text}"
        return f"Analyze this log data: {str(data)}"
    
    def postprocess(self, response: str, original_data: Any) -> Any:
        # Extract severity and recommendations
        severity = "low"
        if any(word in response.lower() for word in ["critical", "error", "failed"]):
            severity = "high"
        elif any(word in response.lower() for word in ["warning", "unusual"]):
            severity = "medium"
        
        return {
            "log_analysis": response,
            "severity": severity,
            "type": "log_analysis"
        }

class ClassificationAgent(BaseLLMAgent):
    """Agent for classifying text or data"""
    
    def preprocess(self, data: Any) -> str:
        if isinstance(data, dict):
            if "text" in data:
                categories = data.get("categories", ["normal", "anomaly"])
                return f"Classify this text into one of these categories {categories}: {data['text']}"
        return f"Classify this content: {str(data)}"
    
    def postprocess(self, response: str, original_data: Any) -> Any:
        # Extract classification from response
        classification = response.split()[0] if response else "unknown"
        confidence = 0.8  # Mock confidence score
        
        return {
            "classification": classification.upper(),
            "confidence": confidence,
            "reasoning": response,
            "type": "classification"
        }

class EdgeLLMManager:
    """Manager for LLM-based AI agents on edge devices"""
    
    def __init__(self):
        self.agents: Dict[str, BaseLLMAgent] = {}
        self.device_info = self._get_device_info()
        
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "disk_total_gb": psutil.disk_usage('/').total / 1024 / 1024 / 1024,
            "platform": psutil.os.name
        }
    
    def register_agent(self, config: AgentConfig) -> bool:
        """Register a new LLM agent"""
        try:
            if config.agent_id in self.agents:
                raise ValueError(f"Agent {config.agent_id} already exists")
            
            # Create agent based on type
            if config.agent_type == 'chat':
                agent = ChatAgent(config)
            elif config.agent_type == 'analysis':
                agent = DataAnalysisAgent(config)
            elif config.agent_type == 'log_analysis':
                agent = LogAnalysisAgent(config)
            elif config.agent_type == 'classification':
                agent = ClassificationAgent(config)
            else:
                agent = ChatAgent(config)  # Default
            
            self.agents[config.agent_id] = agent
            logger.info(f"Registered LLM agent: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {config.name}: {e}")
            return False
    
    def start_agent(self, agent_id: str) -> bool:
        """Start an agent"""
        if agent_id not in self.agents:
            return False
        self.agents[agent_id].start()
        return True
    
    def stop_agent(self, agent_id: str) -> bool:
        """Stop an agent"""
        if agent_id not in self.agents:
            return False
        self.agents[agent_id].stop()
        return True
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get status of an agent"""
        if agent_id not in self.agents:
            return None
        return self.agents[agent_id].status
    
    def get_all_statuses(self) -> List[AgentStatus]:
        """Get status of all agents"""
        return [agent.status for agent in self.agents.values()]
    
    def send_data_to_agent(self, agent_id: str, data: Any) -> bool:
        """Send data to an agent for processing"""
        if agent_id not in self.agents:
            return False
        self.agents[agent_id].add_input(data)
        return True
    
    def get_conversation_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for an agent"""
        if agent_id not in self.agents:
            return []
        
        history = []
        for entry in self.agents[agent_id].conversation_history:
            history.append({
                "timestamp": entry.timestamp.isoformat(),
                "role": entry.role,
                "content": entry.content,
                "metadata": entry.metadata
            })
        return history
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "temperature": self._get_cpu_temperature(),
            "uptime": time.time() - psutil.boot_time()
        }
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature if available"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
        except:
            pass
        return None

# Flask application
app = Flask(__name__)
manager = EdgeLLMManager()

# Enhanced HTML template for LLM dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Edge LLM Agents Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        .card { background: white; border: 1px solid #ddd; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-running { border-left: 5px solid #28a745; }
        .status-stopped { border-left: 5px solid #dc3545; }
        .status-error { border-left: 5px solid #ffc107; }
        .button { padding: 8px 16px; margin: 5px; cursor: pointer; border: none; border-radius: 4px; font-size: 14px; }
        .button-start { background-color: #28a745; color: white; }
        .button-stop { background-color: #dc3545; color: white; }
        .button-chat { background-color: #17a2b8; color: white; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .refresh-btn { background-color: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 20px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric { text-align: center; padding: 15px; background: #e9ecef; border-radius: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #495057; }
        .metric-label { font-size: 14px; color: #6c757d; }
        .chat-container { display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 500px; height: 600px; background: white; border: 1px solid #ccc; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); z-index: 1000; }
        .chat-header { padding: 15px; border-bottom: 1px solid #ddd; background: #f8f9fa; display: flex; justify-content: space-between; align-items: center; }
        .chat-messages { height: 400px; overflow-y: auto; padding: 15px; }
        .chat-input { padding: 15px; border-top: 1px solid #ddd; }
        .chat-input input { width: 80%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .chat-input button { width: 15%; padding: 8px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .message { margin: 10px 0; padding: 8px; border-radius: 4px; }
        .message.user { background: #e3f2fd; text-align: right; }
        .message.assistant { background: #f1f8e9; text-align: left; }
        .overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 999; }
    </style>
</head>
<body>
    <div class="overlay" id="overlay" onclick="closeChat()"></div>
    
    <div class="container">
        <h1>🤖 Edge LLM Agents Dashboard</h1>
        
        <div class="card">
            <h2>📊 System Resources</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(resources.cpu_percent) }}%</div>
                    <div class="metric-label">CPU Usage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(resources.memory_percent) }}%</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(resources.disk_percent) }}%</div>
                    <div class="metric-label">Disk Usage</div>
                </div>
                {% if resources.temperature %}
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(resources.temperature) }}°C</div>
                    <div class="metric-label">CPU Temperature</div>
                </div>
                {% endif %}
            </div>
        </div>
        
        <div class="card">
            <h2>💻 Device Information</h2>
            <p><strong>CPU Cores:</strong> {{ device_info.cpu_count }} | <strong>Memory:</strong> {{ "%.1f"|format(device_info.memory_total_gb) }}GB | <strong>Platform:</strong> {{ device_info.platform }}</p>
        </div>
        
        <div class="card">
            <h2>🤖 LLM Agents</h2>
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Dashboard</button>
            
            <table>
                <tr>
                    <th>Agent Name</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Model</th>
                    <th>Inferences</th>
                    <th>Avg Response</th>
                    <th>Memory (MB)</th>
                    <th>Context</th>
                    <th>Actions</th>
                </tr>
                {% for status in agent_statuses %}
                <tr class="status-{{ status.status }}">
                    <td><strong>{{ agents[status.agent_id].name }}</strong></td>
                    <td>{{ agents[status.agent_id].agent_type }}</td>
                    <td>
                        <span style="text-transform: capitalize;">{{ status.status }}</span>
                        {% if status.error_message %}
                        <br><small style="color: red;">{{ status.error_message }}</small>
                        {% endif %}
                    </td>
                    <td>{{ agents[status.agent_id].llm_config.model_name }}</td>
                    <td>{{ status.total_inferences }}</td>
                    <td>{{ "%.2f"|format(status.average_response_time) }}s</td>
                    <td>{{ "%.1f"|format(status.memory_usage_mb) }}</td>
                    <td>{{ status.context_length }}</td>
                    <td>
                        {% if status.status == 'running' %}
                        <button class="button button-stop" onclick="controlAgent('{{ status.agent_id }}', 'stop')">⏹️ Stop</button>
                        <button class="button button-chat" onclick="openChat('{{ status.agent_id }}', '{{ agents[status.agent_id].name }}')">💬 Chat</button>
                        {% else %}
                        <button class="button button-start" onclick="controlAgent('{{ status.agent_id }}', 'start')">▶️ Start</button>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </table>
        </div>
    </div>
    
    <!-- Chat Interface -->
    <div class="chat-container" id="chatContainer">
        <div class="chat-header">
            <h3 id="chatTitle">Chat with Agent</h3>
            <button onclick="closeChat()" style="background: none; border: none; font-size: 18px; cursor: pointer;">✖️</button>
        </div>
        <div class="chat-messages" id="chatMessages"></div>
        <div class="chat-input">
            <input type="text" id="chatInput" placeholder="Type your message..." onkeypress="handleChatKeypress(event)">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    
    <script>
        let currentAgentId = null;
        
        function controlAgent(agentId, action) {
            fetch(`/api/agents/${agentId}/${action}`, {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        setTimeout(() => location.reload(), 1000);
                    } else {
                        alert('Operation failed: ' + data.error);
                    }
                });
        }
        
        function openChat(agentId, agentName) {
            currentAgentId = agentId;
            document.getElementById('chatTitle').textContent = `Chat with ${agentName}`;
            document.getElementById('chatContainer').style.display = 'block';
            document.getElementById('overlay').style.display = 'block';
            document.getElementById('chatMessages').innerHTML = '';
            loadChatHistory();
        }
        
        function closeChat() {
            document.getElementById('chatContainer').style.display = 'none';
            document.getElementById('overlay').style.display = 'none';
            currentAgentId = null;
        }
        
        function loadChatHistory() {
            if (!currentAgentId) return;
            
            fetch(`/api/agents/${currentAgentId}/history`)
                .then(response => response.json())
                .then(history => {
                    const messagesDiv = document.getElementById('chatMessages');
                    history.forEach(entry => {
                        if (entry.role !== 'system') {
                            addMessageToChat(entry.role, entry.content);
                        }
                    });
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                });
        }
        
        function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message || !currentAgentId) return;
            
            addMessageToChat('user', message);
            input.value = '';
            
            // Send to agent
            fetch(`/api/agents/${currentAgentId}/data`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Poll for response
                    setTimeout(pollForResponse, 1000);
                }
            });
        }
        
        function pollForResponse() {
            if (!currentAgentId) return;
            
            fetch(`/api/agents/${currentAgentId}/history`)
                .then(response => response.json())
                .then(history => {
                    const lastEntry = history[history.length - 1];
                    if (lastEntry && lastEntry.role === 'assistant') {
                        const messagesDiv = document.getElementById('chatMessages');
                        const lastMessage = messagesDiv.lastElementChild;
                        if (!lastMessage || !lastMessage.classList.contains('assistant')) {
                            addMessageToChat('assistant', lastEntry.content);
                        }
                    }
                });
        }
        
        function addMessageToChat(role, content) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.textContent = content;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function handleChatKeypress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main LLM dashboard"""
    agent_configs = {agent_id: agent.config for agent_id, agent in manager.agents.items()}
    return render_template_string(
        DASHBOARD_HTML,
        agent_statuses=manager.get_all_statuses(),
        agents=agent_configs,
        resources=manager.get_system_resources(),
        device_info=manager.device_info
    )

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get all LLM agents"""
    agents_data = []
    for agent in manager.agents.values():
        agents_data.append({
            'config': {
                'agent_id': agent.config.agent_id,
                'name': agent.config.name,
                'agent_type': agent.config.agent_type,
                'llm_config': asdict(agent.config.llm_config),
                'system_prompt': agent.config.system_prompt,
                'input_type': agent.config.input_type,
                'output_type': agent.config.output_type,
                'inference_interval': agent.config.inference_interval,
                'max_memory_mb': agent.config.max_memory_mb,
                'priority': agent.config.priority,
                'context_size': agent.config.context_size,
                'enabled': agent.config.enabled
            },
            'status': asdict(agent.status)
        })
    return jsonify(agents_data)

@app.route('/api/agents', methods=['POST'])
def create_agent():
    """Create a new LLM agent"""
    try:
        data = request.json
        
        # Create LLM configuration
        llm_config = LLMConfig(
            model_name=data.get('model_name', 'mock-llm'),
            model_path=data.get('model_path'),
            api_endpoint=data.get('api_endpoint'),
            api_key=data.get('api_key'),
            max_tokens=data.get('max_tokens', 512),
            temperature=data.get('temperature', 0.7),
            context_window=data.get('context_window', 4096),
            model_type=data.get('model_type', 'local')
        )
        
        # Create agent configuration
        config = AgentConfig(
            agent_id=data.get('agent_id', str(uuid.uuid4())),
            name=data['name'],
            agent_type=data.get('agent_type', 'chat'),
            llm_config=llm_config,
            system_prompt=data.get('system_prompt', 'You are a helpful AI assistant for edge device monitoring and analysis.'),
            input_type=data.get('input_type', 'text'),
            output_type=data.get('output_type', 'response'),
            inference_interval=data.get('inference_interval', 1.0),
            max_memory_mb=data.get('max_memory_mb', 512),
            priority=data.get('priority', 5),
            context_size=data.get('context_size', 10)
        )
        
        if manager.register_agent(config):
            return jsonify({'success': True, 'agent_id': config.agent_id})
        else:
            return jsonify({'success': False, 'error': 'Failed to register agent'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/agents/<agent_id>/start', methods=['POST'])
def start_agent(agent_id):
    """Start an LLM agent"""
    if manager.start_agent(agent_id):
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Agent not found'}), 404

@app.route('/api/agents/<agent_id>/stop', methods=['POST'])
def stop_agent(agent_id):
    """Stop an LLM agent"""
    if manager.stop_agent(agent_id):
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Agent not found'}), 404

@app.route('/api/agents/<agent_id>/status', methods=['GET'])
def get_agent_status(agent_id):
    """Get LLM agent status"""
    status = manager.get_agent_status(agent_id)
    if status:
        return jsonify(asdict(status))
    return jsonify({'error': 'Agent not found'}), 404

@app.route('/api/agents/<agent_id>/data', methods=['POST'])
def send_data(agent_id):
    """Send data to an LLM agent"""
    data = request.json
    if manager.send_data_to_agent(agent_id, data):
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Agent not found'}), 404

@app.route('/api/agents/<agent_id>/history', methods=['GET'])
def get_conversation_history(agent_id):
    """Get conversation history for an agent"""
    history = manager.get_conversation_history(agent_id)
    return jsonify(history)

@app.route('/api/agents/<agent_id>/clear-history', methods=['POST'])
def clear_conversation_history(agent_id):
    """Clear conversation history for an agent"""
    if agent_id in manager.agents:
        # Keep only system messages
        agent = manager.agents[agent_id]
        system_messages = [entry for entry in agent.conversation_history if entry.role == "system"]
        agent.conversation_history = system_messages
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Agent not found'}), 404

@app.route('/api/system/resources', methods=['GET'])
def get_system_resources():
    """Get system resources"""
    return jsonify(manager.get_system_resources())

@app.route('/api/system/info', methods=['GET'])
def get_device_info():
    """Get device information"""
    return jsonify(manager.device_info)

@app.route('/api/models/available', methods=['GET'])
def get_available_models():
    """Get available LLM models"""
    models = {
        "local": [
            {"name": "mock-llm", "description": "Mock LLM for testing", "size": "0MB"},
        ],
        "ollama": [
            {"name": "llama2:7b", "description": "Llama 2 7B model", "size": "3.8GB"},
            {"name": "codellama:7b", "description": "Code Llama 7B", "size": "3.8GB"},
            {"name": "mistral:7b", "description": "Mistral 7B model", "size": "4.1GB"},
            {"name": "phi:2.7b", "description": "Microsoft Phi 2.7B", "size": "1.7GB"},
        ],
        "api": [
            {"name": "gpt-3.5-turbo", "description": "OpenAI GPT-3.5 Turbo", "provider": "OpenAI"},
            {"name": "gpt-4", "description": "OpenAI GPT-4", "provider": "OpenAI"},
        ]
    }
    return jsonify(models)

def create_sample_llm_agents():
    """Create sample LLM agents for demonstration"""
    
    # Chat agent
    chat_llm_config = LLMConfig(
        model_name="mock-llm",
        model_type="local",
        max_tokens=512,
        temperature=0.7
    )
    
    chat_config = AgentConfig(
        agent_id="chat_001",
        name="General Chat Assistant",
        agent_type="chat",
        llm_config=chat_llm_config,
        system_prompt="You are a helpful AI assistant for edge device management. You can help with system monitoring, troubleshooting, and general questions about device operations.",
        input_type="text",
        output_type="response",
        inference_interval=0.5,
        max_memory_mb=256,
        priority=7,
        context_size=15
    )
    
    # Data analysis agent
    analysis_llm_config = LLMConfig(
        model_name="mock-llm",
        model_type="local",
        max_tokens=1024,
        temperature=0.3  # Lower temperature for more consistent analysis
    )
    
    analysis_config = AgentConfig(
        agent_id="analysis_001",
        name="Data Analysis Expert",
        agent_type="analysis",
        llm_config=analysis_llm_config,
        system_prompt="You are a data analysis expert specializing in IoT sensor data and system metrics. Analyze data for patterns, anomalies, and provide actionable insights. Focus on identifying issues that require attention and suggest preventive measures.",
        input_type="sensor_data",
        output_type="alert",
        inference_interval=2.0,
        max_memory_mb=512,
        priority=9,
        context_size=20
    )
    
    # Log analysis agent
    log_llm_config = LLMConfig(
        model_name="mock-llm",
        model_type="local",
        max_tokens=768,
        temperature=0.2  # Even lower temperature for log analysis
    )
    
    log_config = AgentConfig(
        agent_id="log_001",
        name="Log Analysis Specialist",
        agent_type="log_analysis",
        llm_config=log_llm_config,
        system_prompt="You are a system administrator expert specializing in log analysis. Examine log entries for errors, warnings, security issues, and performance problems. Provide clear severity assessments and recommend specific actions.",
        input_type="log_analysis",
        output_type="alert",
        inference_interval=5.0,
        max_memory_mb=384,
        priority=8,
        context_size=25
    )
    
    # Classification agent
    classifier_llm_config = LLMConfig(
        model_name="mock-llm",
        model_type="local",
        max_tokens=256,
        temperature=0.1  # Very low temperature for consistent classification
    )
    
    classifier_config = AgentConfig(
        agent_id="classifier_001",
        name="Content Classifier",
        agent_type="classification",
        llm_config=classifier_llm_config,
        system_prompt="You are a classification expert. Classify input content into predefined categories with high accuracy. Always provide your classification as the first word of your response, followed by your reasoning.",
        input_type="text",
        output_type="classification",
        inference_interval=1.0,
        max_memory_mb=192,
        priority=6,
        context_size=5
    )
    
    # Register all agents
    manager.register_agent(chat_config)
    manager.register_agent(analysis_config)
    manager.register_agent(log_config)
    manager.register_agent(classifier_config)

if __name__ == '__main__':
    # Create sample LLM agents
    create_sample_llm_agents()
    
    # Print startup information
    print("🚀 Starting Edge LLM Agents Server...")
    print("=" * 50)
    print("🌐 Dashboard: http://localhost:5001")
    print("📡 API Base URL: http://localhost:5001/api")
    print("")
    print("📋 Available API Endpoints:")
    print("  GET  /api/agents                    - List all LLM agents")
    print("  POST /api/agents                    - Create new LLM agent")
    print("  POST /api/agents/<id>/start         - Start LLM agent")
    print("  POST /api/agents/<id>/stop          - Stop LLM agent")
    print("  POST /api/agents/<id>/data          - Send data to agent")
    print("  GET  /api/agents/<id>/history       - Get conversation history")
    print("  POST /api/agents/<id>/clear-history - Clear conversation history")
    print("  GET  /api/system/resources          - Get system resources")
    print("  GET  /api/models/available          - Get available LLM models")
    print("")
    print("🤖 Sample Agents Created:")
    print("  • General Chat Assistant    (chat)")
    print("  • Data Analysis Expert      (analysis)")
    print("  • Log Analysis Specialist   (log_analysis)")
    print("  • Content Classifier        (classification)")
    print("")
    print("💡 Usage Examples:")
    print("  • Chat: {'message': 'How is the system performing?'}")
    print("  • Analysis: {'sensor_data': {'temp': 75.2, 'humidity': 45}}")
    print("  • Logs: {'log_entries': ['ERROR: Connection failed', 'WARN: High CPU']}")
    print("  • Classify: {'text': 'System running normally', 'categories': ['normal', 'warning', 'error']}")
    print("=" * 50)
    
    app.run(host='127.0.0.1', port=5001, debug=True)