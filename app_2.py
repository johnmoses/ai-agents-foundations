import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import psutil
import queue
import uuid

from flask import Flask, request, jsonify, render_template_string
from werkzeug.serving import make_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for AI agents"""

    agent_id: str
    name: str
    model_path: str
    input_type: str  # 'sensor', 'camera', 'audio', 'text'
    output_type: str  # 'action', 'alert', 'data'
    inference_interval: float  # seconds
    max_memory_mb: int
    priority: int  # 1-10, higher is more important
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
    error_message: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all AI agents"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.status = AgentStatus(
            agent_id=config.agent_id,
            status="initializing",
            last_inference=None,
            total_inferences=0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
        )
        self.running = False
        self.thread = None
        self.input_queue = queue.Queue(maxsize=100)

    @abstractmethod
    def load_model(self):
        """Load the AI model"""
        pass

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess input data"""
        pass

    @abstractmethod
    def inference(self, data: Any) -> Any:
        """Run inference on preprocessed data"""
        pass

    @abstractmethod
    def postprocess(self, result: Any) -> Any:
        """Postprocess inference results"""
        pass

    def start(self):
        """Start the agent"""
        if self.running:
            return

        try:
            self.load_model()
            self.running = True
            self.status.status = "running"
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            logger.info(f"Agent {self.config.name} started")
        except Exception as e:
            self.status.status = "error"
            self.status.error_message = str(e)
            logger.error(f"Failed to start agent {self.config.name}: {e}")

    def stop(self):
        """Stop the agent"""
        self.running = False
        self.status.status = "stopped"
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
                # Get input data
                if not self.input_queue.empty():
                    data = self.input_queue.get_nowait()

                    # Process data
                    start_time = time.time()
                    preprocessed = self.preprocess(data)
                    result = self.inference(preprocessed)
                    output = self.postprocess(result)

                    # Update statistics
                    self.status.last_inference = datetime.now()
                    self.status.total_inferences += 1
                    self._update_resource_usage()

                    # Handle output
                    self._handle_output(output)

                    # Log inference time
                    inference_time = time.time() - start_time
                    logger.debug(
                        f"Agent {self.config.name} inference time: {inference_time:.3f}s"
                    )

                # Sleep based on inference interval
                time.sleep(self.config.inference_interval)

            except Exception as e:
                self.status.status = "error"
                self.status.error_message = str(e)
                logger.error(f"Error in agent {self.config.name}: {e}")
                time.sleep(1)  # Wait before retrying

    def _update_resource_usage(self):
        """Update resource usage statistics"""
        process = psutil.Process()
        self.status.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        self.status.cpu_usage_percent = process.cpu_percent()

    def _handle_output(self, output: Any):
        """Handle agent output"""
        logger.info(f"Agent {self.config.name} output: {output}")


class DummyVisionAgent(BaseAgent):
    """Example vision agent for demonstration"""

    def load_model(self):
        """Simulate loading a lightweight vision model"""
        time.sleep(1)  # Simulate model loading time
        logger.info(f"Loaded vision model for {self.config.name}")

    def preprocess(self, data: Any) -> Any:
        """Simulate image preprocessing"""
        return {"processed_image": data.get("image", "dummy_image")}

    def inference(self, data: Any) -> Any:
        """Simulate object detection inference"""
        # Simulate processing time
        time.sleep(0.1)
        return {
            "detections": [
                {"class": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300]},
                {"class": "car", "confidence": 0.87, "bbox": [300, 150, 500, 400]},
            ]
        }

    def postprocess(self, result: Any) -> Any:
        """Process detection results"""
        detections = result.get("detections", [])
        high_confidence = [d for d in detections if d["confidence"] > 0.9]
        return {"high_confidence_detections": high_confidence}


class DummySensorAgent(BaseAgent):
    """Example sensor data agent"""

    def load_model(self):
        """Simulate loading a sensor data model"""
        time.sleep(0.5)
        logger.info(f"Loaded sensor model for {self.config.name}")

    def preprocess(self, data: Any) -> Any:
        """Normalize sensor data"""
        return {"normalized_data": data.get("sensor_value", 0) / 100.0}

    def inference(self, data: Any) -> Any:
        """Simulate anomaly detection"""
        time.sleep(0.05)
        normalized_value = data.get("normalized_data", 0)
        is_anomaly = abs(normalized_value) > 0.8
        return {"anomaly_detected": is_anomaly, "confidence": abs(normalized_value)}

    def postprocess(self, result: Any) -> Any:
        """Format anomaly results"""
        if result.get("anomaly_detected"):
            return {"alert": "Anomaly detected", "confidence": result.get("confidence")}
        return {"status": "normal"}


class EdgeAIManager:
    """Main manager for AI agents on edge devices"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.device_info = self._get_device_info()

    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "disk_total_gb": psutil.disk_usage("/").total / 1024 / 1024 / 1024,
            "platform": psutil.os.name,
        }

    def register_agent(self, config: AgentConfig) -> bool:
        """Register a new agent"""
        try:
            if config.agent_id in self.agents:
                raise ValueError(f"Agent {config.agent_id} already exists")

            # Create agent based on input type
            if config.input_type == "camera":
                agent = DummyVisionAgent(config)
            elif config.input_type == "sensor":
                agent = DummySensorAgent(config)
            else:
                # Default to sensor agent
                agent = DummySensorAgent(config)

            self.agents[config.agent_id] = agent
            logger.info(f"Registered agent: {config.name}")
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

    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "temperature": self._get_cpu_temperature(),
            "uptime": time.time() - psutil.boot_time(),
        }

    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature if available"""
        try:
            temps = psutil.sensors_temperatures()
            if "coretemp" in temps:
                return temps["coretemp"][0].current
        except:
            pass
        return None


# Flask application
app = Flask(__name__)
manager = EdgeAIManager()

# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Edge AI Agents Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .status-running { background-color: #d4edda; }
        .status-stopped { background-color: #f8d7da; }
        .status-error { background-color: #fff3cd; }
        .button { padding: 8px 16px; margin: 5px; cursor: pointer; border: none; border-radius: 3px; }
        .button-start { background-color: #28a745; color: white; }
        .button-stop { background-color: #dc3545; color: white; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .refresh-btn { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Edge AI Agents Dashboard</h1>
        
        <div class="card">
            <h2>System Resources</h2>
            <p>CPU: {{ resources.cpu_percent }}% | Memory: {{ resources.memory_percent }}% | Disk: {{ resources.disk_percent }}%</p>
            {% if resources.temperature %}
            <p>CPU Temperature: {{ "%.1f"|format(resources.temperature) }}°C</p>
            {% endif %}
        </div>
        
        <div class="card">
            <h2>Device Information</h2>
            <p>CPU Cores: {{ device_info.cpu_count }} | Memory: {{ "%.1f"|format(device_info.memory_total_gb) }}GB | Platform: {{ device_info.platform }}</p>
        </div>
        
        <div class="card">
            <h2>AI Agents</h2>
            <button class="refresh-btn" onclick="location.reload()">Refresh</button>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Status</th>
                    <th>Type</th>
                    <th>Inferences</th>
                    <th>Memory (MB)</th>
                    <th>Last Inference</th>
                    <th>Actions</th>
                </tr>
                {% for status in agent_statuses %}
                <tr class="status-{{ status.status }}">
                    <td>{{ agents[status.agent_id].name }}</td>
                    <td>{{ status.status }}</td>
                    <td>{{ agents[status.agent_id].input_type }}</td>
                    <td>{{ status.total_inferences }}</td>
                    <td>{{ "%.1f"|format(status.memory_usage_mb) }}</td>
                    <td>{% if status.last_inference %}{{ status.last_inference.strftime('%H:%M:%S') }}{% else %}Never{% endif %}</td>
                    <td>
                        {% if status.status == 'running' %}
                        <button class="button button-stop" onclick="controlAgent('{{ status.agent_id }}', 'stop')">Stop</button>
                        {% else %}
                        <button class="button button-start" onclick="controlAgent('{{ status.agent_id }}', 'start')">Start</button>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </table>
        </div>
    </div>
    
    <script>
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
    </script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    """Main dashboard"""
    agent_configs = {
        agent_id: agent.config for agent_id, agent in manager.agents.items()
    }
    return render_template_string(
        DASHBOARD_HTML,
        agent_statuses=manager.get_all_statuses(),
        agents=agent_configs,
        resources=manager.get_system_resources(),
        device_info=manager.device_info,
    )


@app.route("/api/agents", methods=["GET"])
def get_agents():
    """Get all agents"""
    agents_data = []
    for agent in manager.agents.values():
        agents_data.append(
            {"config": asdict(agent.config), "status": asdict(agent.status)}
        )
    return jsonify(agents_data)


@app.route("/api/agents", methods=["POST"])
def create_agent():
    """Create a new agent"""
    try:
        data = request.json
        config = AgentConfig(
            agent_id=data.get("agent_id", str(uuid.uuid4())),
            name=data["name"],
            model_path=data.get("model_path", ""),
            input_type=data["input_type"],
            output_type=data["output_type"],
            inference_interval=data.get("inference_interval", 1.0),
            max_memory_mb=data.get("max_memory_mb", 512),
            priority=data.get("priority", 5),
        )

        if manager.register_agent(config):
            return jsonify({"success": True, "agent_id": config.agent_id})
        else:
            return jsonify({"success": False, "error": "Failed to register agent"}), 400

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/agents/<agent_id>/start", methods=["POST"])
def start_agent(agent_id):
    """Start an agent"""
    if manager.start_agent(agent_id):
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Agent not found"}), 404


@app.route("/api/agents/<agent_id>/stop", methods=["POST"])
def stop_agent(agent_id):
    """Stop an agent"""
    if manager.stop_agent(agent_id):
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Agent not found"}), 404


@app.route("/api/agents/<agent_id>/status", methods=["GET"])
def get_agent_status(agent_id):
    """Get agent status"""
    status = manager.get_agent_status(agent_id)
    if status:
        return jsonify(asdict(status))
    return jsonify({"error": "Agent not found"}), 404


@app.route("/api/agents/<agent_id>/data", methods=["POST"])
def send_data(agent_id):
    """Send data to an agent"""
    data = request.json
    if manager.send_data_to_agent(agent_id, data):
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Agent not found"}), 404


@app.route("/api/system/resources", methods=["GET"])
def get_system_resources():
    """Get system resources"""
    return jsonify(manager.get_system_resources())


@app.route("/api/system/info", methods=["GET"])
def get_device_info():
    """Get device information"""
    return jsonify(manager.device_info)


def create_sample_agents():
    """Create sample agents for demonstration"""
    # Vision agent
    vision_config = AgentConfig(
        agent_id="vision_001",
        name="Security Camera Agent",
        model_path="/models/yolo_tiny.onnx",
        input_type="camera",
        output_type="alert",
        inference_interval=0.5,
        max_memory_mb=256,
        priority=8,
    )

    # Sensor agent
    sensor_config = AgentConfig(
        agent_id="sensor_001",
        name="Temperature Monitor",
        model_path="/models/anomaly_detector.pkl",
        input_type="sensor",
        output_type="data",
        inference_interval=2.0,
        max_memory_mb=128,
        priority=5,
    )

    manager.register_agent(vision_config)
    manager.register_agent(sensor_config)


if __name__ == "__main__":
    # Create sample agents
    create_sample_agents()

    # Start the Flask server
    print("Starting Edge AI Agents Server...")
    print("Dashboard available at: http://localhost:5001")
    print("API endpoints:")
    print("  GET  /api/agents - List all agents")
    print("  POST /api/agents - Create new agent")
    print("  POST /api/agents/<id>/start - Start agent")
    print("  POST /api/agents/<id>/stop - Stop agent")
    print("  POST /api/agents/<id>/data - Send data to agent")
    print("  GET  /api/system/resources - Get system resources")

    app.run(host="127.0.0.1", port=5001, debug=True)
