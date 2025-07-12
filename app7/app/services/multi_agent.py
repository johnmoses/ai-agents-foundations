from typing import List, Dict, Optional
import re

# Import your existing RAG and LLM services
from app.services.service import generate_quiz_text

class BaseAgent:
    """Abstract base class for all agents."""
    def run(self, query: str, history: List[Dict]) -> str:
        raise NotImplementedError("Agents must implement the run method.")

class RagAgent(BaseAgent):
    """Agent that uses RAG to answer questions."""
    def __init__(self, rag_service, llm_callable):
        self.rag = rag_service
        self.llm = llm_callable

    def run(self, query: str, history: List[Dict]) -> str:
        return self.rag.chat(query, history, self.llm)

class CalculatorAgent(BaseAgent):
    """Agent that evaluates simple arithmetic expressions."""
    def run(self, query: str, history: List[Dict]) -> str:
        # Extract simple math expressions using regex (basic example)
        expr = self.extract_expression(query)
        if not expr:
            return "Sorry, I couldn't find a math expression to calculate."
        try:
            # Safe eval: allow only numbers and operators
            if not re.match(r'^[0-9+\-*/().\s]+$', expr):
                return "Sorry, I can only calculate basic arithmetic expressions."
            result = eval(expr, {"__builtins__": {}})
            return f"The result is {result}."
        except Exception:
            return "Sorry, I couldn't calculate that expression."

    def extract_expression(self, text: str) -> Optional[str]:
        # Simple heuristic to extract math expression from text
        match = re.search(r'([-+/*()\d\s.]+)', text)
        return match.group(1).strip() if match else None

class QuizAgent(BaseAgent):
    """Agent that generates quizzes on given topics."""
    def run(self, query: str, history: List[Dict]) -> str:
        # Extract topic and number of questions (basic parsing)
        topic, num_questions = self.parse_quiz_request(query)
        if not topic:
            return "Please specify a topic for the quiz."
        try:
            questions = generate_quiz_text(topic, num_questions)
            # Format questions nicely
            formatted = "\n\n".join(
                f"Q{i+1}: {q['question']}\nOptions: {', '.join(q['options'])}"
                for i, q in enumerate(questions)
            )
            return f"Here is your quiz on {topic}:\n\n{formatted}"
        except Exception as e:
            return f"Sorry, I couldn't generate the quiz: {str(e)}"

    def parse_quiz_request(self, text: str):
        # Basic parsing: look for "quiz on X with Y questions"
        topic_match = re.search(r'quiz on ([\w\s]+)', text, re.IGNORECASE)
        num_match = re.search(r'(\d+) questions', text)
        topic = topic_match.group(1).strip() if topic_match else None
        num_questions = int(num_match.group(1)) if num_match else 5
        return topic, num_questions

class AgentOrchestrator:
    """Routes queries to appropriate agents based on simple rules."""
    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents

    def select_agent(self, query: str) -> str:
        query_lower = query.lower()
        # Priority-based keyword routing
        if any(word in query_lower for word in ['calculate', 'sum', 'plus', 'minus', 'times', 'divide']):
            return 'calculator'
        if 'quiz' in query_lower:
            return 'quiz'
        # Default to RAG agent
        return 'rag'

    def handle_query(self, query: str, history: List[Dict]) -> str:
        agent_key = self.select_agent(query)
        agent = self.agents.get(agent_key)
        if not agent:
            return "Sorry, I don't know how to handle that request."
        return agent.run(query, history)
