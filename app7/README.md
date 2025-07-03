# Healthcare app with Agentic Ai

This is an Agentic AI

## Features

- Data ingestion: Loads healthcare-related PDFs (e.g., clinical guidelines, patient leaflets, treatment protocols) from a local folder.
- Intent recognition: Classifies queries into healthcare intents like diagnosis, treatment, appointment scheduling, or escalation to human staff.
- Contextual retrieval: Uses vector search over medical documents with intent-aware query expansion.
- Escalation logic: Automatically detects urgent or complex queries to recommend human intervention.
- Safe response generation: The LLM is prompted to answer carefully, advising professional consultation when uncertain.
- Multi-agent orchestration: Separate agents for intent recognition, retrieval, escalation, and generation improve modularity and reliability.
- Local open-source stack: Uses Ollama for running LLMs locally, Chroma for vector DB, and LangChain utilities for document processing.
- Appointment DB integration: Use a relational database (e.g., SQLite or MySQL) to store and query appointment slots and patient info.
- Appointment Scheduler Agent: Checks doctor availability, books appointments, updates DB.
- Reminder Agent: Sends reminders to patients via email or SMS (using e.g., SMTP or Twilio).
- API endpoints: For booking appointments and triggering reminders.
- Agent orchestration: The main handler routes queries to scheduling or info retrieval accordingly.
- Appointment scheduling: Patients send requests with preferred date/time and doctor; the scheduler checks availability and books if free.
- Reminder sending: A separate agent scans upcoming appointments within 24-25 hours and sends reminder emails, marking them as sent.
- Patient queries: Routed through the RAG system for general healthcare info; appointment-related queries are redirected to the booking API.
- Database: SQLite used here for simplicity; can be replaced with MySQL/PostgreSQL for production.
- Email reminders: Uses SMTP; you can replace with SMS or other notification services (e.g., Twilio).
