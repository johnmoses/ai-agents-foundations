import sqlite3
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

# --- Database setup (SQLite for simplicity) ---
conn = sqlite3.connect("appointments.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT NOT NULL,
    patient_contact TEXT NOT NULL,
    doctor_name TEXT NOT NULL,
    appointment_datetime TEXT NOT NULL,
    reminder_sent INTEGER DEFAULT 0
)
""")
conn.commit()

# --- Data models ---
class AppointmentRequest(BaseModel):
    patient_name: str
    patient_contact: str  # email or phone
    doctor_name: str
    preferred_datetime: str  # ISO format datetime string

class QueryRequest(BaseModel):
    query: str

# --- Healthcare RAG system setup (same as before, simplified) ---
class HealthcareMultiAgentRAG:
    def __init__(self, data_dir="healthcare_data/"):
        self.loader = PyPDFLoader(data_dir)
        self.documents = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = self.text_splitter.split_documents(self.documents)
        self.embeddings = OllamaEmbeddings()
        self.vector_db = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory="chroma_healthcare_db"
        )
        self.intent_model = "mistral"
        self.llm_model = "llama3:70b"

    def intent_recognizer(self, query: str) -> str:
        system_prompt = (
            "You are a healthcare intent classifier. "
            "Classify the intent of the following patient query into one of: "
            "'diagnosis', 'treatment', 'appointment', 'general', 'escalate'. "
            "Respond with only the intent keyword."
        )
        response = ollama.chat(
            model=self.intent_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        return response['message']['content'].strip().lower()

    def context_retriever(self, query: str, intent: str) -> str:
        search_query = f"{intent}: {query}"
        results = self.vector_db.similarity_search(search_query, k=5)
        return "\n\n".join([doc.page_content for doc in results])

    def escalation_handler(self, intent: str, context: str) -> Optional[str]:
        if intent == "escalate":
            return "Your query requires attention from a healthcare professional. Connecting you now..."
        critical_terms = ["emergency", "urgent", "immediate", "severe"]
        if any(term in context.lower() for term in critical_terms):
            return "This seems urgent. Please contact emergency services or your doctor immediately."
        return None

    def response_generator(self, query: str, context: str) -> str:
        system_prompt = (
            "You are a helpful healthcare assistant. "
            "Use the following context from trusted medical documents to answer the patient's question. "
            "If the answer is not contained in the context, respond cautiously and advise consulting a healthcare professional.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        return response['message']['content']

    def handle_patient_query(self, query: str) -> str:
        intent = self.intent_recognizer(query)
        if intent == "appointment":
            return "To book or check appointments, please use the appointment API."
        context = self.context_retriever(query, intent)
        escalation_msg = self.escalation_handler(intent, context)
        if escalation_msg:
            return escalation_msg
        return self.response_generator(query, context)

# --- Appointment Scheduler Agent ---
class AppointmentScheduler:
    def __init__(self, db_conn):
        self.conn = db_conn

    def is_slot_available(self, doctor_name: str, dt: datetime) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM appointments 
            WHERE doctor_name=? AND appointment_datetime=?
        """, (doctor_name, dt.isoformat()))
        count = cursor.fetchone()[0]
        return count == 0

    def book_appointment(self, patient_name: str, patient_contact: str, doctor_name: str, preferred_datetime: datetime) -> str:
        if not self.is_slot_available(doctor_name, preferred_datetime):
            return "Requested slot is not available. Please choose a different time."
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO appointments (patient_name, patient_contact, doctor_name, appointment_datetime) 
            VALUES (?, ?, ?, ?)
        """, (patient_name, patient_contact, doctor_name, preferred_datetime.isoformat()))
        self.conn.commit()
        return f"Appointment booked successfully for {preferred_datetime.strftime('%Y-%m-%d %H:%M')} with Dr. {doctor_name}."

# --- Reminder Agent ---
import smtplib
from email.mime.text import MIMEText

class ReminderAgent:
    def __init__(self, db_conn, email_host, email_port, email_user, email_password):
        self.conn = db_conn
        self.email_host = email_host
        self.email_port = email_port
        self.email_user = email_user
        self.email_password = email_password

    def send_email(self, to_email: str, subject: str, body: str):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.email_user
        msg['To'] = to_email
        with smtplib.SMTP(self.email_host, self.email_port) as server:
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)

    def send_reminders(self):
        cursor = self.conn.cursor()
        now = datetime.now()
        reminder_window_start = now + timedelta(hours=24)
        reminder_window_end = now + timedelta(hours=25)
        cursor.execute("""
            SELECT id, patient_name, patient_contact, doctor_name, appointment_datetime FROM appointments
            WHERE reminder_sent=0 AND appointment_datetime BETWEEN ? AND ?
        """, (reminder_window_start.isoformat(), reminder_window_end.isoformat()))
        appointments = cursor.fetchall()
        for appt in appointments:
            appt_id, patient_name, patient_contact, doctor_name, appt_dt = appt
            body = (
                f"Dear {patient_name},\n\n"
                f"This is a reminder of your appointment with Dr. {doctor_name} on {appt_dt}.\n"
                "Please contact us if you need to reschedule.\n\nBest regards,\nHealthcare Team"
            )
            try:
                self.send_email(patient_contact, "Appointment Reminder", body)
                cursor.execute("UPDATE appointments SET reminder_sent=1 WHERE id=?", (appt_id,))
                self.conn.commit()
            except Exception as e:
                print(f"Failed to send reminder to {patient_contact}: {e}")

# --- FastAPI app ---
app = FastAPI()

healthcare_rag = HealthcareMultiAgentRAG(data_dir="healthcare_data/")
scheduler = AppointmentScheduler(conn)
# Configure with your SMTP server info
reminder_agent = ReminderAgent(
    conn,
    email_host="smtp.example.com",
    email_port=587,
    email_user="your_email@example.com",
    email_password="your_password"
)

@app.post("/patient_query")
async def patient_query_endpoint(query: QueryRequest):
    response = healthcare_rag.handle_patient_query(query.query)
    return {"response": response}

@app.post("/book_appointment")
async def book_appointment_endpoint(appt: AppointmentRequest):
    try:
        preferred_dt = datetime.fromisoformat(appt.preferred_datetime)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format. Use ISO format.")
    result = scheduler.book_appointment(
        appt.patient_name, appt.patient_contact, appt.doctor_name, preferred_dt
    )
    return {"message": result}

@app.post("/send_reminders")
async def send_reminders_endpoint():
    reminder_agent.send_reminders()
    return {"message": "Reminder sending process completed."}
