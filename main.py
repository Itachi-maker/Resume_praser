import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from io import BytesIO

# Third-party libraries
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from pymongo import MongoClient
from docx import Document as DocxDocument
from pypdf import PdfReader
from huggingface_hub import InferenceClient  # ✅ Official HF client

# --- Configuration and Setup ---

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "resumes")
HF_API_KEY = os.getenv("HF_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "resume_db")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "candidates")

# Hugging Face Models
HF_EXTRACTION_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_QA_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# Required env check
for env in ["SUPABASE_URL", "SUPABASE_KEY", "HF_API_KEY", "MONGO_URI"]:
    if not os.getenv(env):
        logger.warning(f"⚠️ Missing environment variable: {env}")

# --- Pydantic Models ---

class CandidateDetail(BaseModel):
    candidate_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    education: Dict[str, Any] = {}
    experience: Dict[str, Any] = {}
    skills: List[str] = []
    hobbies: List[str] = []
    certifications: List[str] = []  # ✅ flattened to strings
    projects: List[str] = []        # ✅ flattened to strings
    introduction: Optional[str] = None

class CandidateSummary(BaseModel):
    candidate_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    skills: List[str] = []

class HealthStatus(BaseModel):
    status: str

class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    answer: str
    question: str
    candidate_id: str


# --- Database Setup ---

mongo_client = None
if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI)
        mongo_db = mongo_client[MONGO_DB_NAME]
        candidates_collection = mongo_db[MONGO_COLLECTION_NAME]
        logger.info("✅ MongoDB connection successful.")
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        mongo_client = None


# --- File Extraction Helpers ---

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(file_content))
        return "".join([page.extract_text() or "" for page in reader.pages]).strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    try:
        document = DocxDocument(BytesIO(file_content))
        return "\n".join(p.text for p in document.paragraphs).strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return ""


# --- Supabase Helpers ---

def upload_file_to_supabase(file_data: bytes, file_path: str) -> str:
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{file_path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/octet-stream"
    }
    response = requests.post(url, headers=headers, data=file_data)
    if response.status_code not in [200, 201]:
        logger.error(f"Supabase upload failed: {response.text}")
        raise HTTPException(500, f"Failed to upload file: {response.text}")
    return f"{SUPABASE_BUCKET}/{file_path}"

def save_metadata_to_supabase(metadata: Dict[str, Any]) -> str:
    url = f"{SUPABASE_URL}/rest/v1/resumes"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    response = requests.post(url, headers=headers, data=json.dumps(metadata))
    if response.status_code != 201:
        logger.error(f"Supabase metadata insert failed: {response.text}")
        raise HTTPException(500, f"Failed to save metadata: {response.text}")
    return response.json()[0]['id']

def get_supabase_public_url(storage_path: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/public/{storage_path}"


# --- Hugging Face Inference (REFACTORED) ---

hf_client = InferenceClient(api_key=HF_API_KEY)

def call_text_generation(model_id: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        prompt_text = payload.get("inputs", "")
        params = payload.get("parameters", {"max_new_tokens": 256})

        if not prompt_text:
            raise ValueError("Payload must contain an 'inputs' key with a prompt string.")

        response_text = hf_client.text_generation(
            model=model_id,
            prompt=prompt_text,
            **params
        )
        return [{"generated_text": response_text}]
    except Exception as e:
        logger.error(f"Hugging Face text_generation error for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Hugging Face (text-gen) inference error: {e}"
        )

def call_chat_completion(model_id: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        prompt_text = payload.get("inputs", "")
        params = payload.get("parameters", {})
        max_tokens = params.get("max_tokens", 1024)
        temperature = params.get("temperature", 0.1)

        if not prompt_text:
            raise ValueError("Payload must contain an 'inputs' key with a prompt string.")

        messages = [{"role": "user", "content": prompt_text}]

        response = hf_client.chat_completion(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        generated_text = response.choices[0].message.content
        return [{"generated_text": generated_text}]
    except Exception as e:
        logger.error(f"Hugging Face chat_completion error for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Hugging Face (chat) inference error: {e}"
        )


# --- Resume Processing ---

def process_resume_with_hf(resume_text: str) -> CandidateDetail:
    schema = {
        "name": "string",
        "email": "string",
        "phone": "string",
        "education": "object",
        "experience": "object",
        "skills": "list",
        "hobbies": "list",
        "certifications": "list",
        "projects": "list",
        "introduction": "string"
    }

    prompt = f"""
    You are a resume parsing AI. Extract information into a JSON object following this schema:
    {json.dumps(schema, indent=2)}

    Resume:
    ---
    {resume_text}
    ---
    Output ONLY a valid JSON object:
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_tokens": 1024,
            "temperature": 0.1
        }
    }

    response_data = call_chat_completion(HF_EXTRACTION_MODEL, payload)
    generated_text = response_data[0].get("generated_text", "")

    try:
        json_str = generated_text[generated_text.find("{"):generated_text.rfind("}")+1]
        data = json.loads(json_str)
        
        # --- Flatten certifications and projects to string lists ---
        data['certifications'] = [c.get("name") for c in data.get("certifications", []) if isinstance(c, dict)]
        data['projects'] = [p.get("name") for p in data.get("projects", []) if isinstance(p, dict)]

        return CandidateDetail(**data, candidate_id="temp")
    except Exception as e:
        logger.error(f"Failed to parse model output: {e}\nRaw: {generated_text[:500]}")
        raise HTTPException(500, "Failed to parse structured data from model.")


# --- FastAPI Setup ---

app = FastAPI(title="Resume Parser & Q&A API", version="2.0")


@app.get("/health", response_model=HealthStatus)
async def health_check():
    return {"status": "ok"}


@app.post("/upload", response_model=CandidateDetail)
async def upload_resume(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(400, "Invalid file type. Only PDF and DOCX supported.")

    file_bytes = await file.read()
    file_size = len(file_bytes)

    resume_text = (
        extract_text_from_pdf(file_bytes)
        if file.content_type == "application/pdf"
        else extract_text_from_docx(file_bytes)
    )

    if not resume_text:
        raise HTTPException(422, "Could not extract readable text.")

    upload_time = datetime.utcnow().isoformat()
    unique_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    storage_path = f"{datetime.now().year}/{unique_name}"

    uploaded_path = upload_file_to_supabase(file_bytes, storage_path)
    public_url = get_supabase_public_url(uploaded_path)

    metadata = {
        "file_name": file.filename,
        "file_size": file_size,
        "upload_time": upload_time,
        "storage_path": uploaded_path,
        "public_url": public_url
    }
    candidate_id = save_metadata_to_supabase(metadata)

    candidate = process_resume_with_hf(resume_text)
    candidate_data = candidate.model_dump()
    candidate_data["candidate_id"] = candidate_id

    if mongo_client:
        candidates_collection.insert_one(candidate_data)
    else:
        logger.warning("MongoDB not connected, skipping persistence.")

    return CandidateDetail(**candidate_data)


@app.get("/candidates", response_model=List[CandidateSummary])
async def list_candidates():
    if not mongo_client:
        raise HTTPException(503, "MongoDB unavailable.")
    projection = {"candidate_id": 1, "name": 1, "email": 1, "skills": 1, "_id": 0}
    results = list(candidates_collection.find({}, projection))
    return [CandidateSummary.model_validate(r) for r in results]


@app.get("/candidate/{candidate_id}", response_model=CandidateDetail)
async def get_candidate(candidate_id: str):
    if not mongo_client:
        raise HTTPException(503, "MongoDB unavailable.")
    result = candidates_collection.find_one({"candidate_id": candidate_id}, {"_id": 0})
    if not result:
        raise HTTPException(404, f"Candidate '{candidate_id}' not found.")
    return CandidateDetail.model_validate(result)


@app.post("/ask/{candidate_id}", response_model=QAResponse)
async def ask_candidate(candidate_id: str, qa_request: QARequest):
    if not mongo_client:
        raise HTTPException(503, "MongoDB unavailable.")

    candidate = candidates_collection.find_one({"candidate_id": candidate_id}, {"_id": 0})
    if not candidate:
        raise HTTPException(404, "Candidate not found.")

    context = json.dumps(candidate, indent=2)
    prompt = f"""
    You are a recruiter assistant. Answer the user's question using ONLY the following JSON context.
    If info not available, say "Information not available."

    Context:
    {context}

    Question: {qa_request.question}
    """

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256, "temperature": 0.5}
    }

    response = call_text_generation(HF_QA_MODEL, payload)
    answer = response[0].get("generated_text", "").strip()

    return QAResponse(answer=answer, question=qa_request.question, candidate_id=candidate_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
