# FastAPI Resume Parser and Q&A Service

This is a complete end-to-end FastAPI application designed to process resume files, extract structured candidate data using a state-of-the-art Hugging Face LLM, and persist both the file metadata and the extracted data into Supabase and MongoDB respectively. It also provides a natural-language Q&A interface for the stored candidates.

## System Overview

The application follows a clean, modular architecture based on the following flow:

1. **File Upload (`/upload`)**: A user uploads a `.pdf` or `.docx` file.
2. **Supabase**: The raw file bytes and associated metadata (name, size, time, path) are saved to Supabase Storage and a PostgREST table, generating a unique `candidate_id`.
3. **Text Extraction**: Text content is extracted from the file using `pypdf` or `python-docx`.
4. **Hugging Face (Extraction)**: The raw text is passed to an instruction-tuned LLM (`Mixtral-8x7B`) via the Inference API to generate a structured JSON object.
5. **MongoDB**: The structured JSON data is stored in the `candidates` collection, linked by the `candidate_id`.
6. **Q&A (`/ask`)**: A user asks a question, the MongoDB data is retrieved, and another LLM (`Zephyr-7B`) uses the data as context to generate an answer.

## Setup and Installation

### 1. Prerequisites

- Python 3.8+
- **Supabase Project**: Database table named `resumes` and a Storage bucket named `resumes` (or modify `SUPABASE_BUCKET`).
- **MongoDB Instance**: A connection URI for a local or cloud instance.
- **Hugging Face Account**: An API Key for the Inference API.

### 2. Environment Variables

Create a file named `.env` in the root directory of the project and populate it with your credentials:

```env
# Supabase Configuration
SUPABASE_URL="https://your-project-ref.supabase.co"
SUPABASE_KEY="your-anon-or-service-role-key"

# Hugging Face Configuration
HF_API_KEY="hf_your_api_key_here"

# MongoDB Configuration
MONGO_URI="mongodb://localhost:27017/"
MONGO_DB_NAME="resume_db"
MONGO_COLLECTION_NAME="candidates"

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run the Application
uvicorn main:app --reload

The application will be accessible at http://127.0.0.1:8000

# Database and Model Notes
| Component             | Integration      | Model/Service                          | Purpose                                                       |
| --------------------- | ---------------- | -------------------------------------- | ------------------------------------------------------------- |
| File Storage          | Direct API Calls | Supabase Storage                       | Store raw `.pdf` / `.docx` files                              |
| Metadata DB           | Direct API Calls | Supabase PostgREST                     | Store file paths, upload time, generate `candidate_id`        |
| Structured Data       | pymongo          | MongoDB                                | Store the extracted candidate JSON                            |
| Structured Extraction | Hugging Face API | `mistralai/Mixtral-8x7B-Instruct-v0.1` | Converts raw text into structured JSON                        |
| Q&A Generation        | Hugging Face API | `HuggingFaceH4/zephyr-7b-beta`         | Answers natural-language questions based on candidate context |

# API Endpoints Reference
1. Health Check

Checks the server status.

Endpoint: GET /health

Response:
{
  "status": "ok"
}
 
2. File Upload and Processing

Uploads a resume and initiates the full processing pipeline.

Endpoint: POST /upload

Request Body: Form Data file (a .pdf or .docx file)

Sample Response:
{
  "candidate_id": "714f2a34-4af7-465b-b9c7-e28dd731f0e9",
  "name": "Soumya Subhra Datta",
  "email": "soumyasubhradatta@gmail.com",
  "phone": "+91 8610928762",
  "education": {
    "degree": "Bachelor of Engineering (B.E)",
    "field": "Computer Science and Engineering (Artificial Intelligence)",
    "institution": "Sathyabama Institute of Science and Technology, Chennai, India",
    "duration": "2023 – 2027 (Currently in 3rd Year)",
    "cgpa": "7.86 / 10"
  },
  "experience": {
    "AI Medical Chatbot Backend": {
      "description": "Built a healthcare chatbot using LangChain + Gemini API + RAG pipelines.",
      "features": "Symptom analysis, multilingual disease prediction, memory-based conversations.",
      "skills": [
        "Python",
        "LangChain",
        "Google Generative AI",
        "Prompt Engineering",
        "RAG",
        "Error Handling",
        "Logging",
        "Regex",
        "Dataclasses"
      ]
    },
    "Proactive Fraud Detection (6.36M Rows)": {
      "description": "Developed a fraud prediction pipeline with missing data handling, outlier removal, and class imbalance correction.",
      "skills": [
        "Python",
        "Scikit-learn",
        "Pandas",
        "NumPy",
        "Feature Engineering",
        "Anomaly Detection",
        "Model Evaluation",
        "Cost-sensitive Learning"
      ]
    },
    "Career Advisor (Full-Stack Platform)": {
      "description": "Created a career counseling web app with Flask backend + HTML/CSS/JS frontend.",
      "features": "User authentication (login/signup), aptitude quiz, college search with filters, timeline management, profile dashboard.",
      "skills": [
        "Flask",
        "HTML",
        "CSS",
        "JavaScript",
        "REST APIs",
        "JSON",
        "MongoDB Atlas",
        "GitHub",
        "Responsive UI"
      ]
    },
    "AI Learning Assistant (File-Based Q&A System)": {
      "description": "Built an AI assistant that processes PDFs, Word, and CSV files for instant question answering.",
      "skills": [
        "Python",
        "LangChain",
        "FAISS",
        "ChromaDB",
        "Prompt Engineering",
        "Document Parsing",
        "RAG Pipelines"
      ]
    }
  },
  "skills": [
    "Python",
    "Java",
    "JavaScript",
    "Scikit-learn",
    "RandomForest",
    "SGDClassifier",
    "Naive Bayes",
    "Anomaly Detection",
    "Cost-Sensitive Learning",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Feature Engineering",
    "Data Cleaning",
    "Outlier Handling",
    "Model Evaluation",
    "Business Analytics",
    "Data Analytics",
    "LangChain",
    "OpenAI API",
    "Gemini API",
    "RAG Pipelines",
    "Prompt Engineering",
    "Memory Modules",
    "FAISS",
    "ChromaDB",
    "OpenCV",
    "CNN Architectures",
    "Image Preprocessing",
    "REST APIs",
    "Flask",
    "FastAPI",
    "Google Colab",
    "Jupyter",
    "Hugging Face",
    "MongoDB Atlas",
    "NoSQL",
    "AWS",
    "GitHub",
    "SQL",
    "dotenv",
    "Logging",
    "Regex",
    "Dataclasses",
    "Excel",
    "Google Sheets",
    "Tableau"
  ],
  "hobbies": [
    "Exploring emerging AI tools, frameworks, and creative technologies"
  ],
  "certifications": [],
  "projects": [],
  "introduction": "Results-driven AI Intern with hands-on experience in building Generative AI, Data Science, and ML-powered solutions."
}
Response headers

3. List All Candidates

Retrieves a summary of all candidates stored in MongoDB.

Endpoint: GET /candidates

Sample Response:
[
  {
    "candidate_id": "8485078c-0731-48fa-8316-3e75529f3456",
    "name": "Alex Johnson",
    "email": "alex.johnson@example.com",
    "skills": ["Python", "FastAPI", "MongoDB", "Docker"]
  },
  {
    "candidate_id": "d1e3450a-99b8-4c2d-90e1-0f2c4a7b8e9c",
    "name": "Maria Garcia",
    "email": "maria.garcia@web.dev",
    "skills": ["TypeScript", "React", "Node.js", "Tailwind CSS"]
  }
]

4. Get Single Candidate Details

Retrieves the complete structured data for a specific candidate.

Endpoint: GET /candidate/{candidate_id}

Sample Request:
GET /candidate/8485078c-0731-48fa-8316-3e75529f3456

Sample Response: Same full JSON structure as the /upload response.

5. Candidate Q&A

Asks a natural-language question about a specific candidate's data.

Endpoint: POST /ask/{candidate_id}

Request Body:
{
  "question": "What company did Alex Johnson work for between 2018 and 2023?"
}


---

This version is **clean, Markdown-compliant, readable, and free of emojis or special symbols**, making it perfect for GitHub.  

If you want, I can also **add a nice Table of Contents and badges** so it looks even more professional on GitHub.  

Do you want me to do that?
