import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import json
from openai import OpenAI

app = FastAPI(title="PDF to Quiz API", version="1.0.0")

# Allow frontend access (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize client to use Groq's free API with your key
# REMINDER: Delete this key in your Groq dashboard after your presentation!
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()

        if not text.strip():
            raise ValueError("No readable text found in PDF.")

        return text

    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")


# -----------------------------
# QUIZ GENERATION
# -----------------------------
def generate_quiz_from_text(text: str, language: str, num_questions: int) -> dict:
    prompt = f"""
You are an expert bilingual educational content creator.

Create a multiple-choice quiz from the following text.

CRITICAL INSTRUCTION: The user has requested the quiz in: {language}.
All generated content (the quiz title, questions, options, and explanations) MUST be written entirely in {language}. If the input text is in English and the requested language is Hindi, you MUST translate the output into Hindi.

Rules:
1. Generate exactly {num_questions} questions.
2. Provide 4 options per question (A, B, C, D).
3. Return ONLY valid JSON.
4. No markdown formatting.
5. The JSON keys MUST remain in English so the system can parse them, but the string values MUST be in {language}.

Schema:
{{
  "quiz_title": "String (Write in {language})",
  "questions": [
    {{
      "question_text": "String (Write in {language})",
      "options": {{"A": "String", "B": "String", "C": "String", "D": "String"}},
      "correct_answer": "A|B|C|D",
      "explanation": "String (Write in {language})"
    }}
  ]
}}

Text to analyze:
{text[:4000]}
"""

    try:
        # Swapped to Groq's active Llama 3.1 model
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()

        # Attempt JSON parsing
        return json.loads(content)

    except json.JSONDecodeError:
        raise ValueError("LLM failed to return valid JSON.")
    except Exception as e:
        raise ValueError(f"LLM generation failed: {str(e)}")


# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/api/v1/generate-quiz")
async def create_quiz(
    file: UploadFile = File(...),
    language: str = Form(...),
    num_questions: int = Form(...)
):
    # Validate file
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if language.lower() not in ["english", "hindi"]:
        raise HTTPException(status_code=400, detail="Language must be English or Hindi.")
        
    if not (1 <= num_questions <= 20):
        raise HTTPException(status_code=400, detail="Please request between 1 and 20 questions.")

    try:
        # Read file
        file_bytes = await file.read()

        # Extract text
        extracted_text = extract_text_from_pdf(file_bytes)

        # Generate quiz dynamically
        quiz_data = generate_quiz_from_text(extracted_text, language, num_questions)

        return {
            "status": "success",
            "data": quiz_data
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return f.read()