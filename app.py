from dotenv import load_dotenv
import os
import requests
load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
TTS_API_URL = os.environ.get("TTS_API_URL", "http://localhost:8000") # Fallback for local testing
TTS_API_KEY = os.environ.get("TTS_API_KEY", "manager-tts-render-2024-xyz789secure")
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
        start_idx=content.find('{')
        end_idx=content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            content=content[start_idx:end_idx+1]

        # Attempt JSON parsing
        return json.loads(content)

    except json.JSONDecodeError:
        raise ValueError("LLM failed to return valid JSON.")
    except Exception as e:
        raise ValueError(f"LLM generation failed: {str(e)}")
# -----------------------------
# PODCAST GENERATION LOGIC
# -----------------------------
def generate_podcast_script(text: str, language: str, duration_minutes: int) -> str:
    # Core Business Logic: Standard speaking rate is 150 words per minute
    target_word_count = duration_minutes * 150
    
    prompt = f"""
    You are an expert podcast host. Your job is to summarize the provided educational text into an engaging, continuous conversational podcast script.
    
    CRITICAL CONSTRAINTS:
    1. Language: The script MUST be written entirely in {language}. If the input is English and the requested language is Hindi, translate it perfectly into Hindi.
    2. Length: The script MUST be approximately {target_word_count} words long. This is strictly required to ensure the final audio is exactly {duration_minutes} minutes long.
    3. Format: Write ONLY the spoken words. DO NOT include speaker labels (like "Host:"), sound effect cues (like "[Music fades in]"), or markdown. 
    
    Text to adapt:
    {text[:4000]}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, # Higher temperature for more natural, conversational phrasing
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise ValueError(f"LLM script generation failed: {str(e)}")
@app.post("/api/v1/generate-podcast")
async def create_podcast(
    file: UploadFile = File(...),
    language: str = Form(...),
    duration: int = Form(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    if language.lower() not in ["english", "hindi"]:
        raise HTTPException(status_code=400, detail="Language must be English or Hindi.")
    
    # QA Check: Protect against excessive cloud usage
    if not (2 <= duration <= 10):
        raise HTTPException(status_code=400, detail="Podcast duration must be between 2 and 10 minutes.")

    try:
        # 1. Parse Document
        file_bytes = await file.read()
        extracted_text = extract_text_from_pdf(file_bytes)
        
        # 2. Call LLM for Script
        script = generate_podcast_script(extracted_text, language, duration)
        
        # 3. Orchestrate: Send Script to External TTS API
        headers = {"x-api-key": TTS_API_KEY}
        payload = {
            "text": script,
            "target_language": language
        }
        
        # Execute cross-service HTTP POST request
        tts_response = requests.post(f"{TTS_API_URL}/tts/text", json=payload, headers=headers)
        
        if tts_response.status_code != 200:
            raise ValueError(f"TTS Microservice failed to generate audio. Status: {tts_response.status_code}")
            
        tts_data = tts_response.json()
        
        # 4. Construct Final URL and return
        audio_url = f"{TTS_API_URL}{tts_data['file_path']}"
        
        return {
            "status": "success",
            "script": script,
            "audio_url": audio_url
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
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
