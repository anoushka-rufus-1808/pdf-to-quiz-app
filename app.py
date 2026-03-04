from dotenv import load_dotenv
import os
import requests
import time
load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import fitz  # PyMuPDF
import json
from openai import OpenAI

app = FastAPI(title="PDF to Quiz API", version="1.0.0")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize client
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
TTS_API_URL = os.environ.get("TTS_API_URL", "http://localhost:8000")
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
Create a multiple-choice quiz from the following text in {language}.
All content MUST be in {language}. Return ONLY valid JSON.
Schema:
{{
  "quiz_title": "String",
  "questions": [
    {{
      "question_text": "String",
      "options": {{"A": "String", "B": "String", "C": "String", "D": "String"}},
      "correct_answer": "A|B|C|D",
      "explanation": "String"
    }}
  ]
}}
Text: {text[:4000]}
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
        return json.loads(content)
    except Exception as e:
        raise ValueError(f"LLM generation failed: {str(e)}")

# -----------------------------
# PODCAST GENERATION LOGIC
# -----------------------------
def generate_podcast_script(text: str, language: str, duration_minutes: int) -> str:
    target_word_count = duration_minutes * 150
    prompt = f"""
    You are an expert podcast host. Summarize this text into a conversational script in {language}.
    Target length: {target_word_count} words. No speaker labels or markdown.
    Text: {text[:4000]}
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise ValueError(f"LLM script generation failed: {str(e)}")

# -----------------------------
# MAIN PODCAST ENDPOINT (WITH RETRY LOGIC)
# -----------------------------
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
    
    if not (2 <= duration <= 10):
        raise HTTPException(status_code=400, detail="Duration must be 2-10 mins.")

    try:
        # 1. Parse Document
        file_bytes = await file.read()
        extracted_text = extract_text_from_pdf(file_bytes)
        
        # 2. Generate Script
        script = generate_podcast_script(extracted_text, language, duration)
        
        # 3. Orchestrate with Exponential Backoff
        headers = {"x-api-key": TTS_API_KEY}
        payload = {"text": script, "target_language": language}
        
        tts_data = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Use a timeout so we don't hang forever
                tts_response = requests.post(
                    f"{TTS_API_URL}/tts/text", 
                    json=payload, 
                    headers=headers, 
                    timeout=90 
                )
                
                if tts_response.status_code == 200:
                    tts_data = tts_response.json()
                    break # Success!
                
                # If we get rate limited (429), wait and retry
                if tts_response.status_code == 429:
                    wait_time = (attempt + 1) * 4 # 4s, 8s, 12s
                    print(f"Rate limited. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise ValueError(f"TTS Service returned {tts_response.status_code}")

            except requests.exceptions.RequestException:
                if attempt == max_retries - 1:
                    raise ValueError("Could not connect to TTS service after multiple attempts.")
                time.sleep(3)

        if not tts_data:
            raise ValueError("TTS Service is currently unavailable. Please try again in a few minutes.")
            
        # 4. Construct Final URL safely
        clean_base_url = TTS_API_URL.rstrip('/')
        audio_url = f"{clean_base_url}{tts_data['file_path']}"
        
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
# QUIZ ENDPOINT
# -----------------------------
@app.post("/api/v1/generate-quiz")
async def create_quiz(
    file: UploadFile = File(...),
    language: str = Form(...),
    num_questions: int = Form(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        file_bytes = await file.read()
        extracted_text = extract_text_from_pdf(file_bytes)
        quiz_data = generate_quiz_from_text(extracted_text, language, num_questions)
        return {"status": "success", "data": quiz_data}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return f.read()
