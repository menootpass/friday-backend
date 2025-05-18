from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
import json
import logging
import httpx
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Enable CORS with specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Get API key from environment variable
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise ValueError("NVIDIA_API_KEY environment variable is not set. Please set it in your .env file or environment variables.")

# Configure client with longer timeout
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    category: str
    model: str

class SolidStatePhysicsQuiz:
    def __init__(self):
        self.client = client

    def generate_question(self) -> Dict[str, Any]:
        prompt = """Buat soal fisika zat padat level menengah untuk mahasiswa dalam format JSON:
        {
            "question": "Your question in Indonesian",
            "options": ["opsi a", "opsi b", "opsi c", "opsi d"],
            "correctAnswer": 0,
            "explanation": "Detailed explanation in Indonesian"
        }
        Topik: getaran kisi."""

        try:
            logger.info("Sending request to NVIDIA API...")
            completion = self.client.chat.completions.create(
                model="deepseek-ai/deepseek-r1",
                messages=[
                    {"role": "system", "content": "You are a physics teacher creating quiz questions. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Get the response text
            response_text = completion.choices[0].message.content
            
            # Clean the response text
            # Remove markdown code blocks and any extra whitespace
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Find the first { and last }
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            
            logger.info("Cleaned response text:")
            logger.info(response_text)
            
            try:
                # Try to parse the response as JSON
                question_data = json.loads(response_text)
                
                # Validate the required fields
                required_fields = ["question", "options", "correctAnswer", "explanation"]
                for field in required_fields:
                    if field not in question_data:
                        raise ValueError(f"Missing required field: {field}")
                
                if not isinstance(question_data["options"], list) or len(question_data["options"]) != 4:
                    raise ValueError("Options must be a list with exactly 4 items")
                
                if not isinstance(question_data["correctAnswer"], int) or question_data["correctAnswer"] not in [0, 1, 2, 3]:
                    raise ValueError("correctAnswer must be an integer between 0 and 3")
                
                # Ensure all fields are strings
                question_data["question"] = str(question_data["question"])
                question_data["options"] = [str(opt) for opt in question_data["options"]]
                question_data["explanation"] = str(question_data["explanation"])
                
                logger.info("Parsed question data:")
                logger.info(json.dumps(question_data, indent=2))
                return question_data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.error(f"Raw response text: {response_text}")
                raise
            except ValueError as e:
                logger.error(f"Invalid question format: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            # Fallback to a default question if API fails
            fallback_question = {
                "question": "Apa yang dimaksud dengan struktur kristal BCC?",
                "options": [
                    "Struktur dengan atom di setiap sudut kubus",
                    "Struktur dengan atom di setiap sudut dan pusat kubus",
                    "Struktur dengan atom di setiap sudut dan pusat setiap permukaan",
                    "Struktur dengan atom tersusun secara heksagonal"
                ],
                "correctAnswer": 1,
                "explanation": "Struktur BCC (Body Centered Cubic) adalah struktur kristal yang memiliki atom di setiap sudut kubus dan satu atom di pusat kubus. Struktur ini memiliki faktor pengepakan atom sebesar 0.68."
            }
            logger.info("Using fallback question:")
            logger.info(json.dumps(fallback_question, indent=2))
            return fallback_question

# Initialize quiz generator
quiz_generator = SolidStatePhysicsQuiz()

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Get the last message from the messages array
        last_message = request.messages[-1]["content"]
        
        # Create chat completion with streaming
        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-r1",
            messages=[{"role": "user", "content": last_message}],
            temperature=0.6,
            top_p=0.7,
            max_tokens=4096,
            stream=True
        )
        
        # Collect the full response
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        
        # Return response in the format expected by the frontend
        return {
            "response": full_response,
            "model": request.model
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/quiz/question")
async def get_question():
    try:
        logger.info("Generating new question...")
        question = quiz_generator.generate_question()
        logger.info("Question generated successfully")
        
        # Ensure the response is properly formatted
        response = {
            "question": question["question"],
            "options": question["options"],
            "correctAnswer": question["correctAnswer"],
            "explanation": question["explanation"]
        }
        
        # Log the final response being sent to frontend
        logger.info("Sending response to frontend:")
        logger.info(json.dumps(response, indent=2))
        
        # Add CORS headers explicitly
        return JSONResponse(
            content=response,
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
    except Exception as e:
        logger.error(f"Error in get_question endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)