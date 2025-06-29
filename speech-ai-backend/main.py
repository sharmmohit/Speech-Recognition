from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import tempfile
import openai
import os

app = FastAPI()

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

@app.post("/api/speech")
async def process_audio(audio: UploadFile = File(...)):
    # Save the uploaded audio to a temporary .wav file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await audio.read())
        temp_audio_path = temp_audio.name

    try:
        # Use Whisper to transcribe the audio
        with open(temp_audio_path, "rb") as f:
            transcription = openai.Audio.transcribe("whisper-1", f)

        prompt = transcription["text"]
        print("Transcribed:", prompt)

        # Send transcription to GPT for a response
        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful voice assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        reply = chat_response["choices"][0]["message"]["content"]
        return {"response": reply}

    except Exception as e:
        print("Error:", str(e))
        return {"response": "An error occurred while processing your request."}
