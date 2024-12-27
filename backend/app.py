from pathlib import Path
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

@app.route('/api/audio', methods=['POST'])
def generate_audio():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "Text is required for audio generation."}), 400

    try:
        # Call OpenAI API for audio generation
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": f"Please provide the pronunciation for: {text}"
                }
            ]
        )

        # Decode the audio data
        wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)

        # Save the audio file temporarily
        audio_path = "output.wav"
        with open(audio_path, "wb") as f:
            f.write(wav_bytes)

        return jsonify({"audio_path": audio_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '')
    target_language = data.get('language', 'en')

    if not text:
        return jsonify({"error": "Text is required for translation."}), 400

    try:
        # Call OpenAI API for translation
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": f"Translate the following text to {target_language}: {text}"}
            ]
        )
        translated_text = response['choices'][0]['message']['content']
        return jsonify({"translated_text": translated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
