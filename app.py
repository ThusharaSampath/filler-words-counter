import os
import re
import whisper
import numpy as np
import librosa
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = whisper.load_model("base")

FILLER_WORDS = {
    "um": ["um", "uhm", "ums"],
    "ah": ["ah", "ahh"],
    "err": ["err", "er"],
    "mmm": ["mmm", "hmm"],
    "like": ["like"],
    "you know": ["you know"],
    "I mean": ["i mean"],
    "so": ["so"],
    "basically": ["basically"],
    "actually": ["actually"],
    "literally": ["literally"],
    "kind of": ["kind of", "kinda"],
    "sort of": ["sort of", "sorta"],
    "you know what I mean": ["you know what I mean"],
    "well": ["well"],
}

def analyze_audio(audio_path):
    """Process audio file and count filler words"""
    audio, sr = librosa.load(audio_path, sr=16000)
    
    result = model.transcribe(audio_path)
    transcription = result["text"].lower()
    
    filler_counts = {}
    
    for filler_category, variants in FILLER_WORDS.items():
        count = 0
        for variant in variants:
            pattern = r'\b' + re.escape(variant) + r'\b'
            count += len(re.findall(pattern, transcription))
        
        if count > 0:
            filler_counts[filler_category] = count
    
    total_words = len(transcription.split())
    
    return {
        "filler_counts": filler_counts,
        "total_words": total_words,
        "transcription": transcription
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    allowed_extensions = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "File format not supported"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        analysis_result = analyze_audio(filepath)
        # Clean up - delete the uploaded file after processing
        os.remove(filepath)
        
        return jsonify(analysis_result)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)