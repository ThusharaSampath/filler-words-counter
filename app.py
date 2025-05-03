import os
import re
import whisper
import numpy as np
import librosa
import torch
import time
import threading
import json
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Force CPU for now due to MPS compatibility issues with sparse tensors
device = torch.device("cpu")
print("Using CPU for processing (MPS has compatibility issues with Whisper)")

# Load Whisper model (can be 'tiny', 'base', 'small', 'medium', 'large')
# Choose based on your accuracy needs and computational resources
model = whisper.load_model("base")

# Global dictionary to store processing status
processing_status = {}

# Define common filler words to detect
# This list can be expanded or customized
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

def analyze_audio(file_id, audio_path):
    """Process audio file and count filler words with progress tracking"""
    try:
        # Update status to started
        processing_status[file_id] = {"status": "processing", "progress": 0}
        
        # Load audio file
        processing_status[file_id]["progress"] = 10
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Get audio duration for estimating progress
        duration = librosa.get_duration(y=audio, sr=sr)
        
        processing_status[file_id]["progress"] = 20
        
        # Transcribe using Whisper
        processing_status[file_id]["status"] = "transcribing"
        processing_status[file_id]["progress"] = 30
        
        # For tracking progress - whisper doesn't have built-in progress callbacks
        # so we'll simulate with time estimation based on file duration
        def update_progress():
            # Estimate progress based on audio duration
            # Longer files take more time to process
            start_progress = 30
            end_progress = 80
            steps = 10
            total_time = duration * 0.5  # Rough estimate: processing takes ~0.5x of audio duration
            time_per_step = total_time / steps
            
            for i in range(steps):
                time.sleep(time_per_step)
                if file_id in processing_status:
                    current_progress = start_progress + (end_progress - start_progress) * (i + 1) / steps
                    processing_status[file_id]["progress"] = int(current_progress)
        
        # Start progress tracking in background
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Actual transcription
        result = model.transcribe(audio_path)
        transcription = result["text"].lower()
        
        # Signal that transcription is complete
        processing_status[file_id]["status"] = "analyzing"
        processing_status[file_id]["progress"] = 85
        
        # Count filler words
        filler_counts = {}
        
        for filler_category, variants in FILLER_WORDS.items():
            count = 0
            for variant in variants:
                # Use word boundary to avoid counting partial matches
                pattern = r'\b' + re.escape(variant) + r'\b'
                count += len(re.findall(pattern, transcription))
            
            if count > 0:
                filler_counts[filler_category] = count
        
        # Calculate total words for context
        total_words = len(transcription.split())
        
        processing_status[file_id]["progress"] = 95
        
        result = {
            "filler_counts": filler_counts,
            "total_words": total_words,
            "transcription": transcription
        }
        
        # Complete
        processing_status[file_id]["status"] = "complete"
        processing_status[file_id]["progress"] = 100
        processing_status[file_id]["result"] = result
        
        return result
    except Exception as e:
        processing_status[file_id]["status"] = "error"
        processing_status[file_id]["error"] = str(e)
        raise e

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
    
    # Check if the file is an audio file
    allowed_extensions = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "File format not supported"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Generate a unique ID for this analysis job
    file_id = str(int(time.time() * 1000))  # timestamp as ID
    
    # Start processing in a background thread
    def process_in_background():
        try:
            analyze_audio(file_id, filepath)
            # Clean up - delete the uploaded file after processing
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            # Make sure we clean up even if there's an error
            if os.path.exists(filepath):
                os.remove(filepath)
    
    processing_thread = threading.Thread(target=process_in_background)
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({"file_id": file_id})

@app.route('/progress/<file_id>', methods=['GET'])
def get_progress(file_id):
    """Get the current progress of analysis"""
    if file_id not in processing_status:
        return jsonify({"error": "Invalid file ID or processing completed"}), 404
    
    status_info = processing_status[file_id].copy()
    
    # If processing is complete, return the result and remove from status tracking
    if status_info["status"] == "complete":
        result = status_info.pop("result", {})
        # Remove from tracking after client gets the final result
        if request.args.get('final') == 'true':
            processing_status.pop(file_id, None)
        return jsonify({"status": "complete", "progress": 100, "result": result})
    
    # If error occurred
    if status_info["status"] == "error":
        error_msg = status_info.get("error", "Unknown error")
        processing_status.pop(file_id, None)  # Clean up
        return jsonify({"status": "error", "error": error_msg}), 500
    
    # Otherwise return the current progress
    return jsonify({
        "status": status_info["status"],
        "progress": status_info["progress"]
    })

if __name__ == '__main__':
    app.run(debug=True)