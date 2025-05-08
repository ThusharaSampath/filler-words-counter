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
from flask_sock import Sock # <-- Import Sock

app = Flask(__name__)
sock = Sock(app) # <-- Initialize Sock
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Force CPU for now due to MPS compatibility issues with sparse tensors
device = torch.device("cpu")
print("Using CPU for processing (MPS has compatibility issues with Whisper)")

# Load Whisper model (can be 'tiny', 'base', 'small', 'medium', 'large')
# Choose based on your accuracy needs and computational resources
# For real-time, 'tiny' or 'base' is recommended.
model_name = "base" # Consider "tiny.en" or "base.en" for faster English-only
model = whisper.load_model(model_name, device=device)
print(f"Whisper model '{model_name}' loaded on {device}")


# Global dictionary to store processing status for file uploads
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

def highlight_filler_words_text(transcription_text):
    """
    Identifies filler words in a given text and returns a list of
    word objects with a flag if they are fillers.
    This will be used by the real-time endpoint to send structured data.
    """
    words = transcription_text.split()
    highlighted_segments = []
    # A more robust way would be to iterate and build segments
    # For now, let's re-use the regex approach for identifying,
    # then map back to words for highlighting (can be complex).

    # Simpler approach for real-time: send raw text and let frontend highlight.
    # Or, send text with inline markers (less clean for JS to parse than structured data).
    # For now, we'll just focus on getting the text out.
    # Highlighting logic can be enhanced on the frontend or here later.

    # This function can be enhanced to return a structure like:
    # [{"word": "actually", "is_filler": True}, {"word": "this", "is_filler": False}]
    # For now, it will just return the text, and the frontend will do the regex highlighting.
    return transcription_text


# --- Existing File Upload Code (analyze_audio, /analyze, /progress) remains largely the same ---
# ... (Keep your existing analyze_audio function and routes)
def analyze_audio(file_id, audio_path):
    """Process audio file and count filler words with progress tracking"""
    try:
        # Update status to started
        processing_status[file_id] = {"status": "processing", "progress": 0}

        # Load audio file
        processing_status[file_id]["progress"] = 10
        audio, sr = librosa.load(audio_path, sr=16000) # Ensure 16kHz for Whisper

        # Get audio duration for estimating progress
        duration = librosa.get_duration(y=audio, sr=sr)

        processing_status[file_id]["progress"] = 20

        # Transcribe using Whisper
        processing_status[file_id]["status"] = "transcribing"
        processing_status[file_id]["progress"] = 30

        # For tracking progress - whisper doesn't have built-in progress callbacks
        # so we'll simulate with time estimation based on file duration
        def update_progress_simulation():
            # Estimate progress based on audio duration
            # Longer files take more time to process
            start_progress = 30
            end_progress = 80 # Leave some room for final analysis
            steps = 10 # Number of updates during transcription simulation
            # Rough estimate: processing takes ~0.5x of audio duration for 'base' model
            # This is highly dependent on CPU, model size. Adjust as needed.
            total_transcription_time_estimate = duration * 0.3
            if total_transcription_time_estimate < 1: total_transcription_time_estimate = 1 # min 1 sec
            time_per_step = total_transcription_time_estimate / steps

            for i in range(steps):
                time.sleep(time_per_step)
                if file_id in processing_status and processing_status[file_id]["status"] == "transcribing":
                    current_progress = start_progress + (end_progress - start_progress) * (i + 1) / steps
                    processing_status[file_id]["progress"] = int(current_progress)
                elif file_id not in processing_status or processing_status[file_id]["status"] != "transcribing":
                    break # Stop if status changed or job removed

        # Start progress tracking in background
        progress_thread = threading.Thread(target=update_progress_simulation)
        progress_thread.daemon = True
        progress_thread.start()

        # Actual transcription
        # For potentially long files, consider transcribing in segments or using more advanced options.
        result = model.transcribe(audio, language="en") # Pass the numpy array directly
        transcription = result["text"].lower()
        progress_thread.join() # Ensure simulated progress finishes if transcription was very fast

        # Signal that transcription is complete for accurate progress update
        if file_id in processing_status:
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

        if file_id in processing_status:
            processing_status[file_id]["progress"] = 95

        analysis_result = {
            "filler_counts": filler_counts,
            "total_words": total_words,
            "transcription": transcription
        }

        # Complete
        if file_id in processing_status:
            processing_status[file_id]["status"] = "complete"
            processing_status[file_id]["progress"] = 100
            processing_status[file_id]["result"] = analysis_result

        return analysis_result
    except Exception as e:
        if file_id in processing_status:
            processing_status[file_id]["status"] = "error"
            processing_status[file_id]["error"] = str(e)
        # Log the exception for server-side debugging
        app.logger.error(f"Error in analyze_audio for file_id {file_id}: {str(e)}", exc_info=True)
        # Do not re-raise here if status is managed, or client will get an alert
        # from the initial POST if it was a direct call, but this runs in a thread.
        return None # Or some error indicator

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
    # To avoid filename collisions if multiple users upload files with the same name,
    # and to make file_id more robust:
    file_id = str(int(time.time() * 1000)) + "_" + filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id) # Use file_id as part of the filename
    file.save(filepath)

    # Initialize status for this file_id immediately
    processing_status[file_id] = {"status": "queued", "progress": 0}

    # Start processing in a background thread
    def process_in_background():
        try:
            analyze_audio(file_id, filepath)
        except Exception as e:
            app.logger.error(f"Error in background processing for {file_id}: {e}", exc_info=True)
            if file_id in processing_status:
                processing_status[file_id]["status"] = "error"
                processing_status[file_id]["error"] = "Processing failed in background"
        finally:
            # Clean up - delete the uploaded file after processing
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e_remove:
                    app.logger.error(f"Error deleting file {filepath}: {e_remove}")


    processing_thread = threading.Thread(target=process_in_background)
    processing_thread.daemon = True
    processing_thread.start()

    return jsonify({"file_id": file_id})


@app.route('/progress/<file_id>', methods=['GET'])
def get_progress(file_id):
    """Get the current progress of analysis"""
    if file_id not in processing_status:
        return jsonify({"error": "Invalid file ID or processing completed and cleaned"}), 404

    status_info = processing_status[file_id].copy() # Work with a copy

    # If processing is complete, return the result
    if status_info.get("status") == "complete":
        # Prepare data to be sent, ensuring result is present
        response_data = {
            "status": "complete",
            "progress": status_info.get("progress", 100),
            "result": status_info.get("result", {}) # Send empty dict if no result for some reason
        }
        # Remove from tracking only if client signals it's the final request
        if request.args.get('final') == 'true':
            processing_status.pop(file_id, None)
        return jsonify(response_data)

    # If error occurred
    if status_info.get("status") == "error":
        error_msg = status_info.get("error", "Unknown error")
        processing_status.pop(file_id, None)  # Clean up
        return jsonify({"status": "error", "progress": status_info.get("progress", 0), "error": error_msg}), 500

    # Otherwise return the current progress
    return jsonify({
        "status": status_info.get("status", "unknown"),
        "progress": status_info.get("progress", 0)
    })


# --- New Real-time WebSocket Endpoint ---
@sock.route('/realtime_transcribe')
def realtime_transcribe(ws): # ws is the WebSocket connection object
    app.logger.info("WebSocket connection established.")
    buffer_duration_seconds = 3 # Process audio in X-second chunks
    sample_rate = 16000
    audio_buffer = bytearray()

    try:
        while True:
            # Receive audio data from the client
            # Expecting raw PCM data (e.g., Float32Array or Int16Array bytes)
            message = ws.receive(timeout=1) # Timeout to allow checking ws.connected
            if message is None: # Connection closed by client or timeout with no data
                if not ws.connected:
                    app.logger.info("WebSocket client disconnected gracefully.")
                    break
                continue # Timeout, but still connected, wait for more data

            audio_buffer.extend(message)

            # Calculate current duration in buffer
            # Assuming frontend sends 16-bit PCM mono (2 bytes per sample)
            # If frontend sends Float32, it's 4 bytes per sample.
            # JS side will send Float32Array.buffer, so 4 bytes per sample.
            bytes_per_sample = 4 # For Float32
            current_duration = len(audio_buffer) / (sample_rate * bytes_per_sample)

            if current_duration >= buffer_duration_seconds:
                # Convert buffer to NumPy array (Float32)
                # Ensure the buffer length is a multiple of bytes_per_sample
                extra_bytes = len(audio_buffer) % bytes_per_sample
                process_bytes = len(audio_buffer) - extra_bytes
                
                if process_bytes == 0:
                    continue # Not enough data to form a complete sample

                audio_np = np.frombuffer(audio_buffer[:process_bytes], dtype=np.float32)
                
                # Keep the remaining bytes for the next round
                audio_buffer = audio_buffer[process_bytes:]


                if audio_np.size == 0:
                    continue

                app.logger.debug(f"Processing audio chunk of duration: {audio_np.size / sample_rate:.2f}s")

                # Transcribe the audio chunk
                # Use a simpler/faster model or settings for real-time if needed
                # Adding `condition_on_previous_text=True` can help with context but needs managing previous text.
                # For simplicity now, we treat each chunk independently.
                result = model.transcribe(audio_np, language="en", fp16=False) # fp16=False if using CPU
                transcription_text = result["text"].strip().lower()
                app.logger.debug(f"Raw transcription: '{transcription_text}'")

                if transcription_text:
                    # Send back the transcription
                    # Frontend will handle highlighting for now based on its FILLER_WORDS
                    ws.send(json.dumps({"type": "partial_transcript", "text": transcription_text}))
                    app.logger.info(f"Sent partial transcript: {transcription_text}")
                else:
                    ws.send(json.dumps({"type": "status", "text": "No speech detected in chunk."}))


    except Exception as e:
        app.logger.error(f"WebSocket Error: {e}", exc_info=True)
        try:
            ws.send(json.dumps({"type": "error", "message": str(e)}))
        except: # If sending fails, ws might be closed
            pass
    finally:
        app.logger.info("WebSocket connection closed.")
        # ws.close() # simple-websocket closes automatically on exit from handler

if __name__ == '__main__':
    # For development, Flask's built-in server is fine.
    # For production, use a proper WSGI server like gunicorn with gevent for WebSockets.
    # Example: gunicorn --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 app:app
    app.run(debug=True, host='0.0.0.0', port=5001) # Ensure port is accessible