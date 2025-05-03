## What is this ?
This is a simple python based web application that can be used to count filler words of a speech. 

## How to run ?
1. Environment Setup (Assume you have Python, Pip and brew)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install FFmpeg
   ```bash
   brew install ffmpeg
   ```
3. Install Requirements
   ```bash
   pip install -r requirements.txt
   ```
4. Run
   ```bash
   python app.py
   ```

## How to use ? 
1. Run the application.
2. Navigate to the server via a browser. 
3. Upload a audio file (<32MB).
4. Click **Analyze**. 
5. After processing, you will have result similar to the following image.

![image](/resources/screenshot.png)

