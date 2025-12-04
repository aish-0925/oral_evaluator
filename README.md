Oral Evaluator Project (Flask)
-----------------------------
This scaffold contains a minimal Flask app and src modules for an automatic answer scoring pipeline.
Files of interest:
- flask_app.py : flask server + upload endpoint
- src/ : core modules (transcription, embedding scoring, evaluation)
- templates/index.html : simple front-end recorder & uploader
Usage:
1. Install requirements (preferably in a virtualenv)
   pip install -r requirements.txt
2. Ensure `ffmpeg` is installed on the system (for audio conversion)
3. Run the server:
   python flask_app.py
