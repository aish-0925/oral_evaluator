import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "csv")
DATA_CSV = os.path.join(DATA_DIR, "os_questions.csv")
REFERENCE_CSV = os.path.join(DATA_DIR, "reference_answers.csv")
HUMAN_LABEL_CSV = os.path.join(DATA_DIR, "human_labeling.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "outputs", "similarity_results.csv")
COLUMN_AUDIO = "audio_path"
COLUMN_HUMAN_SCORE = "human_score"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

