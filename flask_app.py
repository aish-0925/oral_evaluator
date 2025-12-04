from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import uuid
import datetime
import subprocess
from werkzeug.utils import secure_filename

from src.api import evaluate_audio, load_questions
from src.utils import ensure_dir, append_evaluation_log

# ----------------------
# Configuration
# ----------------------
BASE_DIR = os.getcwd()
BASE_OUTPUTS = os.path.join(BASE_DIR, "outputs")
UPLOAD_DIR = os.path.join(BASE_OUTPUTS, "audio_records")
ensure_dir(UPLOAD_DIR)
ensure_dir(BASE_OUTPUTS)

# optional safety: limit upload size (e.g. 10 MB)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"webm", "wav", "mp3", "m4a", "ogg"}

# ----------------------
# App initialization
# ----------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# ----------------------
# Helpers
# ----------------------

def allowed_file(filename: str) -> bool:
    if not filename:
        return False
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def _next_available_wav_path(qid: str, out_dir: str) -> str:
    base = f"{qid}.wav"
    p = os.path.join(out_dir, base)
    if not os.path.exists(p):
        return p
    i = 1
    while True:
        p2 = os.path.join(out_dir, f"{qid}_{i}.wav")
        if not os.path.exists(p2):
            return p2
        i += 1

# ----------------------
# Routes
# ----------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/questions")
def api_questions():
    try:
        qs = load_questions()
        return jsonify(qs)
    except Exception as e:
        app.logger.exception("Failed to load questions")
        return jsonify({"error": "failed_to_load_questions", "details": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    try:
        if "file" not in request.files:
            return jsonify({"error": "no_file_uploaded"}), 400

        f = request.files["file"]
        filename = secure_filename(f.filename or "")
        # if filename is empty, still allow but use .webm by default
        if filename == "":
            filename = "upload.webm"
        if not allowed_file(filename):
            return jsonify({"error": "invalid_file_type", "allowed": sorted(list(ALLOWED_EXTENSIONS))}), 400

        qid_raw = request.form.get("question_id", "Q001") or "Q001"
        qid = secure_filename(str(qid_raw)).strip() or "Q001"

        # Save temp upload
        ext = filename.rsplit('.', 1)[1] if '.' in filename else "webm"
        temp_name = f"{qid}_temp_{uuid.uuid4().hex}.{ext}"
        temp_name = secure_filename(temp_name)
        temp_path = os.path.join(UPLOAD_DIR, temp_name)
        try:
            f.save(temp_path)
        except Exception as e:
            app.logger.exception("Failed to save uploaded file")
            return jsonify({"error": "failed_to_save_upload", "details": str(e)}), 500

        # Target wav path
        wav_path = _next_available_wav_path(qid, UPLOAD_DIR)

        # Convert using ffmpeg (resample to 16k mono)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", temp_path,
            "-ar", "16000",
            "-ac", "1",
            wav_path,
        ]

        conversion_failed = False
        conv_err_details = None
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            conversion_failed = True
            out = (e.stdout.decode("utf-8", errors="ignore") if e.stdout else "")
            err = (e.stderr.decode("utf-8", errors="ignore") if e.stderr else "")
            conv_err_details = {"returncode": getattr(e, "returncode", None), "stdout": out, "stderr": err}
            app.logger.error("ffmpeg conversion failed: %s", conv_err_details)
        except FileNotFoundError:
            conversion_failed = True
            conv_err_details = "ffmpeg not found. Install ffmpeg and ensure it is on PATH."
            app.logger.error(conv_err_details)
        finally:
            # Remove temp to avoid disk accumulation (we keep WAV if conversion succeeded)
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                app.logger.warning("Failed to remove temp upload: %s", temp_path)

        if conversion_failed:
            return jsonify({"error": "ffmpeg_conversion_failed", "details": conv_err_details}), 500

        # Evaluate the WAV file
        try:
            result = evaluate_audio(wav_path, question_id=qid)
        except Exception as e:
            app.logger.exception("Evaluation pipeline failed")
            # keep wav_path for debugging
            return jsonify({"error": "evaluation_failed", "details": str(e), "audio_path": wav_path}), 500

        # Ensure consistent keys in the returned result
        if "audio_path" not in result:
            result["audio_path"] = wav_path
        if "audio_url" not in result:
            result["audio_url"] = "/audio/" + os.path.basename(result["audio_path"])
        if "timestamp" not in result:
            result["timestamp"] = datetime.datetime.utcnow().isoformat()

        # Append evaluation log (best-effort)
        try:
            log_row = {
                "timestamp": result.get("timestamp"),
                "question_id": result.get("question_id"),
                "question_text": result.get("question_text", ""),
                "transcript": result.get("transcript"),
                "asr_confidence": result.get("asr_confidence"),
                "best_reference": result.get("best_reference"),
                "semantic_similarity": result.get("semantic_similarity"),
                "final_score_0_10": result.get("final_score_0_10"),
                "audio_path": result.get("audio_path"),
                "audio_url": result.get("audio_url"),
            }
            append_evaluation_log(log_row)
        except Exception as e:
            app.logger.warning("Failed to append evaluation log: %s", e)

        # Success: always return a JSON response.
        return jsonify(result)

    except Exception as e:
        app.logger.exception("Unexpected error in /api/evaluate")
        return jsonify({"error": "internal_server_error", "details": str(e)}), 500



@app.route("/audio/<path:filename>")
def serve_audio(filename):
    safe = secure_filename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return jsonify({"error": "audio_not_found"}), 404
    return send_from_directory(UPLOAD_DIR, safe, as_attachment=False)


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    safe = secure_filename(filename)
    out_dir = os.path.join(os.getcwd(), "outputs")
    path = os.path.join(out_dir, safe)
    if not os.path.exists(path):
        return jsonify({"error": "output_file_not_found"}), 404
    return send_from_directory(out_dir, safe, as_attachment=False)


@app.route("/metrics")
def show_metrics():
    import json
    json_path = os.path.join("outputs", "evaluation_metrics.json")
    png_path = os.path.join("outputs", "similarity_results.png")

    if not os.path.exists(json_path):
        return "<h2>No metrics available — run batch evaluation first.</h2>"

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as e:
        app.logger.exception("Failed to read metrics file")
        return f"<h2>Failed to read metrics file: {e}</h2>", 500

    html = f"""<html><head><title>Evaluation Metrics</title>
    <style>body{{font-family:Arial;padding:20px;background:#f2f2f2}}.card{{background:white;padding:20px;border-radius:10px;width:700px;margin:auto}}.title{{font-size:22px;font-weight:700}}.metric{{font-size:18px;margin:5px 0}}img{{width:100%;margin-top:20px;border-radius:10px}}</style>
    </head><body><div class='card'><div class='title'>Model Accuracy & Evaluation Metrics</div>"""
    for k, v in metrics.items():
        if k != "plot_path":
            html += f"<div class='metric'><b>{k}:</b> {v}</div>"
    if os.path.exists(png_path):
        html += f"<img src='/outputs/{os.path.basename(png_path)}' />"
    html += "</div></body></html>"
    return html


if __name__ == "__main__":
    # Use 127.0.0.1 if you want to restrict to localhost; 0.0.0.0 listens on all interfaces.
    debug_mode = bool(os.environ.get("FLASK_DEBUG", "1") == "1")
    app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8501)),
    debug=False,
    use_reloader=False
)

