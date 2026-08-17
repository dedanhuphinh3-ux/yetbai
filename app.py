#!/usr/bin/env python3
"""
KL Marketing Production Tool — Web Version
TC1: Khmer/Multilingual keyword sub (OpenAI Whisper API + Gemini)
TC2: TikTok Vietnamese full sub (OpenAI Whisper API + rules)
Deploy: Render.com
"""

import os
import re
import json
import uuid
import tempfile
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from openai import OpenAI
from google import genai

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-flash-lite"

LANG_NAMES = {
    "auto": "Tự động nhận diện",
    "km": "Khmer (Campuchia)",
    "vi": "Tiếng Việt",
    "en": "Tiếng Anh",
    "zh": "Tiếng Trung",
    "ko": "Tiếng Hàn",
    "ja": "Tiếng Nhật",
    "th": "Tiếng Thái",
}

FILLER_PATTERNS = [
    (r'\bá\b', ''),
    (r'\bờ\b', ''),
    (r'\bừ\b', ''),
    (r'\bum\b', ''),
    (r'\buh\b', ''),
    (r'(\bê\b\s*){2,}', 'ê '),
    (r'\bhong\b', 'không'),
    (r'\bhông\b', 'không'),
    (r'\bk\b(?=\s)', 'không'),
]

UPLOAD_DIR = Path(tempfile.gettempdir()) / "klm_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB


# ── Whisper API (OpenAI) ──────────────────────────────────────────────────────

def transcribe_whisper_api(audio_path: str, language: str = None) -> dict:
    """Transcribe audio dùng OpenAI Whisper API — trả về segments với timestamps."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    kwargs = {
        "model": "whisper-1",
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }
    if language and language != "auto":
        kwargs["language"] = language

    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(file=f, **kwargs)

    segments = []
    for seg in result.segments:
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })
    return {
        "segments": segments,
        "language": result.language or language or "unknown",
    }


# ── TC1: Khmer / Multilingual ─────────────────────────────────────────────────

def tc1_process(audio_path: str, language: str, gemini_key: str) -> dict:
    """TC1: Whisper API timecodes + Gemini content."""
    # Step 1: Whisper API → timecodes
    print("[TC1] Whisper API → segment timecodes...")
    whisper_result = transcribe_whisper_api(audio_path, language if language != "auto" else None)
    whisper_segs = whisper_result["segments"]
    detected_lang = whisper_result["language"]

    # Step 2: Gemini → content
    print(f"[TC1] Gemini → transcript + translation + keywords ({detected_lang})...")
    client = genai.Client(api_key=gemini_key)

    uploaded = client.files.upload(file=audio_path)
    lang_name = LANG_NAMES.get(detected_lang, detected_lang)
    n = len(whisper_segs)

    prompt = f"""Expert multilingual subtitle creator.
Audio language: {lang_name}. Target: {n} segments.

For each segment return:
1. original: NATIVE SCRIPT (Khmer: ភាសាខ្មែរ NOT romanization)
2. vi_full: full natural Vietnamese translation
3. keywords: 1-3 key phrases (USP, benefits, numbers, brand names)

JSON array only (no markdown):
[{{"original":"...","vi_full":"...","keywords":[{{"original":"...","vi":"..."}}]}}]
Match {n} segments."""

    resp = client.models.generate_content(model=GEMINI_MODEL, contents=[uploaded, prompt])
    text = resp.text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1]).strip()
    gemini_segs = json.loads(text)
    try:
        client.files.delete(name=uploaded.name)
    except Exception:
        pass

    # Step 3: Merge
    segments = []
    for i, ws in enumerate(whisper_segs):
        gs = gemini_segs[i] if i < len(gemini_segs) else {}
        segments.append({
            "id": i,
            "start": ws["start"],
            "end": ws["end"],
            "original": gs.get("original", ws["text"]),
            "vi_full": gs.get("vi_full", ""),
            "keywords": gs.get("keywords", []),
        })

    return {"segments": segments, "language": detected_lang}


# ── TC2: TikTok Vietnamese ────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    t = text.strip()
    for pattern, replacement in FILLER_PATTERNS:
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    if t:
        t = t[0].upper() + t[1:]
    return t


def tc2_process(audio_path: str) -> dict:
    """TC2: OpenAI Whisper API → Vietnamese + cleanup."""
    print("[TC2] Whisper API → Vietnamese transcription...")
    result = transcribe_whisper_api(audio_path, language="vi")
    segments = []
    for i, seg in enumerate(result["segments"]):
        segments.append({
            "id": i,
            "start": seg["start"],
            "end": seg["end"],
            "original": seg["text"],
            "cleaned": clean_text(seg["text"]),
        })
    return {"segments": segments, "language": "vi"}


# ── SRT export ────────────────────────────────────────────────────────────────

def format_tc(sec: float) -> str:
    ms = int((sec % 1) * 1000)
    s = int(sec) % 60
    m = int(sec) // 60 % 60
    h = int(sec) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_srt_tc1(segments: list, mode: str = "keyword") -> str:
    lines = []
    idx = 1
    for seg in segments:
        s, e = format_tc(seg["start"]), format_tc(seg["end"])
        if mode == "keyword":
            lines += [str(idx), f"{s} --> {e}", f"[VI] {seg.get('vi_full', '')}", ""]
            idx += 1
            for kw in seg.get("keywords", []):
                lines += [str(idx), f"{s} --> {e}",
                          f"{kw.get('original', '')} / {kw.get('vi', '')}", ""]
                idx += 1
        elif mode == "keyword_only":
            for kw in seg.get("keywords", []):
                lines += [str(idx), f"{s} --> {e}",
                          f"{kw.get('original', '')} / {kw.get('vi', '')}", ""]
                idx += 1
        else:
            lines += [str(idx), f"{s} --> {e}", seg.get("vi_full", ""), ""]
            idx += 1
    return "\n".join(lines)


def export_srt_tc2(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        s, e = format_tc(seg["start"]), format_tc(seg["end"])
        lines += [str(i), f"{s} --> {e}", seg.get("cleaned", seg.get("original", "")), ""]
    return "\n".join(lines)


# ── In-memory job store ───────────────────────────────────────────────────────
JOBS = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", lang_names=LANG_NAMES)


@app.route("/api/translate", methods=["POST"])
def translate():
    gemini_key = request.json.get("gemini_key", "").strip()
    text = request.json.get("text", "").strip()
    target_lang = request.json.get("target_lang", "km")
    if not gemini_key or not text:
        return jsonify({"error": "Missing params"}), 400
    try:
        client = genai.Client(api_key=gemini_key)
        lang_name = LANG_NAMES.get(target_lang, target_lang)
        prompt = f'Translate to {lang_name} in native script (NOT romanization): "{text}"\nReturn ONLY translation.'
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return jsonify({"translated": resp.text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/validate-key", methods=["POST"])
def validate_key():
    """Validate Gemini API key."""
    key = request.json.get("key", "").strip()
    if not key:
        return jsonify({"ok": False, "error": "Key trống"}), 400
    try:
        client = genai.Client(api_key=key)
        client.models.generate_content(model=GEMINI_MODEL, contents="Reply: OK")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/upload", methods=["POST"])
def upload():
    if "audio" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["audio"]
    tool = request.form.get("tool", "tc1")
    language = request.form.get("language", "auto")
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp3"
    audio_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(str(audio_path))
    return jsonify({"job_id": job_id, "audio_path": str(audio_path),
                    "tool": tool, "language": language})


@app.route("/api/process/<job_id>", methods=["POST"])
def process(job_id):
    if not OPENAI_API_KEY:
        return jsonify({"error": "Server chưa cấu hình OpenAI key"}), 500
    data = request.json
    audio_path = data.get("audio_path")
    tool = data.get("tool", "tc1")
    language = data.get("language", "auto")
    gemini_key = data.get("gemini_key", "")

    try:
        if tool == "tc1":
            if not gemini_key:
                return jsonify({"error": "Cần Gemini API key cho TC1"}), 400
            result = tc1_process(audio_path, language, gemini_key)
        else:
            result = tc2_process(audio_path)

        result["job_id"] = job_id
        result["tool"] = tool
        JOBS[job_id] = result
        return jsonify({"success": True, **result})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/save/<job_id>", methods=["POST"])
def save_job(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "Not found"}), 404
    JOBS[job_id]["segments"] = request.json.get("segments", JOBS[job_id]["segments"])
    return jsonify({"success": True})


@app.route("/api/export/<job_id>")
def export(job_id):
    mode = request.args.get("mode", "keyword")
    if job_id not in JOBS:
        return jsonify({"error": "Not found"}), 404
    job = JOBS[job_id]
    tool = job.get("tool", "tc1")
    if tool == "tc1":
        srt = export_srt_tc1(job["segments"], mode)
        fname = f"khmer_sub_{mode}.srt"
    else:
        srt = export_srt_tc2(job["segments"])
        fname = "tiktok_vi_sub.srt"
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".srt",
                                      encoding="utf-8", delete=False)
    tmp.write(srt)
    tmp.close()
    return send_file(tmp.name, as_attachment=True, download_name=fname)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"🎬 KL Marketing Production (Web) running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
