#!/usr/bin/env python3
"""
KL Marketing Production Tool — Web Version
Subtitle Video:
- TC1: Khmer/Multilingual keyword sub (OpenAI Whisper API + Gemini)
- TC2: TikTok Vietnamese full sub (OpenAI Whisper API + rules)
Enhance hình ảnh:
- GPT Image enhance
- ESRGAN / adjustable enhancement
"""

import os
import re
import io
import json
import uuid
import base64
import tempfile
import threading
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from functools import lru_cache
from flask import Flask, render_template, request, jsonify, send_file
from openai import OpenAI
from google import genai
from PIL import Image, ImageEnhance, ImageFilter

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-flash-lite"
TC1_GEMINI_MODELS = [
    os.environ.get("TC1_GEMINI_MODEL", "").strip(),
    "gemini-3.1-flash",
    "gemini-3.1-flash-lite",
]
TC1_GEMINI_MODELS = [m for m in TC1_GEMINI_MODELS if m]

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
IMAGE_DIR = Path(tempfile.gettempdir()) / "klm_images"
IMAGE_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR = Path(tempfile.gettempdir()) / "klm_realesrgan_weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
JOBS_DIR = Path(tempfile.gettempdir()) / "klm_jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB


# ── Whisper API (OpenAI) ──────────────────────────────────────────────────────

def transcribe_whisper_api(audio_path: str, language: str = None) -> dict:
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

def _unwrap_json_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return text


def _looks_bad_vi_full(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if re.fullmatch(r"[\d\s\.,:/\\-]+", t):
        return True
    if len(t) < 5:
        return True
    return False


def _tc1_generate_with_fallback(client, prompt: str):
    last_error = None
    for model_name in TC1_GEMINI_MODELS:
        try:
            return client.models.generate_content(model=model_name, contents=prompt)
        except Exception as e:
            last_error = e
            if "404" in str(e) or "NOT_FOUND" in str(e) or "not available" in str(e):
                continue
            raise
    raise last_error or RuntimeError("Không có Gemini model khả dụng cho TC1")


def _transcribe_tc1_from_audio(client, audio_path: str, lang_name: str, timing_segments: list[dict]) -> dict[int, dict]:
    if not timing_segments:
        return {}
    payload = [
        {
            "index": s["index"],
            "start": s.get("start"),
            "end": s.get("end"),
        }
        for s in timing_segments
    ]
    uploaded = client.files.upload(file=audio_path)
    try:
        prompt = f"""Bạn đang xử lý subtitle audio.
Ngôn ngữ chính của audio: {lang_name}

Audio đính kèm mới là SOURCE OF TRUTH cho nội dung nói.
Danh sách segment bên dưới chỉ dùng để GIỮ KHUNG index/timecode.

Nhiệm vụ DUY NHẤT:
- Nghe audio đính kèm.
- Với mỗi `index`, trả về `original` là câu gốc / transcript đúng nhất có thể theo audio.

Quy tắc bắt buộc:
- Giữ nguyên số lượng object và đúng `index` như input.
- Chỉ transcript câu gốc, KHÔNG dịch ở bước này.
- Dựa vào AUDIO để hiểu câu, không phụ thuộc transcript giả định bên ngoài.
- Nếu segment ngắn, vẫn cố transcript đủ nhất có thể.
- Chỉ trả JSON array, không markdown.

Input segments:
{json.dumps(payload, ensure_ascii=False)}

Output schema:
[{{"index":0,"original":"..."}}]"""
        resp = _tc1_generate_with_fallback(client, [uploaded, prompt])
        translated = json.loads(_unwrap_json_text(resp.text))
        return {int(item.get("index", -1)): item for item in translated if isinstance(item, dict)}
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass


def _translate_tc1_from_transcript_batch(client, lang_name: str, batch: list[dict]) -> dict[int, str]:
    if not batch:
        return {}
    payload = []
    for i, s in enumerate(batch):
        payload.append({
            "index": s["id"],
            "original": s["original"],
            "prev_original": batch[i-1]["original"] if i > 0 else "",
            "next_original": batch[i+1]["original"] if i + 1 < len(batch) else "",
        })
    prompt = f"""Bạn là biên dịch viên subtitle rất kỹ, dịch sang tiếng Việt cho video quảng cáo/làm đẹp.
Ngôn ngữ gốc: {lang_name}

Nhiệm vụ DUY NHẤT:
- Dịch `original` sang `vi_full` cho từng item.
- Được quyền dùng `prev_original` và `next_original` để hiểu NGỮ CẢNH.

Yêu cầu cực kỳ quan trọng:
- `vi_full` phải ĐỦ Ý, không được lược bỏ chi tiết quan trọng.
- Câu Việt có thể dài hơn câu gốc nếu cần để đủ nghĩa.
- Ưu tiên ĐÚNG và ĐỦ hơn là ngắn.
- Không văn vẻ hóa, không bịa thêm ý, không thêm cảm xúc không có trong câu gốc.
- Không được dịch thành câu quá ngắn nếu câu gốc có nhiều ý.
- Giữ nguyên `index`.
- Chỉ trả JSON array, không markdown.

Input:
{json.dumps(payload, ensure_ascii=False)}

Output schema:
[{{"index":0,"vi_full":"..."}}]"""
    resp = _tc1_generate_with_fallback(client, prompt)
    translated = json.loads(_unwrap_json_text(resp.text))
    return {int(item.get("index", -1)): (item.get("vi_full") or "").strip() for item in translated if isinstance(item, dict)}


def _extract_tc1_keywords_batch(client, lang_name: str, batch: list[dict]) -> dict[int, list]:
    if not batch:
        return {}
    payload = [
        {
            "index": s["id"],
            "original": s["original"],
            "vi_full": s["vi_full"],
        }
        for s in batch
    ]
    prompt = f"""Bạn đang làm bước BÓC TÁCH KEYWORD cho subtitle.
Ngôn ngữ gốc: {lang_name}

Nhiệm vụ DUY NHẤT:
- Dựa trên `original` và `vi_full` đã có sẵn,
- trích ra 1-3 keyword/phrase quan trọng cho mỗi segment.

Quy tắc:
- KHÔNG sửa lại `vi_full`.
- KHÔNG dịch lại cả câu.
- Chỉ lấy keyword thật sự quan trọng: lợi ích, USP, số liệu, tên thương hiệu, CTA.
- Nếu không có keyword rõ, vẫn trả mảng rỗng hoặc 1 keyword an toàn, không bịa.
- Giữ nguyên `index`.
- Chỉ trả JSON array, không markdown.

Input:
{json.dumps(payload, ensure_ascii=False)}

Output schema:
[{{"index":0,"keywords":[{{"original":"...","vi":"..."}}]}}]"""
    resp = _tc1_generate_with_fallback(client, prompt)
    extracted = json.loads(_unwrap_json_text(resp.text))
    return {
        int(item.get("index", -1)): item.get("keywords", [])
        for item in extracted if isinstance(item, dict)
    }


def tc1_process(audio_path: str, language: str, gemini_key: str) -> dict:
    whisper_result = transcribe_whisper_api(audio_path, language if language != "auto" else None)
    whisper_segs = whisper_result["segments"]
    detected_lang = whisper_result["language"]

    client = genai.Client(api_key=gemini_key)
    lang_name = LANG_NAMES.get(detected_lang, detected_lang)
    timing_segments = [
        {
            "index": i,
            "start": ws["start"],
            "end": ws["end"],
        }
        for i, ws in enumerate(whisper_segs)
    ]

    by_index = _transcribe_tc1_from_audio(client, audio_path, lang_name, timing_segments)

    segments = []
    for i, ws in enumerate(whisper_segs):
        gs = by_index.get(i, {})
        segment = {
            "id": i,
            "start": ws["start"],
            "end": ws["end"],
            "original": (gs.get("original") or ws["text"] or "").strip(),
            "vi_full": "",
            "keywords": [],
        }
        segments.append(segment)

    translate_batch_size = 10
    for start_idx in range(0, len(segments), translate_batch_size):
        batch = segments[start_idx:start_idx + translate_batch_size]
        vi_map = _translate_tc1_from_transcript_batch(client, lang_name, batch)
        for seg in batch:
            vi = (vi_map.get(seg["id"]) or "").strip()
            if not _looks_bad_vi_full(vi):
                seg["vi_full"] = vi

    keyword_batch_size = 12
    for start_idx in range(0, len(segments), keyword_batch_size):
        batch = segments[start_idx:start_idx + keyword_batch_size]
        keyword_map = _extract_tc1_keywords_batch(client, lang_name, batch)
        for seg in batch:
            kws = keyword_map.get(seg["id"])
            if isinstance(kws, list):
                seg["keywords"] = kws

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


# ── Image enhancement helpers ─────────────────────────────────────────────────

def pick_gpt_size(w: int, h: int) -> str:
    if w >= h * 1.2:
        return "1536x1024"
    if h >= w * 1.2:
        return "1024x1536"
    return "1024x1024"


def save_pil_image(img: Image.Image, suffix: str = ".png") -> str:
    out = IMAGE_DIR / f"enhanced_{uuid.uuid4().hex[:10]}{suffix}"
    img.save(out, format="PNG")
    return str(out)


def enhance_with_gpt_image(image_path: str, prompt: str = "") -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Server chưa cấu hình OpenAI key")

    client = OpenAI(api_key=OPENAI_API_KEY)
    with Image.open(image_path) as im:
        w, h = im.size
        size = pick_gpt_size(w, h)

    final_prompt = (
        prompt.strip()
        or "Enhance this image naturally. Improve clarity, detail, micro-contrast, and overall polish while preserving the original composition, people, objects, and scene. Avoid changing identity, text, layout, or adding new elements."
    )

    with open(image_path, "rb") as f:
        result = client.images.edit(
            model="gpt-image-1",
            image=f,
            prompt=final_prompt,
            size=size,
        )

    b64 = result.data[0].b64_json
    output_bytes = base64.b64decode(b64)
    out_path = IMAGE_DIR / f"gpt_enhance_{uuid.uuid4().hex[:10]}.png"
    out_path.write_bytes(output_bytes)
    return str(out_path)


@lru_cache(maxsize=1)
def load_realesrgan_runtime():
    import sys
    import torch
    try:
        import torchvision.transforms._functional_tensor as _ft
        sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
    except Exception:
        pass
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    model_path = load_file_from_url(
        url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        model_dir=str(WEIGHTS_DIR), progress=True, file_name=None)
    wdn_model_path = load_file_from_url(
        url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
        model_dir=str(WEIGHTS_DIR), progress=True, file_name=None)

    upsampler = RealESRGANer(
        scale=4,
        model_path=[model_path, wdn_model_path],
        dni_weight=[0.5, 0.5],
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None,
    )
    return upsampler


def classic_adjust(img: Image.Image, sharpness: float, contrast: float, color: float) -> Image.Image:
    if contrast != 1:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if color != 1:
        img = ImageEnhance.Color(img).enhance(color)
    if sharpness != 1:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img


def resize_longest_side(img: Image.Image, target_longest: int) -> Image.Image:
    longest = max(img.width, img.height)
    if longest <= 0 or longest == target_longest:
        return img.copy()
    ratio = target_longest / float(longest)
    new_size = (
        max(1, round(img.width * ratio)),
        max(1, round(img.height * ratio)),
    )
    return img.resize(new_size, Image.LANCZOS)


def enhance_with_esrgan_adjustable(image_path: str,
                                   strength: float = 0.55,
                                   sharpness: float = 1.15,
                                   contrast: float = 1.04,
                                   color: float = 1.02,
                                   outscale: float = 1.0):
    del outscale  # size is now normalized server-side for reliability
    with Image.open(image_path) as src:
        original = src.convert("RGB")

    input_longest = max(original.width, original.height)
    target_longest = 1920
    working_longest = 1024 if input_longest > target_longest else min(target_longest, input_longest)

    working = resize_longest_side(original, working_longest)
    base = resize_longest_side(original, target_longest) if input_longest != target_longest else original.copy()
    if base.size != working.size:
        base = resize_longest_side(base, target_longest)

    work_input_path = IMAGE_DIR / f"esrgan_work_{uuid.uuid4().hex[:10]}.png"
    working.save(str(work_input_path))

    try:
        import cv2
        upsampler = load_realesrgan_runtime()
        bgr = cv2.imread(str(work_input_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Không đọc được ảnh đầu vào")
        output, _ = upsampler.enhance(bgr, outscale=2)
        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        enhanced = Image.fromarray(rgb)
        enhanced = resize_longest_side(enhanced, target_longest)
    except Exception as e:
        raise RuntimeError(f"ESRGAN runtime không chạy được: {e}") from e
    finally:
        try:
            work_input_path.unlink(missing_ok=True)
        except Exception:
            pass

    if base.size != enhanced.size:
        base = base.resize(enhanced.size, Image.LANCZOS)

    mix = max(0.0, min(1.0, strength))
    blended = Image.blend(base, enhanced, mix)
    blended = blended.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(20 + mix * 60), threshold=2))
    blended = classic_adjust(blended, sharpness=sharpness, contrast=contrast, color=color)
    return save_pil_image(blended), "esrgan"


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
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".srt", encoding="utf-8", delete=False)
    tmp.write(srt)
    tmp.close()
    return send_file(tmp.name, as_attachment=True, download_name=fname)


def _run_esrgan_job(job_id: str, input_path: str, params: dict):
    """Background thread: run ESRGAN and write result to JOBS_DIR/<job_id>/."""
    job_dir = JOBS_DIR / job_id
    status_file = job_dir / "status.json"
    output_file = job_dir / "output.png"
    try:
        out_path, _ = enhance_with_esrgan_adjustable(input_path, **params)
        import shutil
        shutil.copy(out_path, str(output_file))
        status_file.write_text('{"status":"done"}', encoding="utf-8")
    except Exception as e:
        status_file.write_text(json.dumps({"status": "error", "error": str(e)}), encoding="utf-8")


@app.route("/api/enhance-image", methods=["POST"])
def enhance_image():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    image = request.files["image"]
    mode = request.form.get("mode", "gpt")
    ext = Path(image.filename or "image.png").suffix or ".png"

    if mode == "gpt":
        input_path = IMAGE_DIR / f"input_{uuid.uuid4().hex[:10]}{ext}"
        image.save(str(input_path))
        try:
            prompt = request.form.get("prompt", "")
            output_path = enhance_with_gpt_image(str(input_path), prompt)
            response = send_file(output_path, mimetype="image/png", as_attachment=False, download_name=Path(output_path).name)
            response.headers["X-Enhance-Engine"] = "gpt-image-1"
            response.headers["X-Output-Filename"] = Path(output_path).name
            return response
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        # Async ESRGAN path
        job_id = uuid.uuid4().hex[:12]
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        input_path = job_dir / f"input{ext}"
        image.save(str(input_path))
        params = {
            "strength": float(request.form.get("strength", "55")) / 100.0,
            "sharpness": float(request.form.get("sharpness", "115")) / 100.0,
            "contrast": float(request.form.get("contrast", "104")) / 100.0,
            "color": float(request.form.get("color", "102")) / 100.0,
            "outscale": float(request.form.get("outscale", "1")),
        }
        (job_dir / "status.json").write_text('{"status":"processing"}', encoding="utf-8")
        t = threading.Thread(target=_run_esrgan_job, args=(job_id, str(input_path), params), daemon=True)
        t.start()
        return jsonify({"job_id": job_id, "status": "processing"})


@app.route("/api/enhance-status/<job_id>")
def enhance_status(job_id):
    job_dir = JOBS_DIR / job_id
    status_file = job_dir / "status.json"
    output_file = job_dir / "output.png"
    if not status_file.exists():
        return jsonify({"error": "Job not found"}), 404
    status = json.loads(status_file.read_text(encoding="utf-8"))
    if status.get("status") == "done" and output_file.exists():
        response = send_file(str(output_file), mimetype="image/png", as_attachment=False, download_name=f"enhanced_{job_id}.png")
        response.headers["X-Enhance-Engine"] = "esrgan"
        return response
    elif status.get("status") == "error":
        return jsonify({"error": status.get("error", "Unknown error")}), 500
    return jsonify({"status": "processing"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"🎬 KL Marketing Production (Web) running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
