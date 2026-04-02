import os
import io
import base64
import json
import functools
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Config ---
OCR_API_KEY = os.getenv("OCR_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_KEY = os.getenv("API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

# --- Auth ---
def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("x-api-key")
        print(f"Received API key: {key}")  #
        if not key or key != API_KEY:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ─── DOCX extraction (local, no OCR needed) ──────────────────────────────────

def extract_docx_text(file_bytes: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ─── OCR.space extraction (PDF + images only) ────────────────────────────────

def ocr_from_base64(file_b64: str, file_type: str) -> str:
    content_type = "application/pdf" if file_type == "pdf" else "image/jpeg"

    payload = {
        "apikey": OCR_API_KEY,
        "language": "eng",
        "isOverlayRequired": False,
        "detectOrientation": True,
        "isTable": True,
        "OCREngine": 2,
        "base64Image": f"data:{content_type};base64,{file_b64}",
    }

    response = requests.post(
        "https://api.ocr.space/parse/image",
        data=payload,
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()

    if result.get("IsErroredOnProcessing"):
        error_msg = result.get("ErrorMessage", ["OCR failed"])[0]
        raise ValueError(f"OCR.space error: {error_msg}")

    parsed = result.get("ParsedResults", [])
    if not parsed:
        raise ValueError("OCR.space returned no results")

    return "\n".join(p.get("ParsedText", "") for p in parsed).strip()


# ─── Groq AI analysis ────────────────────────────────────────────────────────

def analyse_with_groq(text: str) -> dict:
    prompt = f"""You are a document analysis expert. Analyse the document text below and respond with ONLY a valid JSON object — no markdown, no explanation.

Return exactly this structure:
{{
  "summary": "<concise 1-3 sentence summary of the document's purpose and key points>",
  "entities": {{
    "names": ["<person names found>"],
    "dates": ["<dates found>"],
    "organizations": ["<company or organisation names found>"],
    "amounts": ["<monetary amounts found, with currency symbols>"]
  }},
  "sentiment": "<exactly one of: Positive, Neutral, Negative>"
}}

Rules:
- Only include entities actually present in the text. Use [] for categories with none.
- Sentiment should reflect the overall tone of the document.
- Return ONLY the JSON object, nothing else.

Document:
\"\"\"
{text[:4000]}
\"\"\"
"""

    chat = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )

    raw = chat.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/document-analyze", methods=["POST"])
@require_api_key
def document_analyze():
    """
    API endpoint to analyze a single document at a time.
    Accepts one document in Base64 format per request.
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"status": "error", "message": "Request body must be JSON"}), 400

    file_name = body.get("fileName", "")
    file_type = (body.get("fileType") or "").lower().strip()
    file_b64  = body.get("fileBase64", "")

    if not file_type or not file_b64:
        return jsonify({"status": "error", "message": "fileType and fileBase64 are required"}), 400

    if file_type not in ("pdf", "docx", "image"):
        return jsonify({"status": "error", "message": "fileType must be pdf, docx, or image"}), 400

    try:
        file_bytes = base64.b64decode(file_b64, validate=True)
    except Exception:
        return jsonify({"status": "error", "message": "Invalid base64 encoding"}), 400

    # Step 1: Extract text
    try:
        if file_type == "docx":
            text = extract_docx_text(file_bytes)
        else:
            text = ocr_from_base64(file_b64, file_type)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Text extraction failed: {str(e)}"}), 500

    if not text.strip():
        return jsonify({"status": "error", "message": "No text could be extracted from the document"}), 422

    # Step 2: AI analysis
    try:
        analysis = analyse_with_groq(text)
    except json.JSONDecodeError:
        return jsonify({"status": "error", "message": "AI analysis returned invalid JSON"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"AI analysis failed: {str(e)}"}), 500

    return jsonify({
        "status":    "success",
        "fileName":  file_name,
        "summary":   analysis.get("summary", ""),
        "entities":  analysis.get("entities", {
            "names": [], "dates": [], "organizations": [], "amounts": []
        }),
        "sentiment": analysis.get("sentiment", "Neutral"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
