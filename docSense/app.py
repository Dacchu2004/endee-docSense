import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from ingestion import ingest_document
from rag import answer_question

load_dotenv()

app = Flask(__name__)

@app.route("/")
def index():
    """Serves the main UI page."""
    return render_template("index.html")

@app.route("/health")
def health():
    """Health check — confirms Flask + Endee are reachable."""
    try:
        from endee_client import list_sources
        info = list_sources()
        return jsonify({
            "status": "ok",
            "endee": "connected",
            "index_info": info
        })
    except Exception as e:
        return jsonify({
            "status": "ok",
            "endee": "error",
            "error": str(e)
        }), 200


@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Accepts a file upload, extracts text,
    chunks it, embeds it, and stores in Endee.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = [".pdf", ".txt", ".docx"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type. Allowed: {allowed}"}), 400

    try:
        file_bytes = file.read()
        chunk_count = ingest_document(file_bytes, file.filename)

        if chunk_count == 0:
            return jsonify({"error": "Could not extract text from file"}), 400

        return jsonify({
            "message": f"✅ Successfully ingested '{file.filename}'",
            "chunks_stored": chunk_count,
            "filename": file.filename
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """
    Accepts a question, runs RAG pipeline,
    returns answer + source chunks.
    """
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"].strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        result = answer_question(question)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 DocSense is starting...")
    print("📦 Endee Vector DB: http://localhost:8080")
    print("🌐 App UI: http://localhost:5000")
    app.run(debug=True, port=5000)