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

@app.route("/files", methods=["GET"])
def list_files():
    """Returns all ingested files from the registry."""
    from ingestion import get_ingested_files, load_registry
    registry = load_registry()
    files = [
        {
            "filename": fname,
            "chunks": meta.get("chunk_count", 0)
        }
        for fname, meta in registry.items()
    ]
    return jsonify({"files": files})


@app.route("/delete-file", methods=["POST"])
def delete_file():
    """Removes a file and all its chunks from Endee."""
    data = request.get_json()
    filename = data.get("filename", "").strip()
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    try:
        from ingestion import load_registry, save_registry
        from endee_client import client, INDEX_NAME
        from rag import clear_cache

        registry = load_registry()
        if filename not in registry:
            return jsonify({"error": "File not found in registry"}), 404

        # Delete all vectors for this file
        old_ids = registry[filename].get("chunk_ids", [])
        index = client.get_index(INDEX_NAME)
        deleted = 0
        for chunk_id in old_ids:
            try:
                index.delete_vector(chunk_id)
                deleted += 1
            except:
                pass

        # Remove from registry
        del registry[filename]
        save_registry(registry)
        clear_cache()

        return jsonify({
            "message": f"✅ Deleted '{filename}' ({deleted} chunks removed)",
            "filename": filename
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Accepts a file upload, extracts text,
    chunks it, embeds it, and stores in Endee.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}),400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}),400

    allowed = [".pdf", ".txt", ".docx"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type. Allowed: {allowed}"}),400

    try:
        file_bytes = file.read()
        result = ingest_document(file_bytes, file.filename)
        from rag import clear_cache
        clear_cache()

        if not result["success"]:
            return jsonify({"error": result["error"]}),400

        return jsonify({
            "message": f"✅ Successfully ingested '{result['filename']}'",
            "chunks_stored": result["chunks"],
            "filename": result["filename"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}),500


@app.route("/ask", methods=["POST"])
def ask():
    """
    Accepts a question, runs RAG pipeline,
    returns answer + source chunks.
    """
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}),400

    question = data["question"].strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}),400

    try:
        source_filter = data.get("source_filter", None)
        result = answer_question(question, source_filter=source_filter)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}),500

if __name__ == "__main__":
    print("DocSense is starting...")
    print("Pre warming embedding model...")
    from embedder import embed
    embed(["warmup"])
    print("Model ready!")
    print("Endee Vector DB: http://localhost:8080")
    print("App UI: http://localhost:5000")
    app.run(debug=True, port=5000)