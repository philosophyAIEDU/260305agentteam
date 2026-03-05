"""
Proposal Writing Agent Team - Flask Web Application
"""
import os
import uuid

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

import agents

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# In-memory session store for pipeline state
pipeline_states = {}


def get_state(sid):
    if sid not in pipeline_states:
        pipeline_states[sid] = {
            "api_key": None,
            "file_content": None,
            "analysis": None,
            "research_request": None,
            "research_report": None,
            "draft": None,
            "final": None,
        }
    return pipeline_states[sid]


@app.route("/")
def index():
    sid = str(uuid.uuid4())
    session["sid"] = sid
    get_state(sid)
    return render_template("index.html")


@app.route("/api/set-key", methods=["POST"])
def set_key():
    sid = session.get("sid")
    if not sid:
        return jsonify(error="No session"), 400
    data = request.get_json()
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify(error="API key is required"), 400
    state = get_state(sid)
    state["api_key"] = api_key
    return jsonify(ok=True)


@app.route("/api/upload", methods=["POST"])
def upload():
    sid = session.get("sid")
    if not sid:
        return jsonify(error="No session"), 400
    state = get_state(sid)
    if not state["api_key"]:
        return jsonify(error="API key not set"), 400

    f = request.files.get("file")
    if not f:
        return jsonify(error="No file uploaded"), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{sid}_{filename}")
    f.save(filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    state["file_content"] = content

    try:
        client = agents.get_client(state["api_key"])
        analysis = agents.step_analyze(client, content)
        state["analysis"] = analysis
        return jsonify(ok=True, analysis=analysis)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/research-request", methods=["POST"])
def research_request():
    sid = session.get("sid")
    if not sid:
        return jsonify(error="No session"), 400
    state = get_state(sid)
    if not state["analysis"]:
        return jsonify(error="Analysis not done yet"), 400

    data = request.get_json()
    user_answers = data.get("answers", "")

    try:
        client = agents.get_client(state["api_key"])
        rr = agents.step_research_request(client, state["analysis"], user_answers)
        state["research_request"] = rr
        return jsonify(ok=True, research_request=rr)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/research", methods=["POST"])
def research():
    sid = session.get("sid")
    if not sid:
        return jsonify(error="No session"), 400
    state = get_state(sid)
    if not state["research_request"]:
        return jsonify(error="Research request not done yet"), 400

    try:
        client = agents.get_client(state["api_key"])
        report = agents.step_research(client, state["research_request"])
        state["research_report"] = report
        return jsonify(ok=True, research_report=report)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/write", methods=["POST"])
def write_draft():
    sid = session.get("sid")
    if not sid:
        return jsonify(error="No session"), 400
    state = get_state(sid)
    if not state["research_report"]:
        return jsonify(error="Research not done yet"), 400

    try:
        client = agents.get_client(state["api_key"])
        draft = agents.step_write(client, state["analysis"], state["research_report"])
        state["draft"] = draft
        return jsonify(ok=True, draft=draft)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/review", methods=["POST"])
def review():
    sid = session.get("sid")
    if not sid:
        return jsonify(error="No session"), 400
    state = get_state(sid)
    if not state["draft"]:
        return jsonify(error="Draft not done yet"), 400

    try:
        client = agents.get_client(state["api_key"])
        final = agents.step_review(client, state["draft"])
        state["final"] = final
        return jsonify(ok=True, final=final)
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
