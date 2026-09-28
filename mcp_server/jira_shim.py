"""
ISDO Lab C2 — Mock Jira Service Management REST API (Flask Shim)
Mimics the Jira REST API for service requests so the MCP server
can make real HTTP calls without touching production.

Endpoints:
  GET  /rest/agile/1.0/board/requests   — list requests (?request_type= ?priority= ?assignee= ?status=)
  GET  /rest/api/2/issue/<key>          — get one request (Jira-style nested 'fields')
  PUT  /rest/api/2/issue/<key>          — update a request (flat or Jira 'fields' format)
  POST /rest/api/2/issue                — create a request
  GET  /health                          — service status

Run from the project root:  python mcp_server/jira_shim.py      (port 5002)
"""

import csv
import os

from flask import Flask, jsonify, request

app = Flask(__name__)
PORT = 5002
DATA_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "requests.csv"))
FILTERS = ["request_type", "priority", "assignee", "status"]
# Jira nested field -> (flat CSV column, key inside the nested object)
JIRA_FIELDS = {"summary": ("summary", None), "priority": ("priority", "name"),
               "status": ("status", "name"), "assignee": ("assignee", "displayName"),
               "issuetype": ("request_type", "name"), "customfield_sla": ("sla", None)}


def load_requests():
    data = {}
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            for line, row in enumerate(csv.DictReader(f), start=2):
                if None in row:  # more values than headers: an unquoted comma
                    print(f"Warning: skipped line {line} of {DATA_FILE} (unquoted comma?)")
                    continue
                data[row["key"]] = row
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with empty dataset.")
    return data


REQUESTS = load_requests()  # in-memory store; resets on restart


def to_flat(fields):
    """Accept Jira format ({'status': {'name': 'Done'}}) or flat ({'status': 'Done'})."""
    flat = {}
    for name, value in fields.items():
        column, inner = JIRA_FIELDS.get(name, (name, None))
        if isinstance(value, dict):
            value = value.get(inner) if inner else None
        if value is not None:
            flat[column] = value
    flat.pop("key", None)  # the record key cannot be changed
    return flat


def to_jira(req):
    return {"key": req["key"], "fields": {
        "summary": req.get("summary"), "priority": {"name": req.get("priority")},
        "status": {"name": req.get("status")}, "assignee": {"displayName": req.get("assignee")},
        "customfield_sla": req.get("sla"), "issuetype": {"name": req.get("request_type")}}}


@app.route("/rest/agile/1.0/board/requests", methods=["GET"])
def list_requests():
    results = list(REQUESTS.values())
    for key in FILTERS:
        val = request.args.get(key)
        if val:
            results = [r for r in results if r.get(key, "").lower() == val.lower()]
    return jsonify({"issues": results, "total": len(results)})


@app.route("/rest/api/2/issue/<key>", methods=["GET"])
def get_request(key):
    if key not in REQUESTS:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    return jsonify(to_jira(REQUESTS[key]))


@app.route("/rest/api/2/issue/<key>", methods=["PUT"])
def update_request(key):
    if key not in REQUESTS:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data:
        return jsonify({"errorMessages": ["Send a JSON body, e.g. {\"fields\": {\"status\": {\"name\": \"Done\"}}}"]}), 400
    updates = to_flat(data.get("fields", data))
    REQUESTS[key].update(updates)
    print(f"[Jira Mock] Updated {key}: {updates}")
    return jsonify({"key": key, "message": "Updated successfully"})


@app.route("/rest/api/2/issue", methods=["POST"])
def create_request():
    data = request.get_json(silent=True)
    fields = to_flat(data.get("fields", data)) if isinstance(data, dict) else {}
    if not fields.get("summary"):
        return jsonify({"errorMessages": ["Missing required field: summary"]}), 400
    nums = [int(k.split("-")[1]) for k in REQUESTS if k.split("-")[-1].isdigit()]
    key = f"REQ-{max(nums) + 1 if nums else 1001}"
    REQUESTS[key] = {"key": key, "summary": "", "request_type": "", "priority": "Medium",
                     "assignee": "", "sla": "", "status": "Open", **fields}
    print(f"[Jira Mock] Created request: {key}")
    return jsonify({"key": key, "message": "Request created"}), 201


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Jira Mock", "requests_loaded": len(REQUESTS)})


if __name__ == "__main__":
    print(f"Jira Mock API starting on http://localhost:{PORT}")
    print(f"Loaded {len(REQUESTS)} requests from {DATA_FILE}")
    print("Endpoints: GET /rest/agile/1.0/board/requests  |  GET /health")
    app.run(port=PORT, debug=False)  # no auto-reloader, so in-memory updates survive
