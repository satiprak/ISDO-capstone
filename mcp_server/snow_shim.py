"""
ISDO Lab C2 — Mock ServiceNow REST API (Flask Shim)
Mimics the ServiceNow Table API so the MCP server can make real HTTP calls
without touching a production system.

Endpoints:
  GET   /api/now/table/incident            — list incidents (?category= ?priority= ?state= ?assignment_group=)
  GET   /api/now/table/incident/<number>   — get one incident
  PATCH /api/now/table/incident/<number>   — update fields in memory (e.g. state, work_notes)
  POST  /api/now/table/incident            — create an incident (number auto-generated if omitted)
  GET   /health                            — service status

Run from the project root:  python mcp_server/snow_shim.py      (port 5001)
"""

import csv
import os

from flask import Flask, jsonify, request

app = Flask(__name__)
PORT = 5001
DATA_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "incidents.csv"))
FILTERS = ["category", "state", "priority", "assignment_group"]


def load_incidents():
    incidents = {}
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            for line, row in enumerate(csv.DictReader(f), start=2):
                if None in row:  # more values than headers: an unquoted comma
                    print(f"Warning: skipped line {line} of {DATA_FILE} (unquoted comma?)")
                    continue
                incidents[row["number"]] = row
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with empty dataset.")
    return incidents


# In-memory store (simulates the ServiceNow DB for this session; resets on restart)
INCIDENTS = load_incidents()


def next_number():
    nums = [int(n[3:]) for n in INCIDENTS if n.startswith("INC") and n[3:].isdigit()]
    return f"INC{(max(nums) + 1 if nums else 1001):07d}"


@app.route("/api/now/table/incident", methods=["GET"])
def list_incidents():
    results = list(INCIDENTS.values())
    for key in FILTERS:
        val = request.args.get(key)
        if val:
            results = [r for r in results if r.get(key, "").lower() == val.lower()]
    return jsonify({"result": results, "total": len(results)})  # ServiceNow envelope


@app.route("/api/now/table/incident/<number>", methods=["GET"])
def get_incident(number):
    incident = INCIDENTS.get(number)
    if not incident:
        return jsonify({"error": f"Incident {number} not found"}), 404
    return jsonify({"result": incident})


@app.route("/api/now/table/incident/<number>", methods=["PATCH"])
def update_incident(number):
    if number not in INCIDENTS:
        return jsonify({"error": f"Incident {number} not found"}), 404
    updates = request.get_json(silent=True)
    if not isinstance(updates, dict) or not updates:
        return jsonify({"error": "Send a JSON object body, e.g. {\"state\": \"In Progress\"}"}), 400
    updates.pop("number", None)  # the record key cannot be changed
    INCIDENTS[number].update(updates)
    print(f"[ServiceNow Mock] Updated {number}: {updates}")
    return jsonify({"result": INCIDENTS[number], "message": "Updated successfully"})


@app.route("/api/now/table/incident", methods=["POST"])
def create_incident():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data.get("short_description"):
        return jsonify({"error": "Missing required field: short_description"}), 400
    number = data.get("number") or next_number()
    if number in INCIDENTS:
        return jsonify({"error": f"Incident {number} already exists"}), 409
    record = {field: "" for field in ["description", "category", "priority",
                                      "assignment_group", "sla_due"]}
    record.update(data, number=number, state=data.get("state", "Open"))
    INCIDENTS[number] = record
    print(f"[ServiceNow Mock] Created incident: {number}")
    return jsonify({"result": record, "message": "Incident created"}), 201


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ServiceNow Mock", "incidents_loaded": len(INCIDENTS)})


if __name__ == "__main__":
    print(f"ServiceNow Mock API starting on http://localhost:{PORT}")
    print(f"Loaded {len(INCIDENTS)} incidents from {DATA_FILE}")
    print("Endpoints: GET /api/now/table/incident  |  GET /health")
    app.run(port=PORT, debug=False)  # no auto-reloader, so in-memory updates survive
