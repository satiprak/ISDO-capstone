"""
ISDO Lab C2 - Mock ServiceNow Table API (Flask)

  GET   /api/now/table/incident            all incidents (?category= ?priority=)
  GET   /api/now/table/incident/<number>   one incident
  PATCH /api/now/table/incident/<number>   update fields in memory
  GET   /health                            service status

Run from the project root:  python mcp_server/snow_shim.py   (port 5001)
"""

import csv
from pathlib import Path

from flask import Flask, jsonify, request

PORT = 5001
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "incidents.csv"
FILTERS = ["category", "priority"]

app = Flask(__name__)


def load_incidents() -> dict:
    """Read the CSV once at startup into {number: row}."""
    incidents = {}
    if not DATA_FILE.exists():
        print(f"Warning: {DATA_FILE} not found - starting with no incidents.")
        return incidents
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            if None in row:  # more values than headers, usually an unquoted comma
                print(f"Warning: skipped line {line} of {DATA_FILE.name} (unquoted comma?)")
                continue
            incidents[row["number"]] = row
    return incidents


INCIDENTS = load_incidents()  # in-memory store; resets when the server restarts


@app.get("/api/now/table/incident")
def list_incidents():
    results = list(INCIDENTS.values())
    for field in FILTERS:
        value = request.args.get(field)
        if value:
            results = [r for r in results if r[field].lower() == value.lower()]
    return jsonify({"result": results, "total": len(results)})


@app.get("/api/now/table/incident/<number>")
def get_incident(number):
    if number not in INCIDENTS:
        return jsonify({"error": f"Incident {number} not found"}), 404
    return jsonify({"result": INCIDENTS[number]})


@app.patch("/api/now/table/incident/<number>")
def update_incident(number):
    if number not in INCIDENTS:
        return jsonify({"error": f"Incident {number} not found"}), 404
    updates = request.get_json(silent=True)
    if not isinstance(updates, dict) or not updates:
        return jsonify({"error": 'Send a JSON body, e.g. {"state": "In Progress"}'}), 400
    updates.pop("number", None)  # the record key cannot change
    INCIDENTS[number].update(updates)
    print(f"[ServiceNow Mock] Updated {number}: {updates}")
    return jsonify({"result": INCIDENTS[number], "message": "Updated successfully"})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "ServiceNow Mock",
                    "incidents_loaded": len(INCIDENTS)})


if __name__ == "__main__":
    print(f"ServiceNow Mock API starting on http://localhost:{PORT}")
    print(f"Loaded {len(INCIDENTS)} incidents from {DATA_FILE}")
    app.run(port=PORT, debug=False)  # no reloader, so in-memory updates survive