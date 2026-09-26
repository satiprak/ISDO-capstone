"""
ISDO Lab C2 - Mock Jira Service Management REST API (Flask)

  GET /rest/agile/1.0/board/requests   all service requests
  GET /rest/api/2/issue/<key>          one request, Jira-style nested 'fields'
  GET /health                          service status

Run from the project root:  python mcp_server/jira_shim.py   (port 5002)
"""

import csv
from pathlib import Path

from flask import Flask, jsonify

PORT = 5002
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "requests.csv"

app = Flask(__name__)


def load_requests() -> dict:
    """Read the CSV once at startup into {key: row}."""
    requests_data = {}
    if not DATA_FILE.exists():
        print(f"Warning: {DATA_FILE} not found - starting with no requests.")
        return requests_data
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            if None in row:  # more values than headers, usually an unquoted comma
                print(f"Warning: skipped line {line} of {DATA_FILE.name} (unquoted comma?)")
                continue
            requests_data[row["key"]] = row
    return requests_data


REQUESTS = load_requests()  # in-memory store; resets when the server restarts


def to_jira(row: dict) -> dict:
    """Flat CSV row -> the nested shape the real Jira REST API returns."""
    return {
        "key": row["key"],
        "fields": {
            "summary": row["summary"],
            "issuetype": {"name": row["request_type"]},
            "priority": {"name": row["priority"]},
            "assignee": {"displayName": row["assignee"]},
            "status": {"name": row["status"]},
            "customfield_sla": row["sla"],
        },
    }


@app.get("/rest/agile/1.0/board/requests")
def list_requests():
    results = list(REQUESTS.values())
    return jsonify({"issues": results, "total": len(results)})


@app.get("/rest/api/2/issue/<key>")
def get_request(key):
    if key not in REQUESTS:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    return jsonify(to_jira(REQUESTS[key]))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "Jira Mock",
                    "requests_loaded": len(REQUESTS)})


if __name__ == "__main__":
    print(f"Jira Mock API starting on http://localhost:{PORT}")
    print(f"Loaded {len(REQUESTS)} requests from {DATA_FILE}")
    app.run(port=PORT, debug=False)