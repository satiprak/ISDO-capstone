"""
ISDO Lab C2 — Check both mock APIs (Steps 3-6 of the lab, plus the update loop)

Start both shims first, each in its own terminal (from the project root):
    python mcp_server/snow_shim.py
    python mcp_server/jira_shim.py
Then, in a third terminal:
    python labs/c2/test_apis.py

Any update the test makes is set back to its original value at the end,
so the shims are left clean for Lab C3.
"""

import json
from urllib.parse import urlsplit

import requests

SNOW = "http://localhost:5001"
JIRA = "http://localhost:5002"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def get(url):
    parts = urlsplit(url)
    print(f"\n--- Testing GET {parts.path}{'?' + parts.query if parts.query else ''} ---")
    return requests.get(url, timeout=5)


def main():
    # Step 6 first: if the shims aren't running, nothing else can pass
    for name, base in [("ServiceNow", SNOW), ("Jira", JIRA)]:
        try:
            r = get(f"{base}/health")
        except requests.ConnectionError:
            print(f"  [FAIL] {name} shim is not running at {base}. Start it and re-run.")
            raise SystemExit(1)
        print("  " + json.dumps(r.json()))
        check(f"{name} /health", r.ok and r.json().get("status") == "ok")

    # Step 3 — all incidents
    r = get(f"{SNOW}/api/now/table/incident")
    check("All incidents returned", r.json()["total"] == 15, f"total={r.json()['total']}")

    # Step 4 — filters and single record
    r = get(f"{SNOW}/api/now/table/incident?priority=P1")
    p1 = r.json()["result"]
    print("  P1 incidents: " + ", ".join(i["number"] for i in p1))
    check("Filter ?priority=P1", len(p1) > 0 and all(i["priority"] == "P1" for i in p1), f"total={len(p1)}")
    r = get(f"{SNOW}/api/now/table/incident?category=Network")
    check("Filter ?category=Network", all(i["category"] == "Network" for i in r.json()["result"]),
          f"total={r.json()['total']}")
    r = get(f"{SNOW}/api/now/table/incident/INC0001001")
    check("Single incident INC0001001", r.ok and r.json()["result"]["number"] == "INC0001001")
    r = get(f"{SNOW}/api/now/table/incident/INC9999999")
    check("Unknown incident returns 404", r.status_code == 404)

    # PATCH loop — update, read back, restore (what the SLA Agent does in Lab C5)
    print("\n--- Testing PATCH /api/now/table/incident/INC0001002 ---")
    original = requests.get(f"{SNOW}/api/now/table/incident/INC0001002").json()["result"]["state"]
    requests.patch(f"{SNOW}/api/now/table/incident/INC0001002", json={"state": "Escalated"})
    after = requests.get(f"{SNOW}/api/now/table/incident/INC0001002").json()["result"]["state"]
    check("PATCH state -> Escalated, read back", after == "Escalated", f"{original} -> {after}")
    requests.patch(f"{SNOW}/api/now/table/incident/INC0001002", json={"state": original})

    # Step 5 — Jira
    r = get(f"{JIRA}/rest/agile/1.0/board/requests")
    check("All Jira requests returned", r.json()["total"] == 10, f"total={r.json()['total']}")
    r = get(f"{JIRA}/rest/agile/1.0/board/requests?request_type=Access Grant")
    check("Filter ?request_type=Access Grant", r.json()["total"] > 0, f"total={r.json()['total']}")
    r = get(f"{JIRA}/rest/api/2/issue/REQ-1002")
    fields = r.json().get("fields", {})
    print("  " + json.dumps(r.json()))
    check("Single issue has Jira-style nested fields", fields.get("status", {}).get("name") == "Open")

    print("\n--- Testing PUT /rest/api/2/issue/REQ-1002 (Jira 'fields' format) ---")
    requests.put(f"{JIRA}/rest/api/2/issue/REQ-1002", json={"fields": {"status": {"name": "In Progress"}}})
    status = requests.get(f"{JIRA}/rest/api/2/issue/REQ-1002").json()["fields"]["status"]["name"]
    check("PUT status -> In Progress, read back", status == "In Progress", status)
    requests.put(f"{JIRA}/rest/api/2/issue/REQ-1002", json={"fields": {"status": {"name": "Open"}}})

    passed = sum(results)
    print(f"\n{'=' * 60}\n{passed}/{len(results)} checks passed")
    print("Both mock APIs are ready for Lab C3." if passed == len(results) else "Fix the FAIL lines above.")


if __name__ == "__main__":
    main()
