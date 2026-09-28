"""
ISDO Lab C3 - Triage Agent
Reads a ticket and assigns: category, priority, assignment group, and PII flag.
Uses the Anthropic SDK with tool calling and an agentic (ReAct) loop.

Run from the project root:   python agents/triage_agent.py
Needs ANTHROPIC_API_KEY in .env (project root). Optional: ISDO_MODEL in .env.
"""

import csv
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCIDENTS_CSV = PROJECT_ROOT / "data" / "incidents.csv"

load_dotenv(PROJECT_ROOT / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set. Add it to .env in the project root.")

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# The model is swappable (platform-mandate lesson): set ISDO_MODEL in .env to change it.
MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5")
MAX_TURNS = 5  # safety stop for the agentic loop

# Note on temperature: the lab prompt asks for temperature=0.0, but the current
# Anthropic Python SDK no longer accepts temperature (it raises a TypeError), and
# newer models reject it. Consistency comes from strict tool use (the API enforces
# the schema) plus explicit rules in the system prompt.

# -- TOOL DEFINITIONS ---------------------------------------------------------

tools = [
    {
        "name": "classify_ticket",
        "description": (
            "Record the triage decision for an IT support ticket: category, priority, "
            "assignment group, PII flag and a one-sentence reason. Call exactly once per ticket."
        ),
        "strict": True,  # the API guarantees the input matches this schema exactly
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["Network", "Application", "Hardware", "Access",
                             "Email", "Server", "Software"],
                    "description": "The ticket category",
                },
                "priority": {
                    "type": "string",
                    "enum": ["P1", "P2", "P3", "P4"],
                    "description": "P1=critical/many users, P2=high/one team, "
                                   "P3=medium/single user, P4=low/request",
                },
                "assignment_group": {
                    "type": "string",
                    "enum": ["Network-Ops", "App-Support", "Desktop-Support", "Email-Support",
                             "Service-Desk", "Security-Ops", "Server-Ops", "DBA-Team"],
                    "description": "Team that should receive the ticket",
                },
                "pii_detected": {
                    "type": "boolean",
                    "description": "True if the text contains a person's name, email address, "
                                   "employee ID, phone number or IP address",
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence explaining the classification decision",
                },
            },
            "required": ["category", "priority", "assignment_group", "pii_detected", "reasoning"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_open_tickets",
        "description": "Count currently open incidents by category (current queue load).",
        "strict": True,
        # No parameters: the tool always reads data/incidents.csv. Letting the model
        # choose a file path would let it read any file on the machine.
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]

# -- TOOL IMPLEMENTATIONS -----------------------------------------------------

def get_open_tickets():
    """Read incidents.csv and return {category: open_count}."""
    counts = {}
    try:
        with open(INCIDENTS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if None in row:  # malformed row (unquoted comma) - skip it
                    continue
                if row["state"] == "Open":
                    counts[row["category"]] = counts.get(row["category"], 0) + 1
    except FileNotFoundError:
        return {"error": f"File not found: {INCIDENTS_CSV}"}
    return counts


def handle_tool_call(tool_name, tool_input):
    """Route a tool call from the model to its implementation."""
    if tool_name == "get_open_tickets":
        return get_open_tickets()
    if tool_name == "classify_ticket":
        return {"status": "recorded", **tool_input}  # the classification is the output
    return {"error": f"Unknown tool: {tool_name}"}

# -- TRIAGE AGENT -------------------------------------------------------------

SYSTEM_PROMPT = """You are the ISDO Triage Agent for Zensar's IT Service Desk.

For every ticket you receive, call the classify_ticket tool exactly once with your decision.
You may call get_open_tickets first if queue load helps you decide, but it is optional.
After classify_ticket returns, reply with one short confirmation line and stop.

Priority rules (apply strictly, in this order):
- P1: service down, many users or a whole site affected, security breach, or a user
      fully blocked from all corporate systems
- P2: significant impact on one team, department or function, or a single user
      blocked from critical work with no workaround
- P3: single user impacted, workaround exists
- P4: service request (new software, access, equipment) with no outage

Assignment groups:
- Network-Ops: VPN, Wi-Fi, switches, connectivity
- App-Support: business applications (ERP/SAP, CRM, SharePoint)
- Desktop-Support: laptops, workstations, printers, software installs
- Email-Support: Outlook, mobile mail, mailbox issues
- Service-Desk: password resets, account unlocks, onboarding
- Security-Ops: MFA, suspected breaches, security incidents
- Server-Ops: servers and infrastructure alerts
- DBA-Team: databases and backups

PII: set pii_detected to true if the text contains a real person's name, an email
address, an employee ID, a phone number or an IP address. Placeholders such as
[REDACTED] or [REDACTED NAME] are not PII by themselves.

Be consistent: the same ticket text must always get the same classification."""


def triage_ticket(ticket_number, short_description, description):
    """Run the triage agent on one ticket. Returns the classification dict (or None)."""
    print(f"\n{'=' * 55}")
    print(f"Triaging: {ticket_number}")
    print(f"{'=' * 55}")
    print(f"Description: {short_description}")

    messages = [{
        "role": "user",
        "content": (f"Please triage this ticket:\n\nTicket: {ticket_number}\n"
                    f"Summary: {short_description}\nDetails: {description}"),
    }]
    classification = None

    # Agentic loop: Reason -> Act (tool call) -> Observe (tool result) -> Reason ...
    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,  # room for the model's thinking plus the tool call
            output_config={"effort": "low"},  # triage is a simple decision
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            # Keep the full assistant turn (including any thinking blocks) in history
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                print(f"  -> Tool called: {block.name}")
                result = handle_tool_call(block.name, block.input)
                if block.name == "classify_ticket":
                    classification = dict(block.input)
                    print(f"  -> Category:    {result['category']}")
                    print(f"  -> Priority:    {result['priority']}")
                    print(f"  -> Assign To:   {result['assignment_group']}")
                    print(f"  -> PII Found:   {result['pii_detected']}")
                    print(f"  -> Reason:      {result['reasoning']}")
                elif block.name == "get_open_tickets":
                    print(f"  -> Queue load:  {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        if response.stop_reason == "end_turn":
            if classification is None:
                print("  !! Agent finished without calling classify_ticket.")
            break

        # max_tokens, refusal, etc. - stop instead of looping forever
        print(f"  !! Stopped early: stop_reason = {response.stop_reason}")
        break
    else:
        print(f"  !! Gave up after {MAX_TURNS} turns.")

    return classification

# -- RUN ON SAMPLE TICKETS ----------------------------------------------------

if __name__ == "__main__":
    print(f"Model: {MODEL}")

    # 5 test tickets from incidents.csv (plus one Jira request that contains PII)
    test_tickets = [
        ("INC0001001", "VPN not connecting after password change",
         "User reports VPN client fails to connect after AD password was reset. "
         "Error: authentication failed."),
        ("INC0001002", "Cannot access ERP system - login error",
         "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. "
         "Started 09:00 today."),
        ("INC0001008", "Network switch down - Building C",
         "Network switch in Building C server room unresponsive. 40 users in Building C affected."),
        ("INC0001006", "Password reset request",
         "User locked out of AD account after 5 failed attempts. Needs immediate reset."),
        ("REQ-1002", "VPN access for new contractor joining project Phoenix",
         "New contractor [REDACTED NAME] emp-id ZEN-9823 joining next Monday. "
         "Email: contractor@client.com"),
        # Step 5 - uncomment to test your own ticket, then try "Multiple users in Sales ..."
        # ("INC0001016", "Salesforce CRM access issue",
        #  "User cannot access Salesforce CRM from company laptop since this morning."),
    ]

    results = {}
    try:
        for number, short_desc, desc in test_tickets:
            results[number] = triage_ticket(number, short_desc, desc)
    except anthropic.AuthenticationError:
        sys.exit("\nAPI key rejected. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.NotFoundError:
        sys.exit(f"\nModel '{MODEL}' not found. Set ISDO_MODEL in .env (e.g. claude-opus-5-5).")
    except anthropic.APIConnectionError:
        sys.exit("\nCould not reach the Anthropic API. Check your network or proxy.")

    print(f"\n{'=' * 55}")
    print("TRIAGE SUMMARY")
    print(f"{'=' * 55}")
    for number, c in results.items():
        if c:
            print(f"  {number:<12} {c['category']:<12} {c['priority']}  "
                  f"{c['assignment_group']:<16} PII={c['pii_detected']}")
        else:
            print(f"  {number:<12} (no classification)")

    print(f"\n{'=' * 55}")
    print("OPEN TICKET COUNTS BY CATEGORY")
    print(f"{'=' * 55}")
    for cat, count in sorted(get_open_tickets().items()):
        print(f"  {cat:<20} {count} open")
