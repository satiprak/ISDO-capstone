"""
ISDO Lab C5 - SLA & Escalation Agent
Checks SLA breach risk, escalates CRITICAL/BREACHED P1/P2 tickets through the
Lab C2 ServiceNow mock, and pauses at a Human-in-the-Loop (HITL) gate before
any P1 escalation.

Run from the project root:   python agents/sla_agent.py
Needs: ANTHROPIC_API_KEY in .env (optional ISDO_MODEL), and the ServiceNow mock
       running (python mcp_server/snow_shim.py, or .\\start_shims.ps1).
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set. Add it to .env in the project root.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5")
SNOW_URL = os.environ.get("SNOW_URL", "http://localhost:5001")
MAX_TURNS = 6

# Note: the lab prompt asks for temperature=0.0, but the current Anthropic Python
# SDK no longer accepts temperature. The SLA maths, the escalation rule and the
# HITL gate are all enforced in code, so the outcome does not depend on sampling.

# -- SLA POLICY (deterministic) -----------------------------------------------

NOW = datetime(2024, 1, 15, 10, 30)   # simulated "now" for reproducible demos
SLA_MINUTES = {"P1": 60, "P2": 240, "P3": 480, "P4": 1440}
ESCALATE_RISKS = {"BREACHED", "CRITICAL"}
ESCALATE_PRIORITIES = {"P1", "P2"}     # P3/P4 are monitored, never auto-escalated
HITL_PRIORITIES = {"P1"}               # a human must approve every P1 escalation
ESCALATION_TEAMS = {
    "Network": "L2-Network-Ops",
    "Application": "L2-App-Support",
    "Server": "L2-Server-Ops",
    "Access": "L2-Security-Ops",
    "Security": "L2-Security-Ops",
}
DEFAULT_TEAM = "L2-Service-Desk"


def get_sla_status(ticket_number, sla_due, priority):
    """Minutes left vs the priority's SLA target -> breach risk."""
    try:
        due = datetime.strptime(sla_due, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": f"Invalid sla_due format (want YYYY-MM-DD HH:MM:SS): {sla_due}"}
    minutes = int((due - NOW).total_seconds() // 60)
    target = SLA_MINUTES[priority]

    if minutes < 0:
        risk, msg = "BREACHED", f"SLA breached {abs(minutes)} minutes ago"
    elif minutes < target * 0.2:
        risk, msg = "CRITICAL", f"Only {minutes} minutes remaining - breach imminent"
    elif minutes < target * 0.5:
        risk, msg = "AT_RISK", f"{minutes} minutes remaining - at risk"
    else:
        risk, msg = "ON_TRACK", f"{minutes} minutes remaining - on track"

    return {
        "ticket_number": ticket_number,
        "priority": priority,
        "sla_due": sla_due,
        "sla_target_minutes": target,
        "minutes_remaining": minutes,
        "breach_risk": risk,
        "status_message": msg,
        "requires_escalation": risk in ESCALATE_RISKS and priority in ESCALATE_PRIORITIES,
    }

# -- SERVICENOW UPDATE (real PATCH to the Lab C2 mock) -------------------------

def update_ticket(ticket_number, action, escalation_team=None, note=None, new_state=None):
    """PATCH the ticket on the ServiceNow mock (Lab C2)."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if action == "escalate":
        team = escalation_team or DEFAULT_TEAM
        body = {"state": "Escalated", "assignment_group": team,
                "work_notes": note or f"Escalated to {team} by ISDO SLA Agent"}
        label = f"ESCALATED {ticket_number} -> {team}"
    elif action == "add_note":
        body = {"work_notes": note or ""}
        label = f"NOTE ADDED to {ticket_number}"
    elif action == "update_state":
        body = {"state": new_state or ""}
        label = f"STATE CHANGED {ticket_number} -> {new_state}"
    else:
        return {"success": False, "message": f"Unknown action: {action}"}

    try:
        r = requests.patch(f"{SNOW_URL}/api/now/table/incident/{ticket_number}",
                           json=body, timeout=5)
    except requests.ConnectionError:
        print(f"  [ServiceNow Mock] NOT REACHABLE at {SNOW_URL} - start snow_shim.py")
        return {"success": False, "message": "ServiceNow mock not reachable; update not made"}
    if not r.ok:
        print(f"  [ServiceNow Mock] FAILED ({r.status_code}): {r.text[:100]}")
        return {"success": False, "message": f"ServiceNow returned {r.status_code}"}

    print(f"  [ServiceNow Mock] {label}")
    return {"success": True, "ticket_number": ticket_number, "action": action,
            "timestamp": stamp, "record": r.json().get("result", {})}

# -- HITL GATE ----------------------------------------------------------------

def hitl_approve(ticket_number, action, detail):
    """Pause for a human decision. Anything other than 'y' is a rejection."""
    print("\n  " + "!!! " * 10)
    print("  HITL APPROVAL REQUIRED")
    print(f"  Ticket:  {ticket_number}")
    print(f"  Action:  {action}")
    print(f"  Detail:  {detail}")
    print("  " + "!!! " * 10)
    try:
        decision = input("  Approve escalation? [y/n]: ").strip().lower()
    except EOFError:  # no human at the keyboard (e.g. run from a script)
        decision = ""
    return decision == "y"

# -- TOOL DEFINITIONS ---------------------------------------------------------

tools = [
    {
        "name": "get_sla_status",
        "description": "Check a ticket's SLA: minutes remaining vs its priority's target, "
                       "breach risk (BREACHED/CRITICAL/AT_RISK/ON_TRACK) and whether it "
                       "requires escalation.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "sla_due": {"type": "string", "description": "YYYY-MM-DD HH:MM:SS"},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
            },
            "required": ["ticket_number", "sla_due", "priority"],
            "additionalProperties": False,
        },
    },
    {
        "name": "update_ticket",
        "description": "Update the ticket in ServiceNow: escalate, add_note or update_state.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "action": {"type": "string", "enum": ["escalate", "add_note", "update_state"]},
                "escalation_team": {"type": "string",
                                    "description": "Required for escalate"},
                "note": {"type": "string", "description": "Work note text"},
                "new_state": {"type": "string",
                              "description": "e.g. In Progress (use escalate to escalate)"},
            },
            "required": ["ticket_number", "action"],
            "additionalProperties": False,
        },
    },
]

# -- SLA AGENT ----------------------------------------------------------------

SYSTEM_PROMPT = """You are the ISDO SLA & Escalation Agent for Zensar's IT Service Desk.

For each ticket:
1. Call get_sla_status once with the ticket's number, SLA due time and priority.
2. If requires_escalation is true, call update_ticket with action "escalate" and the
   escalation team given in the ticket message, plus a short work note explaining the
   SLA risk. Escalate at most once.
3. If requires_escalation is false, do not escalate. You may add one short work note
   for AT_RISK tickets.
4. If an escalation is rejected by the human approver, do not retry it - add a work
   note that escalation was declined and stop.
Finish with one short summary line."""


def is_escalation(tool_input):
    return tool_input["action"] == "escalate" or (
        tool_input["action"] == "update_state"
        and "escalat" in tool_input.get("new_state", "").lower())


def monitor_ticket(ticket_number, short_description, category, priority, sla_due):
    """Run SLA monitoring for one ticket. Returns a summary dict."""
    team = ESCALATION_TEAMS.get(category, DEFAULT_TEAM)
    status = get_sla_status(ticket_number, sla_due, priority)  # authoritative, for the guardrail
    print(f"\n{'=' * 55}")
    print(f"SLA Check: {ticket_number} | {priority} | Category: {category}")
    print(f"{'=' * 55}")
    print(f"Issue: {short_description}  (SLA due {sla_due})")

    summary = {"ticket_number": ticket_number, "priority": priority,
               "breach_risk": status.get("breach_risk"), "escalation_team": team,
               "escalated": False, "decision": "NOT_REQUIRED"}
    if status.get("requires_escalation"):
        summary["decision"] = "PENDING"

    messages = [{
        "role": "user",
        "content": (f"Monitor SLA for this ticket and escalate if needed:\n\n"
                    f"Ticket: {ticket_number}\nDescription: {short_description}\n"
                    f"Category: {category}\nPriority: {priority}\nSLA Due: {sla_due}\n"
                    f"Escalation team for this category: {team}"),
    }]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            output_config={"effort": "low"},
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        if response.stop_reason != "tool_use":
            if response.stop_reason != "end_turn":
                print(f"  !! Stopped early: stop_reason = {response.stop_reason}")
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = run_tool(block.name, dict(block.input), status, priority, team, summary)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"  !! Gave up after {MAX_TURNS} turns.")

    if summary["decision"] == "PENDING":
        print("  !! Escalation was required but the agent did not escalate - flag for a human.")
        summary["decision"] = "MISSED"
    print(f"  Decision: {summary['decision']}")
    return summary


def run_tool(name, inp, status, priority, team, summary):
    """Execute one tool call, applying the escalation guardrail and HITL gate."""
    if name == "get_sla_status":
        result = get_sla_status(inp["ticket_number"], inp["sla_due"], inp["priority"])
        print(f"  -> Risk Level: {result.get('breach_risk', 'ERROR')}")
        print(f"  -> Status:     {result.get('status_message', result.get('error'))}")
        return result

    if name != "update_ticket":
        return {"error": f"Unknown tool: {name}"}

    if inp["ticket_number"] != summary["ticket_number"]:
        return {"success": False, "message": "Only the ticket being monitored may be updated"}

    if is_escalation(inp):
        if not status.get("requires_escalation"):
            print(f"  -> Guardrail: escalation blocked ({status['breach_risk']}, {priority})")
            return {"success": False,
                    "message": "Blocked by policy: only CRITICAL/BREACHED P1/P2 tickets escalate"}
        inp["action"], inp["escalation_team"] = "escalate", inp.get("escalation_team") or team
        if priority in HITL_PRIORITIES:
            approved = hitl_approve(inp["ticket_number"], "Escalate ticket",
                                    f"Escalate to {inp['escalation_team']}")
            if not approved:
                print("  Escalation cancelled by approver (logged).")
                summary["decision"] = "REJECTED"
                return {"success": False, "message": "Escalation rejected by human approver"}
            summary["decision"] = "APPROVED"
        else:
            summary["decision"] = "AUTO_ESCALATED"
        result = update_ticket(**inp)
        summary["escalated"] = result["success"]
        if not result["success"]:
            summary["decision"] += " (update failed)"
        return result

    return update_ticket(**inp)

# -- RUN SLA MONITORING -------------------------------------------------------

if __name__ == "__main__":
    print(f"Model: {MODEL}   |   Simulated now: {NOW:%Y-%m-%d %H:%M}   |   ServiceNow: {SNOW_URL}")

    # (number, description, category, priority, sla_due)
    test_tickets = [
        # P1 CRITICAL -> HITL prompt: type 'y'.
        # incidents.csv has 11:00, which is exactly 50% of the 60-min target (ON_TRACK
        # by the Step 1 rules), so 10:40 is used to show the CRITICAL case the lab intends.
        ("INC0001002", "Cannot access ERP - SAP login failure", "Application", "P1",
         "2024-01-15 10:40:00"),
        # P1 already BREACHED -> HITL prompt again: type 'n'
        ("INC0001010", "Exchange server high CPU alert", "Server", "P1",
         "2024-01-15 09:30:00"),
        # P2, 210 of 240 min left -> ON_TRACK, no escalation.
        # Step 5: change to "2024-01-15 10:00:00" -> BREACHED -> auto-escalates (no HITL for P2)
        ("INC0001001", "VPN not connecting after password change", "Network", "P2",
         "2024-01-15 14:00:00"),
        # P3 ON_TRACK -> monitored only
        ("INC0001003", "Laptop running very slowly", "Hardware", "P3",
         "2024-01-17 09:00:00"),
    ]

    results = []
    try:
        for ticket in test_tickets:
            results.append(monitor_ticket(*ticket))
    except anthropic.AuthenticationError:
        sys.exit("\nAPI key rejected. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.NotFoundError:
        sys.exit(f"\nModel '{MODEL}' not found. Set ISDO_MODEL in .env (e.g. claude-opus-5-5).")
    except anthropic.APIConnectionError:
        sys.exit("\nCould not reach the Anthropic API. Check your network or proxy.")

    print(f"\n{'=' * 55}")
    print("SLA SUMMARY")
    print(f"{'=' * 55}")
    for s in results:
        print(f"  {s['ticket_number']:<12} {s['priority']}  {s['breach_risk']:<9} "
              f"{s['decision']:<16} {s['escalation_team'] if s['escalated'] else ''}")
