"""
ISDO Lab C6/C7/C8 - LangGraph Orchestrator (Supervisor)
Routes a ticket through Triage -> Resolution -> SLA -> [HITL] -> Communication
using a LangGraph StateGraph with one shared TicketState.

Lab C7: the HITL gate fires for three reasons -
  1. P1 ticket with SLA CRITICAL or BREACHED  (escalation needs approval)
  2. Resolution confidence LOW                (no clear KB fix, any priority)
  3. Access grant request                     (security-sensitive, any priority)

Lab C8: when the KB match is LOW, the Resolution node asks the Knowledge
Specialist agent over A2A (POST /tasks, then GET /tasks/{id}). If it answers with
MEDIUM/HIGH confidence the LOW trigger clears; if it is LOW too, or the A2A server
is not running, the ticket falls back to the HITL gate.

Run from the project root:   python orchestrator/supervisor.py
Needs: Labs C1-C5 done (agents/*.py, chroma_db), ANTHROPIC_API_KEY in .env,
       both mocks running (.\\start_shims.ps1): ServiceNow for INC tickets,
       Jira for REQ tickets. Optional (Lab C8): Knowledge Specialist on :8001
       (python -m uvicorn a2a.knowledge_specialist:app --port 8001).
Run one ticket only:  python orchestrator/supervisor.py INC0001009
"""

import operator
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

import requests
from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # safe box/arrow symbols on Windows

# Reuse the agents built in Labs C3-C5 (each loads .env and checks the API key)
import anthropic  # noqa: E402
from agents import resolution_agent, sla_agent, triage_agent  # noqa: E402

client = anthropic.Anthropic()
MODEL = triage_agent.MODEL  # same ISDO_MODEL setting as the agents
JIRA_URL = os.environ.get("JIRA_URL", "http://localhost:5002")
ACCESS_GRANT_TEAM = "Security-Ops"  # who actions an approved access grant
A2A_URL = os.environ.get("A2A_URL", "http://localhost:8001")  # Knowledge Specialist (Lab C8)
A2A_TIMEOUT = 120  # seconds - the specialist calls Claude before it answers

# -- SHARED STATE -------------------------------------------------------------

class TicketState(TypedDict, total=False):
    """Single shared memory for the graph. Every field is optional at start;
    each node writes only the fields it owns."""
    # input ticket
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    request_type: str          # Jira request type for REQ- tickets, e.g. "Access Grant"
    # triage node
    triage_category: str
    triage_priority: str
    triage_assignment_group: str
    pii_detected: bool
    # resolution node
    kb_article: str
    kb_score: float
    resolution_text: str
    auto_resolve: bool
    confidence: str
    a2a_used: bool             # Lab C8: Knowledge Specialist consulted over A2A
    a2a_task_id: str
    a2a_status: str            # "completed", "unavailable", "error: ..."
    # sla node
    sla_breach_risk: str
    sla_minutes_remaining: int
    escalation_required: bool
    escalation_team: str
    escalated: bool
    hitl_required: bool
    hitl_reason: str           # why the gate fired (one or more triggers, " | "-separated)
    hitl_triggers: list        # machine-readable: "sla_escalation", "low_confidence", "access_grant"
    # hitl node
    hitl_approved: bool
    # communication node
    user_message: str
    final_status: str
    # every node appends; operator.add merges the lists instead of overwriting
    audit_log: Annotated[list, operator.add]


def audit(agent: str, action: str, detail: str) -> list:
    """One audit entry (Lab C9 will persist these to JSONL)."""
    print(f"  [AUDIT] {agent}: {action}")
    return [{"timestamp": datetime.now().isoformat(timespec="seconds"),
             "agent": agent, "action": action, "detail": detail}]


def header(title: str):
    print(f"\n▶ {title}")


def is_jira(ticket_number: str) -> bool:
    return ticket_number.upper().startswith("REQ-")


def lookup_request_type(ticket_number: str) -> str:
    """For REQ- tickets with no request_type given, ask the Jira mock (Lab C2)."""
    try:
        r = requests.get(f"{JIRA_URL}/rest/api/2/issue/{ticket_number}", timeout=5)
        if r.ok:
            return r.json()["fields"]["issuetype"]["name"] or ""
    except requests.RequestException:
        print(f"  (Jira mock not reachable at {JIRA_URL} - request_type unknown)")
    return ""


def update_record(ticket_number, action, team=None, note=None, new_state=None) -> dict:
    """Update the system of record: ServiceNow for INC tickets, Jira for REQ tickets."""
    if not is_jira(ticket_number):
        return sla_agent.update_ticket(ticket_number, action, team, note, new_state)

    fields = {"status": {"name": "Escalated" if action == "escalate" else new_state or "Open"}}
    if team:
        fields["assignee"] = {"displayName": team}
    if action == "add_note":
        fields = {"comment": note or ""}
    try:
        r = requests.put(f"{JIRA_URL}/rest/api/2/issue/{ticket_number}",
                         json={"fields": fields}, timeout=5)
    except requests.RequestException:
        print(f"  [Jira Mock] NOT REACHABLE at {JIRA_URL} - start jira_shim.py")
        return {"success": False, "message": "Jira mock not reachable; update not made"}
    if not r.ok:
        print(f"  [Jira Mock] FAILED ({r.status_code}): {r.text[:100]}")
        return {"success": False, "message": f"Jira returned {r.status_code}"}
    print(f"  [Jira Mock] {action.upper()} {ticket_number}: {fields}")
    return {"success": True}

def call_knowledge_specialist(state: TicketState) -> dict:
    """Lab C8 - A2A round-trip: submit a task, then fetch its result.
    Returns the specialist's result dict, or {"error": ...} if it could not be reached."""
    query = f"{state['short_description']}. {state['description']}"
    context = (f"Category: {state.get('triage_category')}, "
               f"Priority: {state.get('triage_priority')}")
    try:
        # 1) submit the task
        r = requests.post(f"{A2A_URL}/tasks", timeout=A2A_TIMEOUT, json={
            "query": query, "ticket_number": state["ticket_number"], "context": context})
        r.raise_for_status()
        task_id = r.json()["task_id"]
        print(f"  -> A2A task submitted: {task_id} ({r.json().get('status')})")
        # 2) fetch the result
        r = requests.get(f"{A2A_URL}/tasks/{task_id}", timeout=A2A_TIMEOUT)
        r.raise_for_status()
        task = r.json()
    except requests.ConnectionError:
        return {"error": f"Knowledge Specialist not running at {A2A_URL}", "status": "unavailable"}
    except requests.Timeout:
        return {"error": f"Knowledge Specialist timed out after {A2A_TIMEOUT}s", "status": "error: timeout"}
    except (requests.RequestException, KeyError, ValueError) as exc:
        return {"error": f"A2A call failed: {exc}", "status": "error: bad response"}

    if task.get("status") != "completed" or "result" not in task:
        return {"error": f"A2A task {task_id} not completed ({task.get('status')})",
                "status": f"error: {task.get('status')}"}
    return {"task_id": task_id, "status": "completed", **task["result"]}

# -- NODES ----------------------------------------------------------------------

def triage_node(state: TicketState) -> dict:
    header(f"TRIAGE AGENT — {state['ticket_number']}")
    c = triage_agent.triage_ticket(state["ticket_number"], state["short_description"],
                                   state["description"])
    if not c:  # agent failed - fall back to the values already on the ticket
        return {"triage_category": state["category"], "triage_priority": state["priority"],
                "triage_assignment_group": "Service-Desk", "pii_detected": False,
                "audit_log": audit("TriageAgent", "classify_ticket",
                                   "No classification returned - used ticket's own values")}
    return {"triage_category": c["category"], "triage_priority": c["priority"],
            "triage_assignment_group": c["assignment_group"], "pii_detected": c["pii_detected"],
            "audit_log": audit("TriageAgent", "classify_ticket",
                               f"{c['category']} / {c['priority']} -> {c['assignment_group']}, "
                               f"PII={c['pii_detected']}: {c['reasoning']}")}


def resolution_node(state: TicketState) -> dict:
    header("RESOLUTION AGENT — searching KB")
    r = resolution_agent.resolve_ticket(
        state["ticket_number"], state["short_description"], state["description"],
        state["triage_category"], state["triage_priority"])
    if not r:
        return {"kb_article": "none", "kb_score": 0.0, "confidence": "LOW",
                "auto_resolve": False, "resolution_text": "",
                "audit_log": audit("ResolutionAgent", "search_kb",
                                   "No resolution drafted - routed to a human")}
    out = {"kb_article": r["kb_article_used"], "kb_score": r["score"],
           "confidence": r["confidence"], "auto_resolve": r["auto_resolve"],
           "resolution_text": r["resolution_text"], "a2a_used": False,
           "audit_log": audit("ResolutionAgent", "search_kb",
                              f"{r['kb_article_used']} {r['confidence']} ({r['score']:.0%}), "
                              f"auto_resolve={r['auto_resolve']}")}
    if r["confidence"] == "LOW":
        out.update(a2a_fallback(state))
        out["audit_log"] = out["audit_log"] + out.pop("a2a_audit")
    return out


def a2a_fallback(state: TicketState) -> dict:
    """LOW KB match -> ask the Knowledge Specialist (A2A). Any failure keeps LOW,
    so the SLA node sends the ticket to the HITL gate."""
    header("A2A — calling Knowledge Specialist (LOW KB confidence)")
    res = call_knowledge_specialist(state)
    if "error" in res:
        print(f"  !! {res['error']} - keeping LOW confidence (HITL fallback)")
        return {"a2a_used": False, "a2a_status": res["status"],
                "a2a_audit": audit("KnowledgeSpecialist", "a2a_call",
                                   f"FAILED: {res['error']} - falling back to HITL")}

    confidence = res.get("confidence", "LOW")
    if res.get("escalate_to_l2"):
        confidence = "LOW"  # the specialist itself says a human must take it
    print(f"  -> A2A result: {res.get('best_match')}  {confidence} "
          f"({res.get('confidence_score', 0):.0%})  escalate_to_l2={res.get('escalate_to_l2')}")

    out = {"a2a_used": True, "a2a_task_id": res["task_id"], "a2a_status": "completed",
           "confidence": confidence, "kb_score": res.get("confidence_score", 0.0),
           "kb_article": f"{res.get('best_match')} (via A2A)",
           # The specialist writes for an L2 engineer, so it is never sent to the
           # requester as a self-service fix: auto_resolve stays False.
           "auto_resolve": False}
    entries = audit("KnowledgeSpecialist", "a2a_call",
                    f"Task {res['task_id']}: {res.get('best_match')} {confidence} "
                    f"({res.get('confidence_score', 0):.0%})")
    if confidence != "LOW" and res.get("resolution"):
        out["resolution_text"] = res["resolution"]
        note = f"Knowledge Specialist (A2A task {res['task_id']}) suggests:\n{res['resolution']}"
        result = update_record(state["ticket_number"], "add_note", note=note[:2000])
        entries += audit("ResolutionAgent", "update_ticket",
                         "A2A resolution added as work note for the engineer"
                         if result["success"] else f"Note failed: {result['message']}")
    out["a2a_audit"] = entries
    return out


def sla_node(state: TicketState) -> dict:
    """Check the SLA, then decide whether the HITL gate is needed (Lab C7: 3 triggers)."""
    header("SLA AGENT — checking deadline")
    number, priority = state["ticket_number"], state["triage_priority"]
    triggers, reasons, entries = [], [], []
    out = {"escalated": False}

    # -- Trigger 3: access grant (checked first - it changes what "resolution" means)
    request_type = state.get("request_type") or (lookup_request_type(number) if is_jira(number) else "")
    categories = {state.get("category"), state.get("triage_category")}
    access_grant = request_type == "Access Grant" and "Access" in categories
    out["request_type"] = request_type
    if access_grant:
        triggers.append("access_grant")
        reasons.append(f"ACCESS GRANT — '{state['short_description']}' requires security approval")
        # An access grant is approved or refused, never "fixed" from a KB article
        if state.get("auto_resolve"):
            out["auto_resolve"] = False
            entries += audit("SLAAgent", "override_auto_resolve",
                             "Access grant - KB auto-resolve disabled, approval needed")

    # -- SLA check (Lab C5 rules)
    s = sla_agent.get_sla_status(number, state["sla_due"], priority)
    if "error" in s:
        out.update(sla_breach_risk="UNKNOWN", escalation_required=False)
        triggers.append("sla_unknown")
        reasons.append(f"SLA UNKNOWN — {s['error']}")
        entries += audit("SLAAgent", "get_sla_status", s["error"])
        escalate = False
    else:
        auto = state.get("auto_resolve", False) and not access_grant
        escalate = s["requires_escalation"] and not auto  # auto-resolved tickets aren't escalated
        print(f"  SLA Risk: {s['breach_risk']} | Minutes remaining: {s['minutes_remaining']} "
              f"| Target: {s['sla_target_minutes']} min ({priority})")
        out.update(sla_breach_risk=s["breach_risk"], sla_minutes_remaining=s["minutes_remaining"],
                   escalation_required=escalate)
        entries += audit("SLAAgent", "get_sla_status",
                         f"{s['breach_risk']}, {s['minutes_remaining']} min left, "
                         f"escalation_required={escalate}")

    team = sla_agent.ESCALATION_TEAMS.get(state["triage_category"], sla_agent.DEFAULT_TEAM)
    out["escalation_team"] = team

    # -- Trigger 1: P1 CRITICAL/BREACHED escalation
    if escalate and priority in sla_agent.HITL_PRIORITIES:
        triggers.append("sla_escalation")
        reasons.append(f"P1 SLA {s['breach_risk']} — escalation to {team} "
                       f"({s['minutes_remaining']} min left)")

    # -- Trigger 2: LOW resolution confidence (any priority)
    if state.get("confidence") == "LOW":
        triggers.append("low_confidence")
        a2a = state.get("a2a_status")
        a2a_note = ("" if not a2a else " (Knowledge Specialist also LOW)" if a2a == "completed"
                    else f" (Knowledge Specialist {a2a})")
        reasons.append(f"LOW KB CONFIDENCE ({state.get('kb_score', 0):.0%}){a2a_note} — "
                       f"no clear fix, route to {team} for manual resolution")

    hitl = bool(triggers)
    out.update(hitl_required=hitl, hitl_triggers=triggers, hitl_reason=" | ".join(reasons))
    if hitl:
        entries += audit("SLAAgent", "hitl_check", f"HITL required: {out['hitl_reason']}")

    # P2 CRITICAL/BREACHED with no other trigger: escalate without a human (Lab C5 rule)
    if escalate and not hitl:
        result = update_record(number, "escalate", team,
                               note=f"Auto-escalated: SLA {s['breach_risk']}")
        out["escalated"] = result["success"]
        entries += audit("SLAAgent", "update_ticket",
                         f"Auto-escalated to {team}" if result["success"]
                         else f"Escalation to {team} failed: {result['message']}")
    out["audit_log"] = entries
    return out


def ask_human(state: TicketState) -> bool:
    """Show why the gate fired and wait for y/n. Anything else (or no keyboard) = reject."""
    print("  " + "WARNING " * 8)
    print(f"  Ticket:  {state['ticket_number']} | Priority: {state.get('triage_priority')}")
    for reason in state.get("hitl_reason", "").split(" | "):
        print(f"  Reason:  {reason}")
    print("  " + "WARNING " * 8)
    try:
        return input("  Approve action? [y/n]: ").strip().lower() == "y"
    except EOFError:
        return False


def hitl_node(state: TicketState) -> dict:
    header("HITL GATE — human approval required")
    number, triggers = state["ticket_number"], state.get("hitl_triggers", [])
    team = state.get("escalation_team", sla_agent.DEFAULT_TEAM)
    approved = ask_human(state)
    decision = "APPROVED" if approved else "REJECTED"
    print(f"  Decision: {decision}")
    out = {"hitl_approved": approved}
    entries = audit("HITLGate", "approval_decision",
                    f"{decision} by operator — {state.get('hitl_reason', '')}")

    if not approved:
        result = update_record(number, "add_note",
                               note=f"HITL rejected: {state.get('hitl_reason', '')}")
        entries += audit("HITLGate", "update_ticket", "Pending approval noted"
                         if result["success"] else f"Note failed: {result['message']}")
    elif "access_grant" in triggers:
        result = update_record(number, "update_state", ACCESS_GRANT_TEAM, new_state="Approved")
        entries += audit("HITLGate", "update_ticket",
                         f"Access grant approved -> {ACCESS_GRANT_TEAM} to provision"
                         if result["success"] else f"Update failed: {result['message']}")
    elif "sla_escalation" in triggers or "low_confidence" in triggers:
        why = "SLA escalation" if "sla_escalation" in triggers else "manual resolution (LOW KB match)"
        result = update_record(number, "escalate", team, note=f"Approved by operator: {why}")
        out["escalated"] = result["success"]
        entries += audit("SLAAgent", "update_ticket",
                         f"Escalated to {team} for {why}" if result["success"]
                         else f"Escalation to {team} failed: {result['message']}")
    out["audit_log"] = entries
    return out


COMM_PROMPT = """You are the ISDO Communication Agent for Zensar's IT Service Desk.
Write a short, friendly status update (max 120 words) to the person who raised the
ticket. Start with "Dear User," (or "Dear Requester," for a service request) and
mention the ticket number. Use only the facts given - never invent steps, names,
times or promises. Do not include any personal data (names, emails, employee IDs).
Plain text only, no markdown."""


def communication_node(state: TicketState) -> dict:
    header("COMMUNICATION AGENT")
    number = state["ticket_number"]
    group = state.get("triage_assignment_group", "Service-Desk")
    triggers = state.get("hitl_triggers", [])
    sn = {"success": True}  # SLA/HITL nodes already updated the record where needed

    if state.get("hitl_required") and not state.get("hitl_approved"):
        status = "PENDING_APPROVAL"
        facts = ("Outcome: the request needs further review and approval before we can act. "
                 "It is on hold pending approval; we will update them as soon as a decision "
                 "is made. Do not say it was rejected.")
    elif "access_grant" in triggers:
        status = "APPROVED"
        facts = (f"Outcome: their access grant request has been approved. The "
                 f"{ACCESS_GRANT_TEAM} team will now provision the access and confirm when done.")
    elif state.get("auto_resolve"):
        status = "RESOLVED"
        facts = ("Outcome: self-service resolution from the knowledge base. Include these "
                 f"steps exactly:\n{state.get('resolution_text', '')}\n"
                 "Ask them to reply if the steps do not fix it.")
        sn = update_record(number, "update_state", new_state="Resolved")
    elif state.get("escalated"):
        status = "ESCALATED"
        if "low_confidence" in triggers:
            facts = (f"Outcome: no standard fix was found, so a specialist in the "
                     f"{state.get('escalation_team')} team will investigate and contact them.")
        else:
            who = "approved by the duty manager" if state.get("hitl_approved") else "automatically"
            facts = (f"Outcome: escalated {who} to the {state.get('escalation_team')} team "
                     "because of the service-level deadline. They are working on it urgently.")
    else:
        status = "ASSIGNED"
        facts = f"Outcome: assigned to the {group} team, who will contact them."
        if state.get("a2a_used"):
            facts += (" Our knowledge specialist has already prepared a suggested fix for "
                      "the engineer, so they can start straight away.")
            sn = {"success": True}  # keep the A2A work note (the mock stores one note)
        else:
            sn = update_record(number, "add_note",
                               note=f"Requester notified: ticket {status} to {group}")

    response = client.messages.create(
        model=MODEL, max_tokens=2000, output_config={"effort": "low"}, system=COMM_PROMPT,
        messages=[{"role": "user", "content":
                   f"Ticket: {number}\nIssue: {state['short_description']}\n{facts}"}])
    message = "".join(b.text for b in response.content if b.type == "text").strip()

    print(f"  USER MESSAGE:\n    " + message.replace("\n", "\n    "))
    print(f"\n  ✅ FINAL STATUS: {status}")
    system = "Jira" if is_jira(number) else "ServiceNow"
    return {"user_message": message, "final_status": status,
            "audit_log": audit("CommunicationAgent", "draft_message",
                               f"{status}; {system} updated={sn['success']}")}

# -- GRAPH ----------------------------------------------------------------------

def route_after_sla(state: TicketState) -> str:
    return "hitl" if state.get("hitl_required") else "communication"


def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("triage", triage_node)
    graph.add_node("resolution", resolution_node)
    graph.add_node("sla", sla_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("communication", communication_node)

    graph.add_edge(START, "triage")
    graph.add_edge("triage", "resolution")
    graph.add_edge("resolution", "sla")
    graph.add_conditional_edges("sla", route_after_sla,
                                {"hitl": "hitl", "communication": "communication"})
    graph.add_edge("hitl", "communication")
    graph.add_edge("communication", END)
    return graph.compile()


def process_ticket(app, ticket: dict) -> TicketState:
    print("\n" + "═" * 55)
    print(f"PROCESSING TICKET: {ticket['ticket_number']}")
    print("═" * 55)
    return app.invoke({**ticket, "audit_log": []})

# -- RUN ------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Model: {MODEL}   |   Simulated now: {sla_agent.NOW:%Y-%m-%d %H:%M}   |   "
          f"A2A: {A2A_URL}")
    app = build_graph()

    test_tickets = [
        {   # P2 VPN -> Triage -> Resolution -> SLA -> Communication (auto-resolve)
            "ticket_number": "INC0001001",
            "short_description": "VPN not connecting after password change",
            "description": "User reports VPN client fails to connect after AD password "
                           "was reset. Error: authentication failed.",
            "category": "Network", "priority": "P2", "sla_due": "2024-01-15 14:00:00",
        },
        {   # P1 SAP -> Triage -> Resolution -> SLA -> HITL -> Communication (type 'y')
            # incidents.csv has 11:00 = exactly 50% of the 60-min P1 target (ON_TRACK,
            # no HITL). 10:40 makes it CRITICAL, as the lab intends (same as Lab C5).
            "ticket_number": "INC0001002",
            "short_description": "Cannot access ERP system - login error",
            "description": "Multiple users in Finance unable to login to SAP. "
                           "Error code: DBCON_FAIL. Started 09:00 today.",
            "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00",
        },
        {   # Lab C7 Step 4 - access grant (REQ-1002 in requests.csv) -> HITL regardless
            # of priority. request_type may be omitted: it is then read from the Jira mock.
            "ticket_number": "REQ-1002",
            "short_description": "VPN access for new contractor",
            "description": "Contractor needs VPN access. Email: contractor@client.com",
            "category": "Access", "priority": "P2", "sla_due": "2024-01-15 15:00:00",
            "request_type": "Access Grant",
        },
        {   # Lab C7 Step 3 / Lab C8 - LOW KB confidence: no KB article covers Webex.
            # Resolution -> A2A Knowledge Specialist. If it is LOW too (or not running)
            # -> HITL; if it answers MEDIUM/HIGH -> no LOW trigger, ASSIGNED with its fix.
            # Uses INC0001009's record in the ServiceNow mock so updates have a target.
            "ticket_number": "INC0001009",
            "short_description": "Cisco Webex not launching on MacBook M2 after Sonoma update",
            "description": "Cisco Webex app crashes on launch on a MacBook M2 since the "
                           "macOS Sonoma update. Reinstalling did not help.",
            "category": "Software", "priority": "P3", "sla_due": "2024-01-18 17:00:00",
        },
    ]

    # Optional: run only some tickets, e.g.  python orchestrator/supervisor.py INC0001009
    wanted = {a.upper() for a in sys.argv[1:]}
    if wanted:
        test_tickets = [t for t in test_tickets if t["ticket_number"].upper() in wanted]
        if not test_tickets:
            sys.exit(f"No test ticket matches {', '.join(sorted(wanted))}")

    results = []
    try:
        for t in test_tickets:
            results.append(process_ticket(app, t))
    except anthropic.AuthenticationError:
        sys.exit("\nAPI key rejected. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.NotFoundError:
        sys.exit(f"\nModel '{MODEL}' not found. Set ISDO_MODEL in .env (e.g. claude-opus-5-5).")
    except anthropic.APIConnectionError:
        sys.exit("\nCould not reach the Anthropic API. Check your network or proxy.")

    for s in results:
        path = " → ".join(e["agent"] for e in s["audit_log"]
                          if e["action"] in ("classify_ticket", "search_kb",
                                             "a2a_call", "get_sla_status", "approval_decision",
                                             "draft_message"))
        print("\n" + "═" * 55)
        print(f"AUDIT LOG — {s['ticket_number']}  ({s['final_status']})")
        if s.get("hitl_required"):
            print(f"  HITL: {'APPROVED' if s.get('hitl_approved') else 'REJECTED'} — {s['hitl_reason']}")
        print("═" * 55)
        print(f"  Path: {path}")
        for e in s["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<18} {e['action']:<22} {e['detail']}")