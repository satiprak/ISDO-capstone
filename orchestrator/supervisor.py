"""
ISDO Lab C6 - LangGraph Orchestrator (Supervisor)
Routes a ticket through Triage -> Resolution -> SLA -> [HITL] -> Communication
using a LangGraph StateGraph with one shared TicketState.

Run from the project root:   python orchestrator/supervisor.py
Needs: Labs C1-C5 done (agents/*.py, chroma_db), ANTHROPIC_API_KEY in .env,
       ServiceNow mock running (.\\start_shims.ps1 or python mcp_server/snow_shim.py).
"""

import operator
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # safe box/arrow symbols on Windows

# Reuse the agents built in Labs C3-C5 (each loads .env and checks the API key)
import anthropic  # noqa: E402
from agents import resolution_agent, sla_agent, triage_agent  # noqa: E402

client = anthropic.Anthropic()
MODEL = triage_agent.MODEL  # same ISDO_MODEL setting as the agents

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
    # sla node
    sla_breach_risk: str
    sla_minutes_remaining: int
    escalation_required: bool
    escalation_team: str
    escalated: bool
    hitl_required: bool
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
    return {"kb_article": r["kb_article_used"], "kb_score": r["score"],
            "confidence": r["confidence"], "auto_resolve": r["auto_resolve"],
            "resolution_text": r["resolution_text"],
            "audit_log": audit("ResolutionAgent", "search_kb",
                               f"{r['kb_article_used']} {r['confidence']} ({r['score']:.0%}), "
                               f"auto_resolve={r['auto_resolve']}")}


def sla_node(state: TicketState) -> dict:
    header("SLA AGENT — checking deadline")
    priority = state["triage_priority"]
    s = sla_agent.get_sla_status(state["ticket_number"], state["sla_due"], priority)
    if "error" in s:
        return {"sla_breach_risk": "UNKNOWN", "escalation_required": False,
                "hitl_required": True,  # can't judge the SLA -> let a human look
                "audit_log": audit("SLAAgent", "get_sla_status", s["error"])}

    team = sla_agent.ESCALATION_TEAMS.get(state["triage_category"], sla_agent.DEFAULT_TEAM)
    # An auto-resolved ticket is being fixed now, so it is not escalated.
    escalate = s["requires_escalation"] and not state.get("auto_resolve", False)
    hitl = escalate and priority in sla_agent.HITL_PRIORITIES   # P1 CRITICAL/BREACHED
    print(f"  SLA Risk: {s['breach_risk']} | Minutes remaining: {s['minutes_remaining']} "
          f"| Target: {s['sla_target_minutes']} min ({priority})")

    out = {"sla_breach_risk": s["breach_risk"], "sla_minutes_remaining": s["minutes_remaining"],
           "escalation_required": escalate, "escalation_team": team,
           "hitl_required": hitl, "escalated": False}
    entries = audit("SLAAgent", "get_sla_status",
                    f"{s['breach_risk']}, {s['minutes_remaining']} min left, "
                    f"escalation_required={escalate}, hitl_required={hitl}")

    if escalate and not hitl:  # e.g. P2 CRITICAL/BREACHED: escalate without a human
        result = sla_agent.update_ticket(state["ticket_number"], "escalate", team,
                                         note=f"Auto-escalated: SLA {s['breach_risk']}")
        out["escalated"] = result["success"]
        entries += audit("SLAAgent", "update_ticket",
                         f"Auto-escalated to {team}" if result["success"]
                         else f"Escalation to {team} failed: {result['message']}")
    out["audit_log"] = entries
    return out


def hitl_node(state: TicketState) -> dict:
    header("HITL GATE — human approval required")
    team = state.get("escalation_team", sla_agent.DEFAULT_TEAM)
    approved = sla_agent.hitl_approve(
        state["ticket_number"], "Escalate ticket",
        f"Escalate to {team} (SLA {state.get('sla_breach_risk')}, "
        f"{state.get('sla_minutes_remaining')} min left)")
    out = {"hitl_approved": approved}
    entries = audit("HITLGate", "approval", "APPROVED by operator" if approved
                    else "REJECTED by operator - escalation cancelled")
    if approved:
        result = sla_agent.update_ticket(state["ticket_number"], "escalate", team,
                                         note="Escalation approved by human operator")
        out["escalated"] = result["success"]
        entries += audit("SLAAgent", "update_ticket",
                         f"Escalated to {team}" if result["success"]
                         else f"Escalation to {team} failed: {result['message']}")
    else:
        print("  Escalation cancelled.")
    out["audit_log"] = entries
    return out


COMM_PROMPT = """You are the ISDO Communication Agent for Zensar's IT Service Desk.
Write a short, friendly status update (max 120 words) to the person who raised the
ticket. Start with "Dear User," and mention the ticket number. Use only the facts
given - never invent steps, names, times or promises. Do not include any personal
data (names, emails, employee IDs). Plain text only, no markdown."""


def communication_node(state: TicketState) -> dict:
    header("COMMUNICATION AGENT")
    number = state["ticket_number"]
    group = state.get("triage_assignment_group", "Service-Desk")

    if state.get("auto_resolve"):
        status = "RESOLVED"
        facts = ("Outcome: self-service resolution from the knowledge base. Include these "
                 f"steps exactly:\n{state.get('resolution_text', '')}\n"
                 "Ask them to reply if the steps do not fix it.")
        sn = sla_agent.update_ticket(number, "update_state", new_state="Resolved")
    elif state.get("escalated"):
        status = "ESCALATED"
        who = "approved by the duty manager" if state.get("hitl_approved") else "automatically"
        facts = (f"Outcome: escalated {who} to the {state.get('escalation_team')} team "
                 "because of the service-level deadline. They are working on it urgently.")
        sn = {"success": True}  # already updated in ServiceNow by the SLA/HITL node
    else:
        status = "ASSIGNED"
        extra = (" Escalation was reviewed and declined for now."
                 if state.get("hitl_required") and not state.get("hitl_approved") else "")
        facts = f"Outcome: assigned to the {group} team, who will contact them.{extra}"
        sn = sla_agent.update_ticket(number, "add_note",
                                     note=f"Requester notified: ticket {status} to {group}")

    response = client.messages.create(
        model=MODEL, max_tokens=2000, output_config={"effort": "low"}, system=COMM_PROMPT,
        messages=[{"role": "user", "content":
                   f"Ticket: {number}\nIssue: {state['short_description']}\n{facts}"}])
    message = "".join(b.text for b in response.content if b.type == "text").strip()

    print(f"  USER MESSAGE:\n    " + message.replace("\n", "\n    "))
    print(f"\n  ✅ FINAL STATUS: {status}")
    return {"user_message": message, "final_status": status,
            "audit_log": audit("CommunicationAgent", "draft_message",
                               f"{status}; ServiceNow updated={sn['success']}")}

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
    print(f"Model: {MODEL}   |   Simulated now: {sla_agent.NOW:%Y-%m-%d %H:%M}")
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
    ]

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
                                             "get_sla_status", "approval", "draft_message"))
        print("\n" + "═" * 55)
        print(f"AUDIT LOG — {s['ticket_number']}  ({s['final_status']})")
        print("═" * 55)
        print(f"  Path: {path}")
        for e in s["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<18} {e['action']:<16} {e['detail']}")
