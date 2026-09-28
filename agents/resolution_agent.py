"""
ISDO Lab C4 - Resolution / KB Agent
Searches the Lab C1 ChromaDB knowledge base for matching articles and drafts a
resolution. HIGH confidence on a non-P1 ticket auto-resolves; anything else is
flagged for Human-in-the-Loop (HITL) review.

Run from the project root:   python agents/resolution_agent.py
Needs: Lab C1 run first (chroma_db/ with collection 'isdo_kb'),
       ANTHROPIC_API_KEY in .env (project root). Optional: ISDO_MODEL in .env.
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "data" / "kb"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "isdo_kb"

load_dotenv(PROJECT_ROOT / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set. Add it to .env in the project root.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5")
MAX_TURNS = 5  # safety stop for the agentic loop

# Note: the lab prompt asks for temperature=0.0, but the current Anthropic Python
# SDK no longer accepts temperature. Consistency comes from strict tool schemas,
# and the two decisions that matter (confidence, auto_resolve) are made in code.

# -- GUARDRAIL POLICY (deterministic - never left to the model) --------------

HIGH_THRESHOLD = 0.60    # score > 0.60  -> HIGH
MEDIUM_THRESHOLD = 0.35  # score > 0.35  -> MEDIUM, otherwise LOW
# Priorities allowed to auto-resolve on HIGH confidence. P1 always needs a human.
# (The lab doc says "P3/P4" in Step 3 but expects the P2 VPN ticket to auto-resolve;
#  change this set if your instructor wants the stricter rule.)
AUTO_RESOLVE_PRIORITIES = {"P2", "P3", "P4"}


def confidence_level(score: float) -> str:
    if score > HIGH_THRESHOLD:
        return "HIGH"
    if score > MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def auto_resolve_allowed(confidence: str, priority: str) -> bool:
    return confidence == "HIGH" and priority in AUTO_RESOLVE_PRIORITIES

# -- CONNECT TO THE LAB C1 KNOWLEDGE BASE -------------------------------------

def load_kb():
    """Open the persistent collection built by labs/c1/kb_setup.py."""
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        kb = db.get_collection(COLLECTION_NAME)
    except Exception:
        sys.exit(f"Collection '{COLLECTION_NAME}' not found in {CHROMA_DIR}.\n"
                 "Run Lab C1 first:  python labs/c1/kb_setup.py")
    if kb.count() == 0:
        sys.exit("The KB collection is empty. Re-run:  python labs/c1/kb_setup.py")
    print(f"KB connected: {kb.count()} chunks in '{COLLECTION_NAME}' ({CHROMA_DIR})")
    return kb


KB = load_kb()

# -- TOOL DEFINITIONS ---------------------------------------------------------

tools = [
    {
        "name": "search_kb",
        "description": "Search the knowledge base with the ticket text. Returns the top 2 "
                       "matching KB articles (full text) with confidence scores.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "The ticket's summary and details, used as-is"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "draft_resolution",
        "description": "Record the resolution for the ticket, drawn from the KB article.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "resolution_text": {
                    "type": "string",
                    "description": "3-4 numbered steps taken from the KB article, written for "
                                   "the requester. If no article fits, say the ticket is being "
                                   "escalated to L2.",
                },
                "auto_resolve": {"type": "boolean",
                                 "description": "Your view: can this be sent without a human?"},
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "kb_article_used": {"type": "string",
                                    "description": "Exact article file name from search_kb, "
                                                   "or 'none'"},
            },
            "required": ["ticket_number", "resolution_text", "auto_resolve",
                         "confidence", "kb_article_used"],
            "additionalProperties": False,
        },
    },
]

# -- TOOL IMPLEMENTATION ------------------------------------------------------

def search_kb(query: str) -> dict:
    """Return the top 2 articles. A ticket's best-matching chunk is often 'Symptoms',
    not 'Resolution Steps', so rank by article and return each article's full text."""
    raw = KB.query(query_texts=[query], n_results=min(10, KB.count()),
                   include=["metadatas", "distances"])
    best = {}  # article -> smallest cosine distance
    for meta, distance in zip(raw["metadatas"][0], raw["distances"][0]):
        article = meta.get("article", "unknown")
        best[article] = min(distance, best.get(article, distance))

    articles = []
    for article, distance in sorted(best.items(), key=lambda kv: kv[1])[:2]:
        path = KB_DIR / article
        articles.append({
            "article": article,
            "confidence_score": round(max(0.0, 1.0 - distance), 2),  # 1 - cosine distance
            "content": path.read_text(encoding="utf-8") if path.exists() else "",
        })
    return {"query": query, "articles": articles}

# -- RESOLUTION AGENT ---------------------------------------------------------

SYSTEM_PROMPT = f"""You are the ISDO Resolution Agent for Zensar's IT Service Desk.

For each ticket:
1. Call search_kb exactly ONCE, using the ticket's summary and details as the query.
   Do not rephrase or search again - a weak match is a valid result.
2. Call draft_resolution exactly once.

Confidence comes from the best article's confidence_score:
- HIGH   if score > {HIGH_THRESHOLD}  (article clearly covers the issue)
- MEDIUM if score > {MEDIUM_THRESHOLD}  (partial match - a human should review)
- LOW    otherwise (no clear match - escalate to L2)

auto_resolve is true ONLY for HIGH confidence on a P2, P3 or P4 ticket. P1 is never
auto-resolved.

resolution_text: 3-4 numbered, specific steps copied from the article's Resolution
Steps, written for the requester. Never invent steps that are not in the article.
If confidence is LOW, set kb_article_used to 'none' and say the ticket is being
escalated to L2 instead of giving steps.

After draft_resolution returns, reply with one short line and stop."""


def resolve_ticket(ticket_number, short_description, description, category, priority):
    """Run the agent on one ticket (e.g. the output of the Lab C3 Triage Agent).
    Returns the final resolution dict with the guardrail-enforced decision, or None."""
    print(f"\n{'=' * 55}")
    print(f"Resolving: {ticket_number} | Category: {category} | Priority: {priority}")
    print(f"{'=' * 55}")
    print(f"Issue: {short_description}")

    messages = [{
        "role": "user",
        "content": (f"Find a resolution for this ticket:\n\nTicket: {ticket_number}\n"
                    f"Category: {category}\nPriority: {priority}\n"
                    f"Summary: {short_description}\nDetails: {description}"),
    }]
    scores = {}       # article -> score, from the search the agent actually ran
    resolution = None

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

            if block.name == "search_kb":
                result = search_kb(block.input["query"])
                print(f"  -> KB search: '{block.input['query'][:70]}'")
                for art in result["articles"]:
                    scores[art["article"]] = art["confidence_score"]
                    print(f"       [{art['confidence_score']:.0%}] {art['article']}")

            elif block.name == "draft_resolution":
                resolution = enforce_guardrail(dict(block.input), scores, priority)
                result = {"status": "recorded", "final_decision": {
                    k: resolution[k] for k in ("confidence", "auto_resolve")}}
                print_resolution(resolution)

            else:
                result = {"error": f"Unknown tool: {block.name}"}

            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"  !! Gave up after {MAX_TURNS} turns.")

    if resolution is None:
        print("\n  WARNING HITL FLAG: no resolution drafted - route to a human.")
    return resolution


def enforce_guardrail(draft, scores, priority):
    """Replace the model's confidence/auto_resolve with the policy decision."""
    if draft["kb_article_used"] in scores:
        score = scores[draft["kb_article_used"]]
        confidence = confidence_level(score)
    else:  # 'none' or an article that search_kb never returned -> never trusted
        score = max(scores.values(), default=0.0)  # best search score, for display
        confidence = "LOW"
    auto = auto_resolve_allowed(confidence, priority)
    draft["adjusted"] = (draft["confidence"], draft["auto_resolve"]) != (confidence, auto)
    draft.update(score=score, confidence=confidence, auto_resolve=auto, priority=priority)
    if confidence == "LOW":
        reason = "LOW confidence - no clear KB match, escalate to L2"
    elif confidence == "MEDIUM":
        reason = "MEDIUM confidence - partial KB match, human review required"
    elif not auto:
        reason = f"{priority} ticket - critical incidents always need a human"
    else:
        reason = ""
    draft["hitl_reason"] = reason
    return draft


def print_resolution(r):
    note = "  (adjusted by guardrail)" if r["adjusted"] else ""
    print(f"\n  -> Confidence: {r['confidence']} ({r['score']:.0%})  |  "
          f"Auto-resolve: {r['auto_resolve']}{note}")
    print(f"  -> KB Article: {r['kb_article_used']}")
    print("\n  RESOLUTION DRAFT:")
    for line in r["resolution_text"].strip().splitlines():
        print(f"    {line}")
    if not r["auto_resolve"]:
        print(f"\n  WARNING HITL FLAG: {r['hitl_reason']}. Human review required before sending.")

# -- RUN ON SAMPLE TICKETS ----------------------------------------------------

if __name__ == "__main__":
    print(f"Model: {MODEL}")

    # (number, summary, details, category, priority) - priorities from incidents.csv
    test_tickets = [
        ("INC0001001", "VPN not connecting after password change",
         "User reports VPN client fails to connect after AD password was reset. "
         "Error: authentication failed.", "Network", "P2"),
        ("INC0001006", "Password reset request",
         "User locked out of AD account after 5 failed attempts.", "Access", "P2"),
        ("INC0001002", "Cannot access ERP system - login error",
         "Multiple Finance users unable to login to SAP. Error: DBCON_FAIL.", "Application", "P1"),
        # Step 5 - uncomment to test a ticket with no KB match
        # ("INC0001016", "Cisco Webex not launching on Mac M2",
        #  "Cisco Webex not launching on Mac M2", "Software", "P3"),
    ]

    results = {}
    try:
        for number, short_desc, desc, cat, prio in test_tickets:
            results[number] = resolve_ticket(number, short_desc, desc, cat, prio)
    except anthropic.AuthenticationError:
        sys.exit("\nAPI key rejected. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.NotFoundError:
        sys.exit(f"\nModel '{MODEL}' not found. Set ISDO_MODEL in .env (e.g. claude-opus-5-5).")
    except anthropic.APIConnectionError:
        sys.exit("\nCould not reach the Anthropic API. Check your network or proxy.")

    print(f"\n{'=' * 55}")
    print("RESOLUTION SUMMARY")
    print(f"{'=' * 55}")
    for number, r in results.items():
        if r:
            outcome = "AUTO-RESOLVE" if r["auto_resolve"] else "HITL"
            print(f"  {number:<12} {r['confidence']:<7} {r['score']:>4.0%}  {outcome:<13} "
                  f"{r['kb_article_used']}")
        else:
            print(f"  {number:<12} (no resolution - HITL)")
