"""
ISDO - Knowledge Base indexer (ChromaDB)

1) Reads every .md file in data/kb/
2) Splits each article into chunks at '## ' headings
3) Stores all chunks in the ChromaDB collection 'isdo_kb'
4) Runs sample queries and prints the best matching article + confidence

Requirements:  pip install chromadb
Run (from anywhere):  python labs/c1/kb_setup.py
Override paths:       python labs/c1/kb_setup.py --kb-dir data/kb --db-dir chroma_db

Defaults resolve from the project root (two folders above this file), so
data/kb and chroma_db are found no matter which folder you run it from.

Note: the first run downloads ChromaDB's default embedding model
(all-MiniLM-L6-v2, ~80 MB). Later runs use the cached copy.
"""

import argparse
import re
from pathlib import Path

import chromadb

COLLECTION_NAME = "isdo_kb"

# labs/c1/kb_setup.py -> project root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_DIR = PROJECT_ROOT / "data" / "kb"
DEFAULT_DB_DIR = PROJECT_ROOT / "chroma_db"

# One query per KB area; expected best match shown on the right.
SAMPLE_QUERIES = [
    "I forgot my password and my account is locked",           # password_reset.md
    "AnyConnect VPN keeps disconnecting when I work from home",  # vpn_troubleshooting.md
    "Emails are not syncing on my phone and Outlook is stuck",   # email_troubleshooting.md
    "Nobody on the finance floor can log in to SAP",             # erp_connectivity.md
]

HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# 1) Read + 2) chunk
# ---------------------------------------------------------------------------
def chunk_markdown(text: str, article_name: str) -> list[dict]:
    """Split one markdown article at '## ' headings.

    Text before the first '## ' (usually the '# Title' and an intro) becomes an
    'Overview' chunk. Each chunk is prefixed with the article title so a section
    such as '## Steps' still carries the context of what it is about.
    """
    title_match = TITLE_RE.search(text)
    title = title_match.group(1) if title_match else article_name

    headings = list(HEADING_RE.finditer(text))
    sections = []

    # Everything before the first '## ' heading
    intro_end = headings[0].start() if headings else len(text)
    intro = TITLE_RE.sub("", text[:intro_end], count=1).strip()
    if intro:
        sections.append(("Overview", intro))

    # Each '## ' heading up to the next one
    for i, match in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[match.end():end].strip()
        if body:
            sections.append((match.group(1), body))

    chunks = []
    for index, (section, body) in enumerate(sections):
        chunks.append({
            "id": f"{article_name}::{index}",
            "document": f"{title}\n{section}\n\n{body}",
            "metadata": {
                "article": article_name,
                "title": title,
                "section": section,
                "chunk_index": index,
            },
        })
    return chunks


def load_kb(kb_dir: Path) -> list[dict]:
    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {kb_dir.resolve()}")

    all_chunks = []
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, path.name)
        print(f"  {path.name:<40} {len(chunks)} chunks")
        all_chunks.extend(chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# 3) Store in ChromaDB
# ---------------------------------------------------------------------------
def build_collection(client, chunks: list[dict]):
    # Rebuild from scratch each run so edited or deleted articles don't linger.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance -> easy confidence score
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["document"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return collection


# ---------------------------------------------------------------------------
# 4) Test queries
# ---------------------------------------------------------------------------
def run_queries(collection, queries: list[str]) -> None:
    print("\n" + "=" * 78)
    print("Sample queries")
    print("=" * 78)
    for query in queries:
        result = collection.query(
            query_texts=[query],
            n_results=1,
            include=["metadatas", "distances"],
        )
        meta = result["metadatas"][0][0]
        distance = result["distances"][0][0]
        # Cosine distance runs 0 (identical) to 2 (opposite); confidence = 1 - distance
        confidence = max(0.0, min(1.0, 1.0 - distance))

        print(f"\nQuery:      {query}")
        print(f"Best match: {meta['article']}  (section: {meta['section']})")
        print(f"Confidence: {confidence:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Index ISDO KB articles into ChromaDB")
    parser.add_argument("--kb-dir", default=str(DEFAULT_KB_DIR), help="folder of .md articles")
    parser.add_argument("--db-dir", default=str(DEFAULT_DB_DIR), help="where ChromaDB stores data")
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    print(f"Reading KB articles from {kb_dir.resolve()}")
    chunks = load_kb(kb_dir)

    client = chromadb.PersistentClient(path=args.db_dir)
    collection = build_collection(client, chunks)
    print(f"\nStored {collection.count()} chunks in collection '{COLLECTION_NAME}' "
          f"({Path(args.db_dir).resolve()})")

    run_queries(collection, SAMPLE_QUERIES)


if __name__ == "__main__":
    main()
