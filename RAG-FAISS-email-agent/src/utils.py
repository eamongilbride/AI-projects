import re
import time
import json
import requests
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import SentenceTransformerEmbeddings


# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Direct (Regex) Matching ───────────────────────────────────────────────────
PRODUCT_CODE_PATTERN = re.compile(
    r"([A-Za-z]\s*[A-Za-z]\s*[A-Za-z]\s*\d\s*\d\s*\d\s*\d)"
)


def extract_product_codes(text: str) -> List[str]:
    """
    Find all 3-letter+4-digit codes in the text, normalize them to uppercase.
    """
    raw = PRODUCT_CODE_PATTERN.findall(text)
    return [re.sub(r"\s+", "", code).upper() for code in raw]


def find_matching_products_full(
    codes: List[str],
    products_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Map extracted product codes to full catalog entries.
    """
    codes_set = {c.upper() for c in codes}
    matched = products_df[
        products_df["product_id"].str.upper().isin(codes_set)
    ]
    results = matched.to_dict(orient="records")
    for item in results:
        item["match_type"] = "direct"
    return results


# ── FAISS Retriever ────────────────────────────────────────────────────────────
def format_product(row: pd.Series) -> str:
    """
    Serialize the product fields into a JSON string for embedding.
    """
    return json.dumps({
        "product_id":  row["product_id"],
        "name":        row["name"],
        "category":    row["category"],
        "description": row.get("description", ""),
    })


def build_faiss_retriever(
    products_df: pd.DataFrame,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 3
) -> BaseRetriever:
    """
    Build and return a FAISS-based retriever.
    """
    # Build documents
    docs = [
        Document(
            page_content=format_product(row),
            metadata=row.to_dict()
        )
        for _, row in products_df.iterrows()
    ]

    # Instantiate embeddings & vectorstore
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Return retriever
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

# ── Merge Direct + Semantic Matches ──────────────────────────────────────────
def merge_direct_and_semantic(
    inputs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine direct (regex) and semantic (FAISS) matches into the prompt context.
    Expects:
      - inputs["direct_matches"]: List[dict]
      - inputs["semantic_matches"]: List[Document]
      - inputs["email_text"]: str
      - inputs["product_categories"]: List[str]
    Returns a dict with keys:
      - product_matches, email_text, product_categories
    """
    direct = inputs["direct_matches"]
    semantic_docs = inputs["semantic_matches"]

    # Build semantic match dicts from page_content
    semantic_matches = []
    for doc in semantic_docs:
        data = json.loads(doc.page_content)
        data["match_type"] = "semantic"
        semantic_matches.append(data)

    # Deduplicate: direct matches take priority
    direct_ids = {m["product_id"] for m in direct}
    merged = direct + [
        m for m in semantic_matches if m["product_id"] not in direct_ids
    ]

    return {
        "product_matches":    merged,
        "email_text":         inputs["email_text"],
        "product_categories": inputs["product_categories"],
    }


# ── CSV I/O ───────────────────────────────────────────────────────────────────
def load_csv_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load product catalog and sample emails from the data folder.
    """
    products = pd.read_csv(DATA_DIR / "product_catalog.csv")
    emails   = pd.read_csv(DATA_DIR / "sample_emails.csv")
    return products, emails


def _write_csv(df: pd.DataFrame, filename: str) -> None:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)


def write_email_classifications(df: pd.DataFrame) -> None:
    _write_csv(df, "email-classifications.csv")


def write_order_status(df: pd.DataFrame) -> None:
    _write_csv(df, "order-status.csv")


def write_order_responses(df: pd.DataFrame) -> None:
    _write_csv(df, "order-response.csv")


def write_inquiry_responses(df: pd.DataFrame) -> None:
    _write_csv(df, "inquiry-response.csv")


# ── Translation Helper ─────────────────────────────────────────────────────────
def translate_to_english(text: str, source_lang: str) -> str:
    """
    If non-English, translate via Google’s public API; else return original.
    """
    if source_lang.lower().startswith("en"):
        return text

    try:
        time.sleep(1)  # throttle
        enc = urllib.parse.quote(text)
        url = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl={source_lang}&tl=en&dt=t&q={enc}"
        )
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            translated_text = ''.join([seg[0] for seg in result[0]])
            return translated_text
    except Exception:
        # fallback on error
        return text