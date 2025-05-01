import re
import ast
from typing import Tuple

import pandas as pd
from langdetect import detect
from src.utils import translate_to_english

# Suggestion categories for prompts
product_categories = [
    'Accessories', 'Bags', "Kids' Clothing", 'Loungewear',
    "Men's Accessories", "Men's Clothing", "Men's Shoes",
    "Women's Clothing", "Women's Shoes"
]

def safe_extract_label(response_str: str) -> Tuple[str, str]:
    """
    Clean and parse the LLM's JSON response.
    Returns a tuple of (label, cleaned_response_str).
    """
    cleaned = response_str.strip()
    # Strip triple-backtick fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|python)?\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    # Remove any <think>…</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    try:
        parsed = ast.literal_eval(cleaned)
        return parsed.get("label", "unknown"), cleaned
    except Exception as e:
        return "parse_error", cleaned


def process_emails(chain, emails_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each email:
      1. Detect language & translate to English if needed
      2. Invoke the LangChain pipeline (`chain`) on the translated text
      3. Extract the classification label and raw JSON response

    Returns the original DataFrame augmented with:
      - full_text, lang, translated
      - category, gpt_response
    """
    # Build a full_text column from subject+body
    emails_df['full_text'] = (
        "Subject: " + emails_df['subject'].fillna('') + "\n\n"
        "Message: " + emails_df['message'].fillna('')
    )

    # Language detection & translation
    emails_df['lang'] = emails_df['full_text'].apply(detect)
    emails_df['translated'] = emails_df.apply(
        lambda row: translate_to_english(row['full_text'], row['lang']),
        axis=1
    )

    # Classification
    def _classify(text: str) -> pd.Series:
        raw_response = chain.invoke({"email_text": text})
        label, parsed = safe_extract_label(raw_response.content)
        return pd.Series([label, parsed])

    emails_df[['category', 'gpt_response']] = emails_df['translated'].apply(_classify)

    return emails_df