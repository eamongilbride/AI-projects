import ast
from typing import List, Dict

import pandas as pd

def finalize_product_inquiry_responses(
    emails_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate customer-facing responses for all 'product inquiry' emails.

    Expects emails_df to have columns:
      - 'email_id'
      - 'category' == 'product inquiry'
      - 'gpt_response' as a JSON string with a 'response' field

    Returns a DataFrame with columns:
      - 'email_id'
      - 'response'
    """
    records: List[Dict[str, str]] = []

    inquiries = emails_df[emails_df["category"] == "product inquiry"]
    for _, row in inquiries.iterrows():
        email_id = row["email_id"]
        raw = row.get("gpt_response", "")

        try:
            parsed = ast.literal_eval(raw)
            core = parsed.get("response", "").strip()
        except Exception as e:
            print(
                "Failed to parse gpt_response for email_id %s: %s", email_id, e
            )
            core = raw

        full_response = (
            "Dear Customer,\n\n"
            f"{core}\n\n"
            "Best regards,\n"
            "Customer Support"
        )

        records.append({
            "email_id": email_id,
            "response": full_response
        })

    return pd.DataFrame(records)