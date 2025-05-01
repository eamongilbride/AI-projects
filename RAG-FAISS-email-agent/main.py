import os
import argparse

from dotenv import load_dotenv

from src.utils import (
    load_csv_inputs,
    write_email_classifications,
    write_order_status,
    write_order_responses,
    write_inquiry_responses,
    build_faiss_retriever,
    extract_product_codes,
    find_matching_products_full,
    merge_direct_and_semantic
)

from src.task_a import process_emails, product_categories
from src.task_b import process_order_requests
from src.task_c import finalize_product_inquiry_responses

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda


# ──────────────────────────────────────────────────────────────────────────────
def load_config():
    """Load and validate environment variables."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
    model_name = os.getenv("LLM_NAME", "gpt-4o")
    return model_name, api_key, base_url

def build_chain(retriever, model_name, api_key, base_url, products_df):
    """Build the LangChain pipeline."""
    # Regex-based direct matches
    direct_match = RunnableLambda(
        lambda ctx: find_matching_products_full(
            extract_product_codes(ctx["email_text"]), products_df
        )
    )

    # Semantic FAISS matches (unwrap the mapping)
    semantic_match = RunnableLambda(lambda ctx: retriever.invoke(ctx["email_text"]))

    # Merger (prepare package for LLM)
    merger = RunnableLambda(merge_direct_and_semantic)

    # Prompt and LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are a customer service AI trained to classify emails into one of two categories:

            1. "order request" — The sender wants to place an order. These emails typically contain:
            - a product name or reference,
            - a quantity (e.g., "all you have in stock", "5", "a pair", "all available", etc.),
            - urgency or shipping details (e.g., "ASAP", "need 10 units", etc.)
            - and are matched to one or more product catalog entries (see below)

            2. "product inquiry" — The sender is asking for more information, expressing interest, or asking questions but does NOT clearly place an order.

            ---

            Matching product catalog entries (as JSON objects):

            {product_matches}

            ---

            Instructions:

            Return your classification and structured result in this **exact format**. Do not include backticks, code blocks, <think>, or extra explanation. Only return the JSON dictionary in the following exact format:

            For an order request:
            {{
            "label": "order request",
            "order_details": {{
                "PRODUCT_ID_1": "quantity or 'all'",
                "PRODUCT_ID_2": "quantity or 'all'"
            }}
            }}

            For the quantity, use "all" only if the customer is clearly asking for "all in stock" or "all available". Otherwise, use the specific number quantity. If the quantity is less than current stock, still return it as an order request for that quantity, and we'll handle stock issues later.
            If the quantity given is a range (e.g "three to four"), it must be labelled a product inquiry rather than a order request, and make sure to tell them our available stock.

            For a product inquiry:
            {{
            "label": "product inquiry",
            "response": "a helpful, concise customer service reply tailored to the email and product context"
            }}

            When generating a product inquiry "response", consider any product matches given above, then consider suggesting they check the product catalog.
            You can help narrow down their catalog search by suggesting one of our categories, if any are applicable. Categories: {product_categories}.
            ---

            Now, based on the email and product data, classify and respond appropriately.

            Email:
            \"\"\"{email_text}\"\"\"
            """),
        ("user", "Matching products: {product_matches}\nEmail text: {email_text}\nCategories: {product_categories}\n"),
    ])
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        openai_api_key=api_key,
        openai_api_base=base_url,
    )

    return (
        RunnableMap({
            "direct_matches":     direct_match,
            "semantic_matches":   semantic_match,
            "email_text":         RunnableLambda(lambda ctx: ctx["email_text"]),
            "product_categories": lambda _: product_categories,
        })
        | merger
        | prompt
        | llm
    )

def main(demo_mode: int = None):
    model_name, api_key, base_url = load_config()

    # Load inputs
    products_df, emails_df = load_csv_inputs()
    if demo_mode is not None:
        emails_df = emails_df.head(demo_mode)

    # FAISS retriever
    retriever = build_faiss_retriever(products_df)

    # Build and trace pipeline
    chain = build_chain(retriever, model_name, api_key, base_url, products_df)

    # Task A: classify
    process_emails(chain, emails_df)
    write_email_classifications(emails_df[['email_id', 'category']])

    # Task B: order requests
    status_df, responses_df = process_order_requests(emails_df, products_df)
    write_order_status(status_df)
    write_order_responses(responses_df)

    # Task C: product inquiries
    inquiry_df = finalize_product_inquiry_responses(emails_df)
    write_inquiry_responses(inquiry_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG-FAISS email assistant.")
    parser.add_argument(
        "--demo",
        type=int,
        default=None,
        metavar="N",
        help="If set, only process the first N emails for demo."
    )
    args = parser.parse_args()
    main(demo_mode=args.demo)