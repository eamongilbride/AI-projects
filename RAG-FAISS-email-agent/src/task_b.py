import ast
from typing import List, Tuple, Dict

import pandas as pd

OrderLineStatus = Tuple[str, int, str]  # (product_id, quantity, status)

def generate_response_email(
    order_lines: List[OrderLineStatus], products_df: pd.DataFrame
) -> str:
    """
    Build a customer-facing response email based on per-line order status.
    - order_lines: list of (product_id, quantity, status)
    - products_df: full catalog with 'product_id', 'name', 'stock', 'price'
    """
    out_of_stock = [line for line in order_lines if line[2] == "out of stock"]
    in_stock = [line for line in order_lines if line[2] != "out of stock"]

    lines = ["Dear Customer,", ""]
    if out_of_stock:
        lines.append(
            "We are unable to fulfill your order completely due to out-of-stock items:"
        )
        for pid, qty, _ in out_of_stock:
            row = products_df.loc[products_df["product_id"] == pid].iloc[0]
            lines.append(f"- {row['name']}: Requested {qty} (stock: {row['stock']})")

        lines.extend([
            "",
            "Options:",
            "1. Wait for restock (TBD).",
            "2. Choose an alternative product.",
        ])

        # Offer partial fulfillment
        if in_stock:
            lines.append("3. Partial order for available items:")
            for pid, qty, _ in in_stock:
                name = products_df.loc[products_df["product_id"] == pid, "name"].iloc[0]
                lines.append(f"- {name} (quantity: {qty})")

        lines.extend([
            "",
            "Please let us know how you’d like to proceed.",
            "Customer Support"
        ])
    else:
        # All items in stock
        lines.append("Thank you for your order! The following items will ship shortly:")
        total = 0.0
        for pid, qty, _ in in_stock:
            row = products_df.loc[products_df["product_id"] == pid].iloc[0]
            total += row["price"] * qty
            lines.append(f"- {row['name']} (quantity: {qty})")
        lines.extend([
            "",
            f"Invoice total: ${total:.2f}. Please pay within 30 days.",
            "",
            "Thank you for your business!",
            "Customer Support"
        ])

    return "\n".join(lines)


def process_order_requests(
    emails_df: pd.DataFrame, products_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each email labeled 'order request':
    1. Parse GPT-derived order_details
    2. Check stock & update it
    3. Record per-line status
    4. Generate a response email

    Returns two DataFrames:
    - order_status: columns ['email_id','product_id','quantity','status']
    - order_responses: columns ['email_id','response']
    """
    status_rows: List[Dict[str, any]] = []
    response_rows: List[Dict[str, any]] = []

    orders = emails_df[emails_df["category"] == "order request"]
    for _, row in orders.iterrows():
        email_id = row["email_id"]
        raw = row.get("gpt_response", "")
        try:
            details = ast.literal_eval(raw).get("order_details", {})
        except Exception as e:
            print("Email %s: failed to parse order_details: %s", email_id, e)
            continue

        line_status: List[OrderLineStatus] = []
        order_fulfilled = True

        for pid, qty in details.items():
            # Validate product exists
            prod_rows = products_df[products_df["product_id"] == pid]
            if prod_rows.empty:
                line_status.append((pid, qty, "out of stock"))
                order_fulfilled = False
                continue

            stock = prod_rows.iloc[0]["stock"]
            # Handle "all" vs numeric quantities
            if isinstance(qty, str) and qty.lower() == "all":
                qty = int(stock)
            else:
                try:
                    qty = int(qty)
                except Exception:
                    print("Email %s: invalid qty '%s' for %s", email_id, qty, pid)
                    line_status.append((pid, qty, "invalid quantity"))
                    order_fulfilled = False
                    continue

            if stock >= qty:
                products_df.loc[products_df["product_id"] == pid, "stock"] = stock - qty
                line_status.append((pid, qty, "created"))
            else:
                line_status.append((pid, qty, "out of stock"))
                order_fulfilled = False

        # Build status rows
        for pid, qty, status in line_status:
            final_status = status if order_fulfilled else "out of stock"
            status_rows.append({
                "email_id":    email_id,
                "product_id":  pid,
                "quantity":    qty,
                "status":      final_status
            })

        # Build response row
        email_body = generate_response_email(line_status, products_df)
        response_rows.append({
            "email_id": email_id,
            "response": email_body
        })

    return pd.DataFrame(status_rows), pd.DataFrame(response_rows)