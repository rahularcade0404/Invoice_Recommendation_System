import numpy as np
import pandas as pd


def recommend_products_item_based(
    customer_id: str,
    user_item_matrix: pd.DataFrame,
    item_sim_df: pd.DataFrame,
    top_n: int = 5,
    min_threshold: float = 0.1,
) -> pd.DataFrame:
    """Generates AI recommendations using Item-Item Cosine Similarity,

    filtering out relationships below the minimum similarity threshold.
    """
    if customer_id not in user_item_matrix.index:
        return pd.DataFrame(columns=["Part Group","Material Desc","Material", "Score"])

    user_profile = user_item_matrix.loc[customer_id]
    profile_values = user_profile.to_numpy()
    purchased_mask = profile_values > 0
    unpurchased_mask = ~purchased_mask
    purchased = user_item_matrix.columns[purchased_mask]
    unpurchased = user_item_matrix.columns[unpurchased_mask]

    if len(purchased) == 0 or len(unpurchased) == 0:
        return pd.DataFrame(columns=["Material", "Score"])

    similarities = item_sim_df.loc[unpurchased, purchased].to_numpy(dtype=float,copy=False,)
    similarities = np.where(similarities >= min_threshold, similarities, 0.0)
    similarity_totals = similarities.sum(axis=1)
    weighted_totals = similarities @ profile_values[purchased_mask]
    scores = np.divide(
        weighted_totals,
        similarity_totals,
        out=np.zeros_like(weighted_totals),
        where=similarity_totals > 0,
    )

    rec_series = pd.Series(scores, index=unpurchased).sort_values(ascending=False).head(top_n)

    # Convert to percentage score
    max_val = rec_series.max() if rec_series.max() > 0 else 1.0
    rec_df = pd.DataFrame(
        {
            "Material": rec_series.index,
            "Score": (rec_series.values / max_val) * 100,
        }
    )

    # Remove zero-score items
    rec_df = rec_df[rec_df["Score"] > 0]

    return rec_df

def recommend_majority_bought_products(contact_name: str,raw_df: pd.DataFrame,top_n: int = 5,) -> pd.DataFrame:
    """Recommend material-specific repeat purchases for the contact."""
    output_columns = [
        "Contact Name",
        "Recommended Part Group",
        "Material",
        "Material Desc",
        "Previously Bought Quantity",
        "Recommended Quantity",
        "Recommendation Score",
    ]

    required_columns = {
        "Contact Name",
        "Part Group",
        "Material",
        "Material Desc",
        "Quantity",
    }
    if not required_columns.issubset(raw_df.columns):
        return pd.DataFrame(columns=output_columns)

    data = raw_df.copy()
    data["Contact Name"] = data["Contact Name"].astype(str).str.strip()
    data["Part Group"] = data["Part Group"].astype(str).str.strip()
    data["Material"] = data["Material"].astype(str).str.strip()
    data["Material Desc"] = data["Material Desc"].astype(str).str.strip()
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data = data.dropna(
        subset=[
            "Contact Name",
            "Part Group",
            "Material",
            "Material Desc",
            "Quantity",
        ]
    )

    customer_data = data[data["Contact Name"] == str(contact_name).strip()]
    if customer_data.empty:
        return pd.DataFrame(columns=output_columns)

    stats = (customer_data.groupby(["Part Group", "Material", "Material Desc"])
        .agg(Total_Quantity=("Quantity", "sum"),Avg_Order_Qty=("Quantity", "mean"),)
        .reset_index()
        .sort_values("Total_Quantity", ascending=False)
        .head(top_n)
    )
    if stats.empty:
        return pd.DataFrame(columns=output_columns)

    max_total_qty = stats["Total_Quantity"].max()
    stats["Recommendation Score"] = (
        stats["Total_Quantity"] / max_total_qty * 100
        if max_total_qty > 0
        else 0.0
    )
    stats["Recommended Quantity"] = (
        stats["Avg_Order_Qty"].round().astype(int).clip(lower=1)
    )
    stats["Contact Name"] = contact_name

    result = stats[
        [
            "Contact Name",
            "Part Group",
            "Material",
            "Material Desc",
            "Total_Quantity",
            "Recommended Quantity",
            "Recommendation Score",
        ]
    ].rename(columns={"Part Group": "Recommended Part Group",
            "Total_Quantity": "Previously Bought Quantity",}
    )
    result["Recommendation Score"] = result["Recommendation Score"].round(2).clip(0, 100)
    return result.reset_index(drop=True)



def recommend_state_majority_products(
    target_customer: str,
    df: pd.DataFrame,
    customer_col: str = 'Contact Name',  # Or 'Customer ID'
    state_col: str = 'Customer State Name',
    item_col: str = 'Part Group',  # Or 'Product ID'
    quantity_col: str = 'Quantity',
    top_n: int = 5,
    low_qty_threshold: int = 2,
) -> pd.DataFrame:
    """Recommend top-selling state Part Groups, prioritizing low purchases.

    Groups not purchased or purchased in low quantities are listed first. If
    fewer than ``top_n`` groups meet that preference, the remaining state
    bestsellers fill the result.
    """
    data = df.copy()

    required_columns = {
        customer_col,
        state_col,
        item_col,
        quantity_col,
    }
    if not required_columns.issubset(data.columns):
        return pd.DataFrame()

    # 1. Standardize string columns to prevent clean string mismatches
    for col in [customer_col, state_col, item_col]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()

    data[quantity_col] = pd.to_numeric(data[quantity_col], errors="coerce")
    data = data.dropna(subset=[customer_col, state_col, item_col, quantity_col])

    # 2. Retrieve the target customer's state
    target_customer = str(target_customer).strip()
    cust_records = data[data[customer_col].str.casefold() == target_customer.casefold()]
    if cust_records.empty:
        print(f"Customer/Contact '{target_customer}' not found in dataset.")
        return pd.DataFrame()

    target_state = cust_records[state_col].iloc[0]

    # 3. Filter full dataset for transactions within the same state
    state_df = data[data[state_col] == target_state]

    # 4. Calculate total state-wide sales volume for each Part Group / Product
    state_popular_items = (
        state_df.groupby(item_col)[quantity_col]
        .sum()
        .reset_index()
        .rename(columns={quantity_col: 'State_Total_Qty'})
        .sort_values(by='State_Total_Qty', ascending=False)
    )

    # 5. Calculate target customer's personal purchase history for these items
    cust_item_history = (cust_records.groupby(item_col)[quantity_col].sum().reset_index()
        .rename(columns={quantity_col: 'Quantity'})
    )

    # 6. Merge state popularity with target customer's history
    merged_df = state_popular_items.merge(
        cust_item_history, on=item_col, how='left'
    )
    merged_df['Quantity'] = merged_df['Quantity'].fillna(0)

    # 7. Prioritize items the customer has not bought or bought in low quantity.
    # Fill the requested top_n with the remaining state bestsellers when there
    # are not enough low-purchase groups to satisfy the request.
    preferred_recs = merged_df[merged_df['Quantity'] <= low_qty_threshold].copy()
    fallback_recs = merged_df[merged_df['Quantity'] > low_qty_threshold].copy()
    filtered_recs = pd.concat(
        [preferred_recs, fallback_recs],
        ignore_index=True,
    ).head(top_n)

    if filtered_recs.empty:
        return pd.DataFrame()

    # 8. Compute Recommendation Score (0 - 100% relative to the top item in that state)
    max_state_volume = filtered_recs['State_Total_Qty'].max()
    if max_state_volume > 0:
        filtered_recs['Recommendation Score'] = (
            filtered_recs['State_Total_Qty'] / max_state_volume
        ) * 100
    else:
        filtered_recs['Recommendation Score'] = 0.0

    # 9. Compute Recommended Reorder Batch (Average transaction size for this item in that state)
    state_avg_batch = (state_df.groupby(item_col)[quantity_col].mean().round().astype(int).clip(lower=1))
    filtered_recs['Recommended Quantity'] = (filtered_recs[item_col].map(state_avg_batch).fillna(1).astype(int))

    # 10. Format and organize final display table
    filtered_recs[state_col] = target_state
    filtered_recs[customer_col] = target_customer

    output_columns = [
        customer_col,
        state_col,
        item_col,
        'Quantity',
        'Recommended Quantity',
        'Recommendation Score',
    ]

    result = filtered_recs[output_columns].head(top_n)
    return result.rename(columns={'Quantity': 'Previously Bought Qty'})

# def invoice_wise_recommendations(df, customer_id, top_n=3):
    #     """Generates recommendations based on invoice-wise analysis for a specific customer."""
    #     # Filter For the target customer
    #     personal_recommendations(df, customer_id, top_n=3)
    #
    #     # Filter For Same State
    #     same_state_recommendations(df, top_n=3)

    #      # Filter For Cross State
    #      cross_state_recommendations(df, top_n=3)
