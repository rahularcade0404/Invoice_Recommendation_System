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

def recommend_majority_bought_products(
    contact_name: str,
    raw_df: pd.DataFrame,
    top_n: int = 8,
    history_start: str | pd.Timestamp | None = None,
    history_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Rank personal repeat purchases by quantity, frequency, and recency."""
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
        "Posting Date",
    }
    if not required_columns.issubset(raw_df.columns):
        return pd.DataFrame(columns=output_columns)

    data = raw_df.copy()
    for column in ["Contact Name", "Part Group", "Material", "Material Desc"]:
        data[column] = data[column].astype("string").str.strip()
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")

    posting_dates = data["Posting Date"]
    numeric_dates = pd.to_numeric(posting_dates, errors="coerce")
    if pd.api.types.is_numeric_dtype(posting_dates):
        data["Posting Date"] = pd.to_datetime(
            posting_dates,
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
    else:
        data["Posting Date"] = pd.to_datetime(
            posting_dates,
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
        excel_date_mask = data["Posting Date"].isna() & numeric_dates.notna()
        data.loc[excel_date_mask, "Posting Date"] = pd.to_datetime(
            numeric_dates.loc[excel_date_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    data = data.dropna(subset=list(required_columns))
    data = data[data["Quantity"] > 0].copy()
    data["Purchase Month"] = data["Posting Date"].dt.to_period("M")

    start_period = (
        pd.Period(history_start, freq="M")
        if history_start is not None
        else data["Purchase Month"].min()
    )
    end_period = (
        pd.Period(history_end, freq="M")
        if history_end is not None
        else data["Purchase Month"].max()
    )
    if pd.isna(start_period) or pd.isna(end_period) or start_period > end_period:
        return pd.DataFrame(columns=output_columns)
    data = data[data["Purchase Month"].between(start_period, end_period)].copy()

    customer_data = data[
        data["Contact Name"].str.casefold()
        == str(contact_name).strip().casefold()
    ]
    if customer_data.empty:
        return pd.DataFrame(columns=output_columns)

    stats = (
        customer_data.groupby(["Part Group", "Material", "Material Desc"])
        .agg(
            Total_Quantity=("Quantity", "sum"),
            Avg_Order_Qty=("Quantity", "mean"),
            Purchase_Month_Count=("Purchase Month", "nunique"),
            Last_Purchase_Month=("Purchase Month", "max"),
        )
        .reset_index()
    )
    history_month_count = end_period.ordinal - start_period.ordinal + 1
    quantity_score = stats["Total_Quantity"] / stats["Total_Quantity"].max() * 100
    frequency_score = stats["Purchase_Month_Count"] / history_month_count * 100
    recency_months = (
        stats["Last_Purchase_Month"].map(lambda month: month.ordinal)
        - start_period.ordinal
        + 1
    )
    recency_score = recency_months / history_month_count * 100
    stats["Recommendation Score"] = (
        0.40 * quantity_score
        + 0.40 * frequency_score
        + 0.20 * recency_score
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
    ].rename(
        columns={
            "Part Group": "Recommended Part Group",
            "Total_Quantity": "Previously Bought Quantity",
        }
    )
    result["Recommendation Score"] = result["Recommendation Score"].round(2).clip(0, 100)
    return result.sort_values(
        "Recommendation Score", ascending=False
    ).head(top_n).reset_index(drop=True)



def recommend_state_majority_products(
    target_customer: str,
    df: pd.DataFrame,
    customer_col: str = "Contact Name",
    state_col: str = "Customer State Name",
    item_col: str = "Part Group",
    quantity_col: str = "Quantity",
    top_n: int = 8,
) -> pd.DataFrame:
    """
    Recommend popular state-wide Part Groups the contact has never
    purchased in the selected state.

    Recommendations are limited to groups with no purchases by the contact
    in the selected state and are ranked by state-wide popularity.
    """

    output_columns = [
        customer_col,
        state_col,
        item_col,
        "Previously Bought Qty",
        "State Total Quantity",
        "Recommended Quantity",
        "Recommendation Score",
    ]

    data = df.copy()

    required_columns = {customer_col, state_col, item_col, quantity_col}

    if not required_columns.issubset(data.columns):
        return pd.DataFrame(columns=output_columns)

    # 1. Clean and validate data
    for col in [customer_col, state_col, item_col]:
        data[col] = data[col].astype("string").str.strip()

    data[quantity_col] = pd.to_numeric(data[quantity_col], errors="coerce")

    data = data.dropna(subset=[customer_col, state_col, item_col, quantity_col])

    data = data[
        (data[customer_col] != "")
        & (data[state_col] != "")
        & (data[item_col] != "")
        & (data[quantity_col] > 0)
    ]

    target_customer = str(target_customer).strip()

    # 2. Find the contact's historical records
    cust_records = data[data[customer_col].str.casefold()== target_customer.casefold()]

    if cust_records.empty:
        return pd.DataFrame(columns=output_columns)

    # Use the most frequently recorded state for the contact
    target_state = cust_records[state_col].mode().iloc[0]

    # 3. Select transactions from the target state
    state_df = data[data[state_col] == target_state]

    if state_df.empty:
        return pd.DataFrame(columns=output_columns)

    # 4. Calculate state-wide quantity for each Part Group
    state_stats = (
        state_df.groupby(item_col)[quantity_col]
        .agg(State_Total_Quantity="sum", State_Avg_Quantity="mean")
        .reset_index()
    )

    max_state_qty = state_stats["State_Total_Quantity"].max()

    if max_state_qty <= 0:
        return pd.DataFrame(columns=output_columns)

    # 5. Calculate state popularity score
    state_stats["Recommendation Score"] = (
        state_stats["State_Total_Quantity"]
        / max_state_qty
        * 100
    )

    # 6. Keep only groups the contact has never purchased in this state.
    state_customer_records = cust_records[cust_records[state_col] == target_state]
    purchased_groups = set(state_customer_records[item_col].unique())
    unseen_groups = state_stats[
        ~state_stats[item_col].isin(purchased_groups)
    ].copy()

    # 7. Keep only groups never purchased by this contact.
    recommendations = unseen_groups

    if recommendations.empty:
        return pd.DataFrame(columns=output_columns)

    # 8. Recommend the most popular eligible new opportunities
    recommendations = recommendations.sort_values(
        "State_Total_Quantity",
        ascending=False,
    ).head(top_n).copy()

    # Since all recommended part groups have never been purchased,
    # their previously bought quantity is zero.
    recommendations["Previously Bought Qty"] = 0

    # Give zero-purchase opportunities a 20% score uplift, capped at 100.
    recommendations["Recommendation Score"] = (
        recommendations["Recommendation Score"] 
    ).clip(upper=100)

    # Suggested quantity based on average state transaction-row quantity
    recommendations["Recommended Quantity"] = (
    recommendations["State_Avg_Quantity"].round().clip(lower=1).astype(int))

    recommendations[customer_col] = target_customer
    recommendations[state_col] = target_state

    recommendations = recommendations.rename(columns= { item_col: "Part Group", "State_Total_Quantity": "State Total Quantity"} )

    recommendations["Recommendation Score"] = (recommendations["Recommendation Score"].round(2).clip(0, 100))

    return recommendations[output_columns].reset_index(drop=True)
    # def invoice_wise_recommendations(df, customer_id, top_n=3):
    #     """Generates recommendations based on invoice-wise analysis for a specific customer."""
    #     # Filter For the target customer
    #     personal_recommendations(df, customer_id, top_n=3)
    #
    #     # Filter For Same State
    #     same_state_recommendations(df, top_n=3)

    #      # Filter For Cross State
    #      cross_state_recommendations(df, top_n=3)
