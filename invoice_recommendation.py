from pathlib import Path
from cosine_similarity import compute_similarity_matrices
from data_preprocessing import load_and_preprocess_data
from matrix_construction import build_user_item_matrix
from recommendation_engine import (
    recommend_majority_bought_products,
    recommend_products_item_based,
    recommend_state_majority_products,
)


def invoice_recommendation(contact_name, raw_df, top_n=10):
    """
    Generates personalized product recommendations for a specific contact/customer based on their purchase history.

    Parameters:
    - contact_name (str): The name of the contact/customer for whom recommendations are to be generated.
    - raw_df (pd.DataFrame): The raw invoice data containing purchase history.

    Returns:
    - pd.DataFrame: A DataFrame containing recommended products along with their recommended quantities and recommendation scores.
    """
    user_item_df = (
        raw_df.groupby(["Contact Name", "Material"], as_index=False)
        .agg(Quantity=("Quantity", "sum"))
    )
    user_item_matrix = build_user_item_matrix(user_item_df)
    _, _, material_similarity = compute_similarity_matrices(raw_df, user_item_matrix)

    state_col = next(
        (
            column
            for column in ("Customer State Name", "State Name", "State")
            if column in raw_df.columns
        ),
        "Customer State Name",
    )

    return {
        "similar_products": recommend_products_item_based(
            customer_id=str(contact_name).strip(),
            user_item_matrix=user_item_matrix,
            item_sim_df=material_similarity,
            top_n=top_n,
            min_threshold=0.1,
        ),
        "personal_recommendations": recommend_majority_bought_products(
            contact_name=contact_name,
            raw_df=raw_df,
            top_n=top_n,
        ),
        "state_recommendations": recommend_state_majority_products(
            target_customer=contact_name,
            df=raw_df,
            state_col=state_col,
            top_n=top_n,
            low_qty_threshold=0.1,
        ),
    }


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "data" / "Invoice_wise_detail_august.xlsx"
    df_new = load_and_preprocess_data(data_path)
    contact_name = df_new["Contact Name"].dropna().iloc[0]

    for recommendation_type, recommendations in invoice_recommendation(contact_name=contact_name,raw_df=df_new, top_n=10).items():
        print(f"\n{recommendation_type.replace('_', ' ').title()}")
        print(recommendations)

