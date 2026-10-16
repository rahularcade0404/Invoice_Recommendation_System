import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from database.database import (
    MajorityReorderRecommendation,
    SavedRecommendation,
    SessionLocal,
    StateRecommendation,
    init_db,
)
from recommendation_engine import (recommend_majority_bought_products,recommend_products_item_based,recommend_state_majority_products,)

def batch_generate_and_save_to_db(raw_df: pd.DataFrame):
    """Iterates over all contacts, calculates recommendations, and stores them in SQL."""
    init_db()
    db = SessionLocal()

    # Clear previous recommendations before refreshing
    db.query(SavedRecommendation).delete()
    db.query(MajorityReorderRecommendation).delete()
    db.query(StateRecommendation).delete()
    db.commit()

    # Standardize string fields
    for col in [
        "Contact Name",
        "Material",
        "Part Group",
        "Customer State Name",
    ]:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].astype("string").str.strip()

    raw_df["Quantity"] = pd.to_numeric(raw_df["Quantity"], errors="coerce")
    raw_df = raw_df.dropna(subset=["Contact Name", "Material", "Quantity"])

    material_part_groups = (
        raw_df.dropna(subset=["Part Group"])
        .groupby(["Material", "Part Group"], as_index=False)["Quantity"]
        .sum()
        .sort_values(
            ["Material", "Quantity", "Part Group"],
            ascending=[True, False, True],
        )
        .drop_duplicates("Material")
        .set_index("Material")["Part Group"]
        .to_dict()
    )

    customer_state = {}
    if "Customer State Name" in raw_df.columns:
        customer_state = (
            raw_df.dropna(subset=["Customer State Name"])
            .drop_duplicates("Contact Name")
            .set_index("Contact Name")["Customer State Name"]
            .to_dict()
        )

    material_average_quantity = (
        raw_df.groupby("Material")["Quantity"].mean()
        .round()
        .astype(int)
        .clip(lower=1)
        .to_dict()
    )

    # Pre-build matrices for Collaborative Filtering
    user_item_df = (raw_df.groupby(["Contact Name", "Material"], as_index=False)["Quantity"].sum())
    user_item_matrix = user_item_df.pivot_table(
        index="Contact Name", columns="Material", values="Quantity", fill_value=0
    )


    similarity = cosine_similarity(user_item_matrix.T)
    item_sim_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns,
    )

    contacts = raw_df["Contact Name"].unique()
    records_to_insert = []
    state_records_to_insert = []

    for contact in contacts:
        # 1. Collaborative Filtering Recs
        collab_recs = recommend_products_item_based(contact, user_item_matrix, item_sim_df, top_n=3, min_threshold=0.05)
        if not collab_recs.empty:
            for _, row in collab_recs.iterrows():
                part_group = material_part_groups.get(row["Material"])
                if pd.isna(part_group):
                    continue
                records_to_insert.append(
                    SavedRecommendation(
                        contact_name=contact,
                        engine_type="Collaborative",
                        recommended_part_group=str(part_group),
                        recommended_qty=int(material_average_quantity.get(row["Material"], 1)), 

                        recommendation_score=f"{row['Score']:.2f}%",
                        state=customer_state.get(contact),
                    )
                )

        # 2. Majority Reorders
        majority_recs = recommend_majority_bought_products(contact, raw_df, top_n=3)
        if not majority_recs.empty:
            for _, row in majority_recs.iterrows():
                records_to_insert.append(
                    MajorityReorderRecommendation(
                        contact_name=contact,
                        recommended_part_group=str(row["Recommended Part Group"]),
                        material=str(row["Material"]),
                        material_desc=str(row["Material Desc"]),
                        previously_bought_qty=int(row["Previously Bought Quantity"]),
                        recommended_qty=int(row["Recommended Quantity"]),
                        recommendation_score=f"{row['Recommendation Score']:.2f}%",
                    )
                )

        # 3. State Majority Bestsellers
        state_recs = recommend_state_majority_products(contact, raw_df, customer_col="Contact Name", top_n=3)
        if not state_recs.empty:
            for _, row in state_recs.iterrows():
                state_records_to_insert.append(
                    StateRecommendation(
                        contact_name=contact,
                        recommended_part_group=str(row["Part Group"]),
                        previously_bought_qty=int(row["Previously Bought Qty"]),
                        recommended_qty=int(row["Recommended Quantity"]),
                        recommendation_score=f"{row['Recommendation Score']:.2f}%",
                        state=str(row["Customer State Name"]),
                    )
                )

    state_records_to_insert.sort(
        key=lambda record: float(record.recommendation_score.rstrip("%")),
        reverse=True,
    )

    # Bulk insert all generated recommendations into SQL
    if records_to_insert:
        db.bulk_save_objects(records_to_insert)
    if state_records_to_insert:
        db.bulk_save_objects(state_records_to_insert)
    db.commit()
    db.close()
    total_records = len(records_to_insert) + len(state_records_to_insert)
    print(f"Successfully calculated and saved {total_records} recommendations to SQL Database.")


if __name__ == "__main__":
    workbook = Path(__file__).resolve().parent.parent / "data" / "Invoice_wise_detail_august.xlsx"
    df = pd.read_excel(workbook)
    batch_generate_and_save_to_db(df)