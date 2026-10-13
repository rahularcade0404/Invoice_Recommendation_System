# step3_cosine_similarity.py

from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data_preprocessing import load_and_preprocess_data
from matrix_construction import build_user_item_matrix


# ============================================================
# STEP 3: COMPUTING COSINE SIMILARITY MATRICES
# ============================================================

def compute_similarity_matrices(df, user_item_matrix=None):
    """
    Computes cosine similarity matrices for:

    1. Personal Repeat Purchase
       Part Group similarity based on Jan-Aug Quantity

    2. Same-State Recommendations
       Part Group similarity based on State-wise Quantity

    3. Similar Products
       Material similarity based on Part Group-wise Quantity
    """

    print("=" * 70)
    print("STEP 3: COMPUTING COSINE SIMILARITY MATRICES")
    print("=" * 70)


    # ========================================================
    # 1. PERSONAL REPEAT PURCHASE
    # ========================================================

    print("\n1. PERSONAL REPEAT PURCHASE SIMILARITY")

    history_months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug"
    ]

    # Data is already cleaned in Step 1.
    personal_df = df[df["Month"].isin(history_months)]

    # Part Group × Month
    part_group_month_matrix = (
        personal_df
        .pivot_table(
            index="Part Group",
            columns="Month",
            values="Quantity",
            aggfunc="sum",
            fill_value=0
        )
        .reindex(
            columns=history_months,
            fill_value=0
        )
    )

    print("Part Group-Month Matrix Shape:",part_group_month_matrix.shape)

    # Cosine similarity between Part Groups
    part_group_month_similarity = cosine_similarity(part_group_month_matrix)

    part_group_month_sim_df = pd.DataFrame(
        part_group_month_similarity,
        index=part_group_month_matrix.index,
        columns=part_group_month_matrix.index
    )

    print("\nPart Group Similarity - Jan-Aug Quantity:")
    print(part_group_month_sim_df.iloc[:5, :5])


    # ========================================================
    # 2. SAME-STATE RECOMMENDATION
    # ========================================================

    print("\n2. SAME-STATE PART GROUP SIMILARITY")

    # Part Group × State
    part_group_state_matrix = (
        df
        .pivot_table(
            index="Part Group",
            columns="Customer State Name",
            values="Quantity",
            aggfunc="sum",
            fill_value=0
        )
    )

    print("Part Group-State Matrix Shape:",part_group_state_matrix.shape)

    # Cosine similarity between Part Groups
    part_group_state_similarity = cosine_similarity(
        part_group_state_matrix
    )

    part_group_state_sim_df = pd.DataFrame(
        part_group_state_similarity,
        index=part_group_state_matrix.index,
        columns=part_group_state_matrix.index
    )

    print("\nPart Group Similarity - State-wise Quantity:")
    print(
        part_group_state_sim_df.iloc[:5, :5]
    )


    # ========================================================
    # 3. SIMILAR PRODUCTS
    # ========================================================

    print("\n3. SIMILAR PRODUCT SIMILARITY")

    # Material × Part Group
    material_part_group_matrix = (
        df
        .pivot_table(
            index="Material",
            columns="Part Group",
            values="Quantity",
            aggfunc="sum",
            fill_value=0
        )
    )

    print(
        "Material-Part Group Matrix Shape:",
        material_part_group_matrix.shape
    )

    # Cosine similarity between Materials
    material_part_group_similarity = cosine_similarity(material_part_group_matrix)

    material_part_group_sim_df = pd.DataFrame(
        material_part_group_similarity,
        index=material_part_group_matrix.index,
        columns=material_part_group_matrix.index
    )

    print("\nMaterial Similarity - Part Group-wise Quantity:")
    print(
        material_part_group_sim_df.iloc[:5, :5]
    )


    # ========================================================
    # RETURN ALL THREE MATRICES
    # ========================================================

    return (
        part_group_month_sim_df,
        part_group_state_sim_df,
        material_part_group_sim_df
    )


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    file_path = Path(__file__).parent / "data" / "Invoice_wise_detail_august.xlsx"

    # --------------------------------------------------------
    # STEP 1
    # Data preprocessing happens here ONLY
    # --------------------------------------------------------

    cleaned_df = load_and_preprocess_data(file_path)

    # --------------------------------------------------------
    # STEP 2
    # Build User-Item Matrix
    # --------------------------------------------------------

    user_item_df = (cleaned_df.groupby(
            ["Contact Name", "Material"],
            as_index=False
        )
        .agg(Quantity=("Quantity", "sum")))

    user_item_matrix = build_user_item_matrix(user_item_df)

    # --------------------------------------------------------
    # STEP 3
    # Calculate cosine similarities
    # --------------------------------------------------------

    (
        part_group_month_sim_df,
        part_group_state_sim_df,
        material_part_group_sim_df
    ) = compute_similarity_matrices(
        cleaned_df,
        user_item_matrix
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PERSONAL REPEAT PURCHASE")
    print("=" * 70)

    print(part_group_month_sim_df.iloc[:5, :5])

    print("\n" + "=" * 70)
    print("SAME-STATE RECOMMENDATION")
    print("=" * 70)

    print(part_group_state_sim_df.iloc[:5, :5])
    print("\n" + "=" * 70)
    print("SIMILAR PRODUCTS")
    print("=" * 70)
    print(material_part_group_sim_df.iloc[:5, :5])