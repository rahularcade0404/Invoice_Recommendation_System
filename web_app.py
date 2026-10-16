# web_app.py

from pathlib import Path

import pandas as pd
import streamlit as st

from data_preprocessing import load_and_preprocess_data
from matrix_construction import build_user_item_matrix
from cosine_similarity import compute_similarity_matrices

from recommendation_engine import (
    recommend_products_item_based,
    recommend_majority_bought_products,
    recommend_state_majority_products,
)


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Invoice Data Recommendation Dashboard",
    layout="wide"
)

st.title("🛒 Invoice Data Recommendation Dashboard")

st.markdown(
    """
    Select a **Contact Name** below to view purchase history,
    AI-based product recommendations, personal repeat purchases,
    and same-state recommendations.
    """
)


# ============================================================
# LOAD COMPLETE RECOMMENDATION PIPELINE
# ============================================================

@st.cache_data
def load_pipeline():

    file_path = Path(__file__).resolve().parent / "data" / "Invoice_wise_detail_august.xlsx"

    # --------------------------------------------------------
    # STEP 1: DATA PREPROCESSING
    # --------------------------------------------------------

    raw_df = load_and_preprocess_data(file_path)

    # --------------------------------------------------------
    # STANDARDISE IMPORTANT TEXT COLUMNS
    # --------------------------------------------------------

    for col in [
        "Contact Name",
        "Material",
        "Material Desc",
        "Part Group",
        "Customer State Name",
        "MD",
        "Month",
    ]:

        if col in raw_df.columns:

            raw_df[col] = (raw_df[col].astype(str).str.strip())

    # --------------------------------------------------------
    # STEP 2: CREATE CONTACT-MATERIAL INTERACTIONS
    # --------------------------------------------------------
    user_item_df = (
        raw_df
        .groupby(
            [
                "Contact Name",
                "Material"
            ],
            as_index=False
        )
        .agg(Quantity=("Quantity", "sum"))
    )

    # --------------------------------------------------------
    # STEP 2: BUILD USER-ITEM MATRIX
    # --------------------------------------------------------

    user_item_matrix = build_user_item_matrix(
        user_item_df
    )

    # --------------------------------------------------------
    # STEP 3: CALCULATE ALL COSINE SIMILARITIES
    # --------------------------------------------------------

    (
        part_group_month_sim_df,
        part_group_state_sim_df,
        material_part_group_sim_df
    ) = compute_similarity_matrices(
        raw_df,
        user_item_matrix
    )

    return (
        raw_df,
        user_item_matrix,
        part_group_month_sim_df,
        part_group_state_sim_df,
        material_part_group_sim_df
    )


# ============================================================
# LOAD PIPELINE
# ============================================================

(
    raw_df,
    user_item_matrix,
    part_group_month_sim_df,
    part_group_state_sim_df,
    material_part_group_sim_df
) = load_pipeline()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Customer Settings")


# Contact Name selection
selected_contact = st.sidebar.selectbox(
    "Select Contact Name:",
    sorted(
        user_item_matrix.index.astype(str).tolist()
    )
)


# Number of recommendations
top_n = st.sidebar.slider(
    "Number of Recommendations:",
    1,
    10,
    5
)


# Similarity threshold
min_threshold = st.sidebar.slider(
    "Minimum Similarity Threshold:",
    0.0,
    1.0,
    0.10,
    step=0.05
)


# ============================================================
# PRODUCT / MATERIAL CATALOG
# ============================================================

catalog_columns = [
    col
    for col in [
        "Material",
        "Material Desc",
        "Part Group",
        "MD"
    ]
    if col in raw_df.columns
]


product_catalog = (
    raw_df[catalog_columns]
    .drop_duplicates(
        subset=["Material"]
    )
)


# ============================================================
# MAIN DASHBOARD
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# CUSTOMER PURCHASE HISTORY
# ============================================================

with col1:

    st.subheader(f"📋 Purchase History — {selected_contact}")

    cust_purchases = raw_df[
        raw_df["Contact Name"].astype(str)
        == str(selected_contact)
    ].copy()


    # --------------------------------------------------------
    # PURCHASE VALUE
    # --------------------------------------------------------

    if "Total VALUE" in cust_purchases.columns:

        cust_purchases["Total VALUE"] = pd.to_numeric(
            cust_purchases["Total VALUE"],
            errors="coerce"
        )


    if "Posting Date" in cust_purchases.columns:

        posting_dates = cust_purchases["Posting Date"]
        numeric_dates = pd.to_numeric(
            posting_dates,
            errors="coerce"
        )
        parsed_dates = pd.to_datetime(
            numeric_dates,
            unit="D",
            origin="1899-12-30",
            errors="coerce"
        )
        text_date_mask = numeric_dates.isna()
        parsed_dates.loc[text_date_mask] = pd.to_datetime(
            posting_dates.loc[text_date_mask],
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )
        cust_purchases["Posting Date"] = (
            parsed_dates.dt.strftime("%d-%m-%Y")
            .astype("string")
        )


    # --------------------------------------------------------
    # DISPLAY COLUMNS
    # --------------------------------------------------------

    purchase_columns = [
        col
        for col in [
            "Material",
            "Material Desc",
            "Part Group",
            "MD",
            "Customer State Name",
            "Month",
            "Total VALUE",
            "Total Quantity",
            "Posting Date"
        ]
        if col in cust_purchases.columns
    ]


    st.dataframe(
        cust_purchases[purchase_columns],
        width="stretch",
        height=500
    )


# ============================================================
# RIGHT COLUMN
# RECOMMENDATIONS
# ============================================================

with col2:

    # ========================================================
    # 1. ITEM-BASED COLLABORATIVE FILTERING
    # ========================================================

    st.subheader(
        "🤖 Similar Product Recommendations"
    )

    try:

        recs = recommend_products_item_based(
            str(selected_contact),
            user_item_matrix,
            material_part_group_sim_df,
            top_n=top_n,
            min_threshold=min_threshold
        )


        if not recs.empty:

            rec_df = recs.copy()


            # ------------------------------------------------
            # RENAME SCORE
            # ------------------------------------------------

            if "Score" in rec_df.columns:

                rec_df = rec_df.rename(
                    columns={
                        "Score":
                        "Recommendation Score"
                    }
                )


            # ------------------------------------------------
            # ADD MATERIAL INFORMATION
            # ------------------------------------------------

            rec_df = rec_df.merge(
                product_catalog,
                on="Material",
                how="left"
            )


            # ------------------------------------------------
            # SCORE 0-100
            # ------------------------------------------------

            if "Recommendation Score" in rec_df.columns:

                rec_df["Recommendation Score"] = (
                    pd.to_numeric(
                        rec_df["Recommendation Score"],
                        errors="coerce"
                    )
                    .clip(0, 100)
                    .round(2)
                    .map(lambda score: f"{score:.2f}%")
                )


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display_cols = [
                col
                for col in [
                    "Material",
                    "Material Desc",
                    "Part Group",
                    "MD",
                    "Recommendation Score"
                ]
                if col in rec_df.columns
            ]


            st.dataframe(
                rec_df[display_cols],
                width="stretch"
            )

        else:

            st.warning("No similar product recommendations "
                       "met the similarity threshold."
            )


    except Exception as e:

        st.error(
            f"Error generating similar product "
            f"recommendations: {e}"
        )


    st.markdown("---")


    # ========================================================
    # 2. PERSONAL REPEAT PURCHASE
    # ========================================================

    st.subheader(
        "🔄 Personal Repeat Purchase Recommendations"
    )

    try:

        majority_df = recommend_majority_bought_products(
            selected_contact,
            raw_df,
            top_n=top_n
        )


        if not majority_df.empty:

            maj_df = majority_df.copy()


            # ------------------------------------------------
            # SCORE
            # ------------------------------------------------

            if "Recommendation Score" in maj_df.columns:

                maj_df["Recommendation Score"] = (
                    pd.to_numeric(
                        maj_df["Recommendation Score"],
                        errors="coerce"
                    )
                    .clip(0, 100)
                    .round(2)
                    .map(lambda score: f"{score:.2f}%")
                )


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display_maj_cols = [
                col
                for col in [
                    "Contact Name",
                    "Recommended Part Group",
                    "Material",
                    "Material Desc",
                    "Previously Bought Quantity",
                    "Recommended Quantity",
                    "Recommendation Score"
                ]
                if col in maj_df.columns
            ]


            st.dataframe(
                maj_df[display_maj_cols],
                width="stretch"
            )

        else:

            st.warning(
                "No personal repeat purchase "
                "recommendations available."
            )


    except Exception as e:

        st.error(
            f"Error generating personal repeat "
            f"recommendations: {e}"
        )
    st.markdown("---")


    # ========================================================
    # 3. SAME-STATE RECOMMENDATIONS
    # ========================================================

    st.subheader("📍 Same-State Part Group Recommendations")

    try:

        state_df = recommend_state_majority_products(
            selected_contact,
            raw_df,
            top_n=top_n,
        )


        if not state_df.empty:

            state_merged = state_df.copy()


            # ------------------------------------------------
            # SCORE
            # ------------------------------------------------

            if "Recommendation Score" in state_merged.columns:

                state_merged["Recommendation Score"] = (
                    pd.to_numeric(
                        state_merged["Recommendation Score"],
                        errors="coerce"
                    )
                    .clip(0, 100)
                    .round(2)
                    .map(lambda score: f"{score:.2f}%")
                )


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display_state_cols = [
                col
                for col in [
                    "Contact Name",
                    "Customer State Name",
                    "Part Group",
                    "Material",
                    "Material Desc",
                    "Previously Bought Qty",
                    "Recommended Quantity",
                    "Recommendation Score"
                ]
                if col in state_merged.columns
            ]


            display_state_df = (
                state_merged[
                    display_state_cols
                ]
                .rename(
                    columns={
                        "Customer_Bought_Qty":
                        "Previously Bought Qty"
                    }
                )
            )


            st.dataframe(
                display_state_df,
                width="stretch"
            )

        else:

            st.warning(
                "No same-state recommendations "
                "found for this contact."
            )


    except Exception as e:

        st.error(
            f"Error generating same-state "
            f"recommendations: {e}"
        )


# ============================================================
# SIMILARITY INFORMATION
# ============================================================

st.markdown("---")

st.subheader("📊 Cosine Similarity Analysis")


sim_col1, sim_col2, sim_col3 = st.columns(3)


# ------------------------------------------------------------
# PERSONAL REPEAT PURCHASE SIMILARITY
# ------------------------------------------------------------

with sim_col1:

    st.markdown(
        "**Personal Repeat Purchase**"
    )

    st.caption(
        "Part Group similarity based on "
        "Jan-Aug quantity patterns."
    )

    st.dataframe(
        part_group_month_sim_df.iloc[:10, :10],
        width="stretch"
    )


# ------------------------------------------------------------
# SAME-STATE SIMILARITY
# ------------------------------------------------------------

with sim_col2:

    st.markdown(
        "**Same-State Recommendations**"
    )

    st.caption(
        "Part Group similarity based on "
        "state-wise quantity patterns."
    )

    st.dataframe(
        part_group_state_sim_df.iloc[:10, :10],
        width="stretch"
    )


# ------------------------------------------------------------
# SIMILAR PRODUCT SIMILARITY
# ------------------------------------------------------------

with sim_col3:

    st.markdown(
        "**Similar Products**"
    )

    st.caption(
        "Material similarity based on "
        "Part Group-wise quantity patterns."
    )

    st.dataframe(
        material_part_group_sim_df.iloc[:10, :10],
        width="stretch"
    )