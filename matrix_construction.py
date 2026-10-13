# step2_matrix_construction.py
from pathlib import Path
import pandas as pd
from data_preprocessing import load_and_preprocess_data


def build_user_item_matrix(user_item_df):
    """
    Converts aggregated InvoiceWise Contact-Material
    interactions into a User-Item Matrix.

    Rows    -> Contact Name
    Columns -> Material
    Values  -> Quantity
    """

    print("--- STEP 2: CONSTRUCTING USER-ITEM PIVOT MATRIX ---")

    # ------------------------------------------------------------
    # 1. Make a copy so the original DataFrame is not modified
    # ------------------------------------------------------------

    user_item_df = user_item_df.copy()

    # ------------------------------------------------------------
    # 2. Clean Contact Name
    # ------------------------------------------------------------

    user_item_df["Contact Name"] = user_item_df["Contact Name"].astype(str).str.strip()

    # 3. Clean Material
    user_item_df["Material"] = (user_item_df["Material"].astype(str).str.strip())

    # ------------------------------------------------------------
    # 4. Ensure Quantity is numeric
    # ------------------------------------------------------------

    user_item_df["Quantity"] = pd.to_numeric(user_item_df["Quantity"],errors="coerce")

    # Remove invalid quantity values

    user_item_df = user_item_df.dropna(subset=["Contact Name","Material","Quantity"])

    # ------------------------------------------------------------
    # 5. Construct User-Item Matrix
    # ------------------------------------------------------------

    user_item_matrix = user_item_df.pivot_table(
        index="Contact Name",
        columns="Material",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    )

    # 6. Display Matrix Dimensions
   
    print(f"User-Item Matrix Shape: "f"{user_item_matrix.shape}")
    print(f"Number of Contacts: "f"{user_item_matrix.shape[0]}")

    print(f"Number of Materials: "f"{user_item_matrix.shape[1]}")
    return user_item_matrix


# ================================================================
# DIRECT EXECUTION
# ================================================================

if __name__ == "__main__":

    file_path = Path(__file__).resolve().parent / "data" / "Invoice_wise_detail_august.xlsx"

    # STEP 1
    cleaned_df = load_and_preprocess_data(file_path)

    # ------------------------------------------------------------
    # Create aggregated Contact-Material interactions
    # ------------------------------------------------------------

    user_item_df = (cleaned_df.groupby(["Contact Name","Material"],as_index=False).agg(Quantity=("Quantity", "sum")))

    # STEP 2
    user_item_matrix = build_user_item_matrix(user_item_df)

    # Display Matrix Preview
    print("\nUser-Item Matrix Preview:")
    print(user_item_matrix.iloc[:5, :5])

