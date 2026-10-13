# step1_data_preprocessing.py

from pathlib import Path
import pandas as pd


def _normalise_column_name(column):
    """
    Normalizes column names so that small differences
    in spaces, underscores and hyphens do not cause issues.
    """
    return (str(column).strip().lower().replace("_", " ").replace("-", " "))


def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the InvoiceWise dataset.

    Main columns used by the recommendation system:

    Contact Name  -> Customer/User
    Material      -> Product/Item
    Material Desc -> Product Description
    Part Group    -> Product Group
    Quantity      -> Customer-Product Interaction
    Total VALUE   -> Transaction Value
    Bill Doc      -> Transaction/Order
    Month         -> Purchase Month
    """

    print("--- STEP 1: INVOICEWISE DATA PREPROCESSING & EDA ---")

  
    # LOAD DATA

    workbook = pd.ExcelFile(file_path)
    preferred_sheet = next(
        (
            sheet
            for sheet in workbook.sheet_names
            if sheet.strip().casefold() == "invoice data"
        ),
        workbook.sheet_names[0],
    )
    df = pd.read_excel(workbook, sheet_name=preferred_sheet)
    print(f"Original Dataset Shape: {df.shape}")

    
    # 2. NORMALIZE COLUMN NAMES
    
    df.columns = [_normalise_column_name(column) for column in df.columns]

   
    # 3. RENAME INVOICEWISE COLUMNS TO STANDARD NAMES
    rename_map = {
        "contact name": "Contact Name",
        "customer": "Customer",
        "customer no": "Customer No",

        "part group": "Part Group",

        "material": "Material",
        "material desc": "Material Desc",

        "quantity": "Quantity",

        "total value": "Total VALUE",

        "bill doc": "Bill Doc",

        "month": "Month",
        "posting date": "Posting Date",

        "state name": "State Name",
        "customer state name": "Customer State Name",
        "md": "MD",
    }

    df = df.rename(columns={column: rename_map[column]
            for column in df.columns
            if column in rename_map
        }
    )

    # ============================================================
    # 4. CHECK REQUIRED COLUMNS
    # ============================================================

    required_columns = ["Contact Name","Material","Material Desc","Part Group","Quantity",]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required InvoiceWise columns: "
            f"{missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    
    # 5. CLEAN TEXT COLUMNS
    text_columns = [
        "Contact Name",
        "Material",
        "Material Desc",
        "Part Group",
    ]

    for column in text_columns:
        df[column] = (df[column].astype("string").str.strip())

    
    # 6. CLEAN QUANTITY
    df["Quantity"] = pd.to_numeric(
        df["Quantity"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )

    
    # 7. CLEAN TOTAL VALUE
    if "Total VALUE" in df.columns:

        df["Total VALUE"] = (df["Total VALUE"].astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        df["Total VALUE"] = pd.to_numeric(df["Total VALUE"],errors="coerce")

    else:
        df["Total VALUE"] = 0.0

   
    # 8. REMOVE DUPLICATE RECORDS
    
    # 9. REMOVE MISSING REQUIRED VALUES
    before_missing = len(df)

    # Identify rows with NaN values or blank text in required columns
    missing_mask = (
        df[required_columns].isna().any(axis=1)
        | df[text_columns].apply(
            lambda column: column.astype(str).str.strip().eq("")
        ).any(axis=1)
    )

    missing_rows = df.loc[missing_mask]
    print("\nComplete Rows Containing Missing Values:")
    print(f"Number of rows found: {len(missing_rows)}")
    if not missing_rows.empty:
        rows_text = missing_rows.to_string(index=False)
        print(rows_text.encode("ascii", "backslashreplace").decode("ascii"))

    df = df.loc[~missing_mask].copy()
    missing_removed = before_missing - len(df)
    print(f"\nRows Removed Due to Missing Values: {missing_removed}")
    print(f"Rows Remaining: {len(df)}")

    # ============================================================
    # 10. REMOVE RETURNS / INVALID QUANTITIES
    # ============================================================

    df = df[df["Quantity"] > 0].copy()

    if df.empty:
        raise ValueError("No valid rows remain after quantity filtering.")

    # 11. OPTIONAL OUTLIER CONTROL
    # Extremely large quantities can dominate
    # the collaborative filtering matrix.
    # We cap Quantity at the 99th percentile.

    max_threshold = df["Quantity"].quantile(0.99)
    df["Quantity"] = df["Quantity"].clip(upper=max_threshold)

    # ============================================================
    # 12. BASIC EDA
    # ============================================================

    print("\n--- CLEANED DATASET INFORMATION ---")

    print(f"Cleaned Dataset Shape: {df.shape}")

    print(f"Unique Contacts: {df['Contact Name'].nunique()}")

    print(f"Unique Materials: {df['Material'].nunique()}")

    print(f"Unique Part Groups: {df['Part Group'].nunique()}")

    # ============================================================
    # 13. AGGREGATE CUSTOMER-PRODUCT INTERACTIONS
    # ============================================================

    user_item_df = df.groupby(
        ["Contact Name", "Material"], as_index=False
    ).agg(Quantity=("Quantity", "sum"))

    # ============================================================
    # 14. MATRIX SPARSITY
    # ============================================================

    n_contacts = user_item_df["Contact Name"].nunique()

    n_materials = user_item_df["Material"].nunique()

    if n_contacts and n_materials:
        sparsity = (
            1 - len(user_item_df) / (n_contacts * n_materials)
        ) * 100
        print(f"User-Item Matrix Sparsity: {sparsity:.2f}%")

        print("\nAggregated Contact-Material Interactions:")
        print(user_item_df.head())
    return df
# ================================================================
# DIRECT EXECUTION
# ================================================================

if __name__ == "__main__":

    file_path = Path(__file__).resolve().parent / "data" / "Invoice_wise_detail_august.xlsx"
    cleaned_df = load_and_preprocess_data(file_path)

    print("\nFinal Cleaned InvoiceWise Data:")
    print(cleaned_df.head())
