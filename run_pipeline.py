"""End-to-end recommendation pipeline for the InvoiceWise dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


WORKBOOK = Path(__file__).resolve().parent / "data" / "Invoice_wise_detail_august.xlsx"
REQUIRED_COLUMNS = {
    "Contact Name",
    "Material",
    "Material Desc",
    "Part Group",
    "Quantity",
}


def load_invoice_data(file_path: Path = WORKBOOK) -> pd.DataFrame:
    """Load and clean invoice rows from the Invoice Data sheet when available."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Invoice workbook not found: {file_path}")

    with pd.ExcelFile(file_path) as workbook:
        sheet_name = next(
            (
                sheet
                for sheet in workbook.sheet_names
                if sheet.strip().casefold() == "invoice data"
            ),
            workbook.sheet_names[0],
        )
        data = pd.read_excel(workbook, sheet_name=sheet_name)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            "Workbook is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    for column in ("Contact Name", "Material", "Material Desc", "Part Group"):
        data[column] = data[column].astype("string").str.strip()
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    return data.dropna(subset=["Contact Name", "Material", "Quantity"])


def merge_invoice_files(file_paths: list[str | Path]) -> pd.DataFrame:
    """Read and concatenate invoice workbooks, preserving their source names."""
    frames = []
    for path in file_paths:
        file_path = Path(path)
        data = load_invoice_data(file_path)
        data["Source_File"] = file_path.name
        frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def generate_recommendations(data: pd.DataFrame) -> pd.DataFrame:
    """Build item similarities and recommend products for the first contact."""
    if data.empty:
        raise ValueError("No valid invoice rows were found in the workbook.")

    print("\n" + "=" * 60)
    print("STEP 2: CONSTRUCTING USER-ITEM MATRIX")
    print("=" * 60)
    user_item = (
        data.groupby(["Contact Name", "Material"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    print(f"Matrix Dimensions: {user_item.shape}")
    print("\nUser-Item Matrix Preview:")
    print(user_item.iloc[:5, :5])

    print("\n" + "=" * 60)
    print("STEP 3: CALCULATING ITEM-ITEM COSINE SIMILARITY")
    print("=" * 60)
    similarities = cosine_similarity(user_item.T)
    item_similarity = pd.DataFrame(
        similarities, index=user_item.columns, columns=user_item.columns
    )
    print("Item Similarity Matrix successfully calculated.")
    print("\nItem Similarity Matrix Preview:")
    print(item_similarity.iloc[:5, :5])

    print("\n" + "=" * 60)
    print("STEP 4: GENERATING RECOMMENDATIONS")
    print("=" * 60)
    target_contact = user_item.index[0]
    profile = user_item.loc[target_contact]
    purchased = profile[profile > 0]
    candidates = profile[profile == 0].index
    print(f"Target Contact: {target_contact}")
    print(f"Previously purchased materials: {len(purchased)}")
    print(f"Purchased material preview: {list(purchased.index[:10])}")

    scores = {}
    for material in candidates:
        similarity = item_similarity.loc[material, purchased.index]
        similarity_total = similarity.sum()
        scores[material] = (
            float(np.dot(similarity, purchased) / similarity_total)
            if similarity_total > 0
            else 0.0
        )

    recommendations = (
        pd.Series(scores, dtype="float64")
        .sort_values(ascending=False)
        .head(5)
        .rename_axis("Material")
        .reset_index(name="Raw Score")
    )
    max_score = recommendations["Raw Score"].max() if not recommendations.empty else 0
    recommendations["Recommendation Score"] = (
        recommendations["Raw Score"] / max_score * 100 if max_score > 0 else 0.0
    )

    contact_rows = data[data["Contact Name"] == target_contact]
    average_quantity = contact_rows["Quantity"].mean()
    if pd.isna(average_quantity) or average_quantity <= 0:
        average_quantity = 1
    recommendations["Recommended Quantity"] = max(1, int(round(average_quantity)))
    recommendations["Contact Name"] = target_contact

    product_details = data[["Material", "Material Desc", "Part Group"]].drop_duplicates(subset="Material")
    recommendations = recommendations.merge(product_details, on="Material", how="left")
    output = recommendations[
        [
            "Contact Name",
            "Part Group",
            "Material",
            "Material Desc",
            "Recommended Quantity",
            "Recommendation Score",
        ]
    ].copy()
    output["Recommendation Score"] = output["Recommendation Score"].clip(0, 100).round(2)
    return output


def main() -> None:
    print("=" * 60)
    print("STEP 1: PREPROCESSING & DATA EXPLORATION")
    print("=" * 60)
    data = load_invoice_data()
    print(
        f"Dataset Loaded: {len(data)} transactions, "
        f"{data['Contact Name'].nunique()} contacts, "
        f"{data['Material'].nunique()} products."
    )

    recommendations = generate_recommendations(data)
    print("\nTop Recommended Products:")
    if recommendations.empty:
        print("No unpurchased products are available to recommend for this contact.")
    else:
        print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()