from pathlib import Path
import pandas as pd

# 1. Define paths relative to script location
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_loader"

# Path to your main target file
MAIN_FILE = DATA_DIR / "Invoice_wise_detail_august.xlsx"

# List of incoming/additional files to append to the main file
ADDITIONAL_FILES = []

def load_and_clean_sheet(file_path: Path) -> pd.DataFrame:
    """Reads the 'Invoice Data' sheet (or first sheet) from an Excel file."""
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    with pd.ExcelFile(file_path) as workbook:
        # Case-insensitive sheet matching
        sheet_name = next(
            (
                s for s in workbook.sheet_names
                if s.strip().casefold() == "invoice data"
            ),
            workbook.sheet_names[0],
        )
        return pd.read_excel(workbook, sheet_name=sheet_name)


def merge_excel_files(main_file_path: Path, new_file_paths: list[Path]) -> pd.DataFrame:
    """
    Reads the main file and additional files, strips extra header rows,
    and returns a single consolidated DataFrame.
    """
    dataframes = []

    # Load main file (this provides the primary header structure)
    print(f"Loading main file: {main_file_path.name}")
    main_df = load_and_clean_sheet(main_file_path)
    dataframes.append(main_df)

    # Load additional files
    for file_path in new_file_paths:
        print(f"Appending data from: {file_path.name}")
        df = load_and_clean_sheet(file_path)
        dataframes.append(df)

    # pd.concat merges rows together under a single header row,
    # effectively dropping the column header names of incoming files.
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df


if __name__ == "__main__":
    # Execute merge
    merged_df = merge_excel_files(MAIN_FILE, ADDITIONAL_FILES)

    # Save back to the main file (or a new combined file)
    OUTPUT_FILE = DATA_DIR / "Invoice_wise_detail_consolidated.xlsx"
    merged_df.to_excel(OUTPUT_FILE, index=False, sheet_name="Invoice Data")
    
    print(f"\nSuccessfully merged {len(ADDITIONAL_FILES) + 1} files into {OUTPUT_FILE.name}")
    print(f"Total consolidated rows: {len(merged_df)}")
