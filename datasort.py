import os
import shutil
import pandas as pd

# --- Configuration ---
excel_file = "chronology_extracted_entries.xlsx"  # path to your Excel file
txt_folder = "data/input"                         # folder containing your txt files
output_folder = "sorted_files"                    # destination folder for sorted files
aelfric_folder = os.path.join(output_folder, "Aelfric")
not_aelfric_folder = os.path.join(output_folder, "Not Aelfric")

# Create output folders
os.makedirs(aelfric_folder, exist_ok=True)
os.makedirs(not_aelfric_folder, exist_ok=True)

# Load Excel data
df = pd.read_excel(excel_file)

# Get all CorpusIDs with Authorship == 1
aelfric_ids = df.loc[df["Authorship"] == 1, "CorpusID"].dropna().astype(str).tolist()

# --- Helper: check if a file belongs to Aelfric ---
def is_aelfric_file(corpus_id, aelfric_ids):
    for cid in aelfric_ids:
        # exact match
        if corpus_id == cid:
            return True
        # child match: starts with cid + "."
        if corpus_id.startswith(cid + "."):
            return True
    return False

# --- Main Loop ---
for file in os.listdir(txt_folder):
    if file.endswith(".txt"):
        corpus_id = os.path.splitext(file)[0]  # filename without extension
        src = os.path.join(txt_folder, file)

        if is_aelfric_file(corpus_id, aelfric_ids):
            shutil.copy(src, aelfric_folder)
        else:
            shutil.copy(src, not_aelfric_folder)

print("✅ Sorting complete.")
