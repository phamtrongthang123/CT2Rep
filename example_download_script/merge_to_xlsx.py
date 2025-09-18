import pandas as pd

# Read the failed volume names from the text file
with open("valid_failed.txt", "r") as f:
    failed_volumes = [line.strip() for line in f.readlines()]

# Read the CSV files
df_a = pd.read_csv("train_reports.csv")
df_b = pd.read_csv("validation_reports.csv")

# Merge the dataframes
merged_df = pd.concat([df_a, df_b])


# Filter out rows where VolumeName is a substring of any failed volume path
def should_exclude(volume_name):
    return any(volume_name in failed_path for failed_path in failed_volumes)


filtered_df = merged_df[~merged_df["VolumeName"].apply(should_exclude)]

# Extract AccessionNo from VolumeName
filtered_df["AccessionNo"] = filtered_df["VolumeName"].str.replace(r"_\d+\.nii\.gz$", "", regex=True)

# Filter out rows where Findings_EN is NaN
filtered_df = filtered_df.dropna(subset=['Findings_EN'])

# Save as Excel file
filtered_df.to_excel("merged_data.xlsx", index=False)

print("Files merged, filtered, and saved as merged_data.xlsx")
print(f"Excluded {len(merged_df) - len(filtered_df)} rows")
