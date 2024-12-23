
import pandas as pd
import json

# Load the datasets
data_path = "data/merged_with_fips.csv"  # Replace with your dataset path
geojson_path = "data/counties.json"  # Path to the GeoJSON file

# Load the merged dataset
print("Loading dataset...")
df = pd.read_csv(data_path)

# Load GeoJSON file
print("Loading GeoJSON file...")
with open(geojson_path) as response:
    counties = json.load(response)

# Extract FIPS from GeoJSON
geojson_fips = {feature["id"] for feature in counties["features"]}

# Step 1: Check for FIPS column existence
if "fips" not in df.columns:
    raise ValueError("The 'fips' column is missing in the dataset. Ensure the merge with FIPS was successful.")

# Step 2: Ensure FIPS is zero-padded and of type string
print("Ensuring FIPS is zero-padded...")
df["fips"] = df["fips"].astype(str).str.zfill(5)

# Step 3: Check for null or missing FIPS
print("Checking for missing or null FIPS...")
missing_fips = df["fips"].isnull().sum()
if missing_fips > 0:
    print(f"Number of rows with missing FIPS: {missing_fips}")
    print("Dropping rows with missing FIPS...")
    df = df.dropna(subset=["fips"])

# Step 4: Inspect unique FIPS values in the dataset
print("\nUnique FIPS values in the dataset:")
print(df["fips"].unique())

# Step 5: Check for mismatched FIPS between dataset and GeoJSON
print("\nChecking mismatched FIPS between dataset and GeoJSON...")
dataset_fips = set(df["fips"])
mismatched_fips = dataset_fips - geojson_fips
if mismatched_fips:
    print("The following FIPS codes are in the dataset but not in the GeoJSON file:")
    print(mismatched_fips)
else:
    print("All FIPS codes in the dataset match those in the GeoJSON file.")

# Step 6: Clean invalid FIPS codes
print("\nCleaning invalid FIPS codes...")
df = df[df["fips"].apply(lambda x: x.isdigit() if isinstance(x, str) else False)]

# Step 7: Convert FIPS to integer, then zero-pad to 5 digits
print("\nFormatting FIPS codes...")
df["fips"] = df["fips"].astype(float).astype(int).astype(str).str.zfill(5)

# Step 8: Filter dataset for valid FIPS codes
print("\nFiltering for valid FIPS codes...")
df = df[df["fips"].isin(geojson_fips)]
print(f"Filtered dataset contains {len(df)} rows.")

# Step 9: Aggregate accident counts by FIPS
print("\nAggregating accident counts by FIPS...")
accidents_per_county = df.groupby("fips").size().reset_index(name="accident_count")
print("Accident counts per county:")
print(accidents_per_county.head())

# Step 10: Save debug outputs (optional)
accidents_per_county.to_csv("debug_accidents_per_county.csv", index=False)
print("\nAccident counts per county saved to 'debug_accidents_per_county.csv'.")

# Optional: Save the cleaned dataset with validated FIPS
df.to_csv("debug_cleaned_dataset.csv", index=False)
print("Cleaned dataset saved to 'debug_cleaned_dataset.csv'.")
