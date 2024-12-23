import pandas as pd

#%% Load your main dataset
df = pd.read_csv("data/preprocessed_train_delivery1.csv")

# Load the FIPS reference file
county_fips = pd.read_csv("data/county_fips_master.csv", encoding="latin1")

county_fips["county_name"] = county_fips["county_name"].str.replace(" County", "", case=False).str.strip().str.upper()

# Inspect the FIPS reference file to confirm column names
print(county_fips.head())

# Merge the main dataset with the FIPS file
# Assuming `County` in `county_fips` matches `County of Injury` in `df`
df = df.merge(county_fips, left_on="County of Injury", right_on="county_name", how="left")

# Ensure FIPS codes are zero-padded to 5 digits
df["fips"] = df["fips"].astype(str).str.zfill(5)

# Verify the merged data
print("Merged DataFrame:")
print(df.head())

# Save the merged DataFrame if needed
df.to_csv("data/merged_with_fips.csv", index=False)