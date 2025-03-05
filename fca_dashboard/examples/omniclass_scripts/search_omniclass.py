import pandas as pd

# Load the CSV file
df = pd.read_csv('fca_dashboard/generator/ingest/omniclass.csv')

# Print the total number of rows
print(f"Total rows: {len(df)}")

# Function to find rows that match a pattern
def find_rows(pattern, column='OmniClass_Code'):
    matches = df[df[column].str.contains(pattern, na=False)]
    if not matches.empty:
        print(f"\nFound {len(matches)} rows matching '{pattern}':")
        print(f"First match at row {matches.index[0]}")
        print(f"Last match at row {matches.index[-1]}")
        print(f"Sample: {matches.iloc[0]['OmniClass_Code']}, {matches.iloc[0]['OmniClass_Title']}")
        return matches.index[0], matches.index[-1]
    else:
        print(f"\nNo matches found for '{pattern}'")
        return None, None

# Search for each table
print("\n--- Searching for Table 21 (Elements) ---")
start_21, end_21 = find_rows("^21-")  # Codes starting with "21-"

print("\n--- Searching for Table 22 (Work Results) ---")
start_22, end_22 = find_rows("^22-")  # Codes starting with "22-"

print("\n--- Searching for Table 23 (Products) ---")
start_23, end_23 = find_rows("^23-")  # Codes starting with "23-"

# Search for divisions 22 and 23 in each table
print("\n--- Searching for Division 22 (Plumbing) in each table ---")
start_21_22, end_21_22 = find_rows("^21-22")  # Table 21, Division 22
start_22_22, end_22_22 = find_rows("^22-22")  # Table 22, Division 22
start_23_22, end_23_22 = find_rows("^23-22")  # Table 23, Division 22

print("\n--- Searching for Division 23 (HVAC) in each table ---")
start_21_23, end_21_23 = find_rows("^21-23")  # Table 21, Division 23
start_22_23, end_22_23 = find_rows("^22-23")  # Table 22, Division 23
start_23_23, end_23_23 = find_rows("^23-23")  # Table 23, Division 23

# Search for the specific example pattern
print("\n--- Searching for the example pattern (22-23) ---")
start_example, end_example = find_rows("^22-23")  # The pattern from the example

# Print a summary of the ranges for easy reference
print("\n\n=== SUMMARY OF ROW RANGES ===")
print("Use these with the --start and --end parameters")

if start_21 is not None and end_21 is not None:
    print(f"Table 21 (Elements): --start {start_21} --end {end_21+1}")

if start_22 is not None and end_22 is not None:
    print(f"Table 22 (Work Results): --start {start_22} --end {end_22+1}")

if start_23 is not None and end_23 is not None:
    print(f"Table 23 (Products): --start {start_23} --end {end_23+1}")

if start_21_22 is not None and end_21_22 is not None:
    print(f"Table 21, Division 22 (Elements - Plumbing): --start {start_21_22} --end {end_21_22+1}")

if start_21_23 is not None and end_21_23 is not None:
    print(f"Table 21, Division 23 (Elements - HVAC): --start {start_21_23} --end {end_21_23+1}")

if start_22_22 is not None and end_22_22 is not None:
    print(f"Table 22, Division 22 (Work Results - Plumbing): --start {start_22_22} --end {end_22_22+1}")

if start_22_23 is not None and end_22_23 is not None:
    print(f"Table 22, Division 23 (Work Results - HVAC): --start {start_22_23} --end {end_22_23+1}")

if start_23_22 is not None and end_23_22 is not None:
    print(f"Table 23, Division 22 (Products - Plumbing): --start {start_23_22} --end {end_23_22+1}")

if start_23_23 is not None and end_23_23 is not None:
    print(f"Table 23, Division 23 (Products - HVAC): --start {start_23_23} --end {end_23_23+1}")

if start_example is not None and end_example is not None:
    print(f"Example pattern (22-23): --start {start_example} --end {end_example+1}")