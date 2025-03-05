# Feature Engineering for HVAC Equipment Classification

When classifying HVAC equipment, effective feature engineering is crucial for
accurate model performance. Let me explain the most important features and
engineering techniques specifically for HVAC equipment classification:

## Key Features for HVAC Equipment Classification

### 1. Equipment Descriptive Features

- **Equipment Type Indicators**: Terms like "pump," "boiler," "chiller," "fan"
  that directly indicate equipment category
- **Manufacturer-Specific Terminology**: Brand-specific model designations and
  terminology
- **System Service Designations**: "HVAC," "mechanical," "plumbing"
  classifications
- **Function Descriptors**: Terms describing what the equipment does (e.g.,
  "cooling," "heating," "ventilation")

### 2. Technical Specifications

- **Capacity Metrics**: Size indicators like BTU, tons, horsepower, CFM
- **Dimensional Features**: Physical dimensions that may indicate equipment
  class
- **Energy Ratings**: Efficiency metrics like EER, SEER, COP
- **Operating Parameters**: Temperature ranges, pressure ratings
- **Refrigerant Types**: R-410A, R-22, etc., which can indicate equipment
  generation and type

### 3. System Integration Features

- **Connection Types**: How the equipment connects to other systems
- **Control Interface**: BMS integration capabilities
- **Mounting Configuration**: Ceiling, wall, floor, or roof mounted
- **Service Access Requirements**: Clearance needs that indicate equipment type

## Feature Engineering Techniques

### 1. Text-Based Feature Engineering

```python
# Create combined text features from multiple fields
df['text_features'] = (
    df['Asset Category'] + ' ' +
    df['Equip Name ID'] + ' ' +
    df['Sub System Type'] + ' ' +
    df['Drawing Abbreviation'] + ' ' +
    df['Operations System']
)

# Convert specific fields to lowercase for normalization
df['text_features'] = df['text_features'].str.lower()

# Remove special characters that might confuse the model
df['text_features'] = df['text_features'].str.replace('[^\w\s]', ' ', regex=True)
```

### 2. N-gram Feature Extraction

N-grams are crucial for capturing HVAC terminology that often includes
multi-word technical phrases:

```python
# TF-IDF vectorization with n-grams
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),  # Capture up to 3-word phrases (trigrams)
    min_df=2,            # Ignore very rare terms
    max_df=0.9           # Ignore very common terms
)

# Example HVAC-specific n-grams that would be captured:
# "air handling unit", "variable frequency drive", "direct digital control"
```

### 3. Numerical Feature Normalization

```python
# Convert size features to numeric and normalize
df['Equipment_Size'] = pd.to_numeric(df['Equipment Size'], errors='coerce')

# Create unit-normalized sizes (e.g., convert HP to BTU equivalent)
def normalize_to_common_unit(row):
    if row['Unit'] == 'HP':
        return row['Equipment_Size'] * 2545  # Convert HP to BTU/hr
    elif row['Unit'] == 'TONS':
        return row['Equipment_Size'] * 12000  # Convert Tons to BTU/hr
    elif row['Unit'] == 'CFM':
        return row['Equipment_Size'] * 3.16  # Approximate BTU/hr per CFM
    else:
        return row['Equipment_Size']

df['normalized_size'] = df.apply(normalize_to_common_unit, axis=1)
```

### 4. Domain-Specific Feature Creation

```python
# Create binary features for common HVAC characteristics
df['is_air_handler'] = df['text_features'].str.contains('air handling|ahu').astype(int)
df['is_cooling'] = df['text_features'].str.contains('cool|chill|refrig|condenser').astype(int)
df['is_heating'] = df['text_features'].str.contains('heat|boiler|steam').astype(int)
df['is_ventilation'] = df['text_features'].str.contains('fan|ventilat|exhaust').astype(int)

# Create system association features
df['is_hydronic'] = df['text_features'].str.contains('water|hydronic|pump').astype(int)
df['is_air_system'] = df['text_features'].str.contains('duct|air|cfm').astype(int)
df['is_refrigerant'] = df['text_features'].str.contains('refrigerant|dx|compressor').astype(int)
```

### 5. Derived Hierarchical Features

```python
# Create system hierarchy features
df['system_level'] = df.apply(
    lambda x: 'central_plant' if any(term in x['Precon System'].lower()
                                   for term in ['plant', 'central', 'main'])
             else ('distribution' if any(term in x['text_features']
                                       for term in ['pump', 'pipe', 'duct', 'distribution'])
                  else 'terminal_unit'),
    axis=1
)

# Create service life category features
df['replacement_category'] = pd.cut(
    df['Service Life'].astype(float),
    bins=[0, 10, 15, 20, 25, 100],
    labels=['short', 'medium-short', 'medium', 'medium-long', 'long']
)
```

## Feature Selection Techniques for HVAC Classification

### 1. Domain Knowledge Based Selection

```python
# Select features based on HVAC engineering knowledge
primary_features = [
    'Equipment_Category', 'Sub System Type', 'is_air_handler',
    'is_cooling', 'is_heating', 'normalized_size', 'system_level'
]
```

### 2. Correlation Analysis

```python
# Identify correlated features that might be redundant
correlation_matrix = df[numerical_features].corr()

# Visualize correlations
import seaborn as sns
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix for HVAC Equipment')
plt.tight_layout()
```

### 3. Feature Importance Analysis

```python
# Use Random Forest to determine feature importance
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Extract feature importances
importances = rf.feature_importances_
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("Top 10 features for HVAC classification:")
print(feature_importances.head(10))
```

## HVAC-Specific Dimensionality Reduction

```python
# Group related HVAC features together using PCA
from sklearn.decomposition import PCA

# Apply PCA to numeric features
pca = PCA(n_components=5)  # Reduce to 5 principal components
X_pca = pca.fit_transform(X_numeric)

# Examine how much variance is explained by each component
explained_variance = pca.explained_variance_ratio_
print("Variance explained by PCA components:", explained_variance)
```

## Special Considerations for HVAC Equipment

1. **System Integration**: HVAC equipment rarely operates in isolation, so
   features capturing system relationships are crucial

2. **Operating Environment**: Features that indicate where and how the equipment
   is installed provide important classification signals

3. **Energy Source**: The energy source (electric, gas, steam) is a critical
   differentiator for HVAC equipment

4. **Age-Dependent Features**: Equipment terminology changes over time, so
   capturing vintage-specific terms helps classification

5. **Regulatory Considerations**: Features that capture compliance with codes
   and standards can help differentiate equipment classes

By implementing these feature engineering techniques specifically tailored for
HVAC equipment, classification models can achieve significantly higher accuracy,
especially for specialized equipment that might otherwise fall into "Other"
categories.

---

## Data Clean Up

Whether you should add more columns depends entirely on whether those columns
contain **useful, consistently formatted information** that can help your model
distinguish between different equipment classes. Merely adding extra columns for
the sake of it can introduce noise and confusion. However, adding columns that
capture meaningful features often improves model performance. Here are some
guidelines to consider:

---

## 1. Add Columns If They Encode Useful, Distinguishing Information

- **Relevant Numeric Features**: Do you have columns such as:

  - Rated capacity (e.g., GPM, HP, MBH)?
  - Physical dimensions (e.g., length, diameter, clearance)?
  - Date of manufacture or last inspection?

  If these are **accurate** and **consistently populated**, they can help the
  model learn. For instance, if “Fan Motor HP” or “Chiller Tons” are good
  indicators of a specific equipment category, that’s a strong feature to
  include.

- **Relevant Categorical Features**: Do you have columns that provide extra
  detail about the item’s environment or usage?
  - “Installation Type” (e.g., floor-mounted vs. wall-mounted).
  - “Manufacturer” or “Brand” (only if brand strongly correlates with type).
  - “Facility Type” (e.g., hospital vs. office) — sometimes relevant if certain
    items only appear in certain facilities.

Adding those columns can help the model separate, for example, “small
wall-mounted fans for offices” from “large floor-mounted fans in industrial
plants.”

---

## 2. Avoid Columns That Are Duplicative or Low-Quality

- **Duplicates**: If you already have “BoilerSize” and “BoilerCapacity,” but
  both columns store exactly the same numeric data in different forms, combining
  them into one might be simpler.
- **Unreliable Data**: If you have columns with large amounts of missing data or
  inconsistent text, they can muddy the model’s understanding.
- **Rare / Noisy Columns**: If a column is free-form text for internal notes
  (“Spoke with onsite tech…”) and only populated sometimes, it often adds more
  noise than value.

---

## 3. Consider the Effort vs. Benefit

- **Cost of Data Wrangling**: Adding columns means more data preparation,
  cleaning, and maintenance. Will these columns be reliably updated going
  forward?
- **Model Complexity**: Each new column becomes a new feature. More features can
  improve performance if they’re truly relevant—but can also lead to overfitting
  or higher computational load if too many features are largely random or empty.

---

## 4. Example: Helpful vs. Unhelpful Columns

- **Helpful**:

  - **Installation Location**: If “ceiling-mounted” vs. “floor-mounted” is
    crucial to differentiate a type of HVAC unit, that is definitely worth
    adding.
  - **Power Rating**: HP or kW can directly correlate with category for fans,
    pumps, etc.
  - **Fluid Type**: If it’s hot water vs. chilled water vs. condenser water,
    that’s highly relevant to how we classify pumps and exchangers.

- **Unhelpful**:
  - **Free-Text “Notes”**: If it’s unstructured, rarely populated, or mostly
    placeholders, it can add more confusion than clarity.
  - **Internal ID**: e.g., “Equipment ID #3456,” which doesn’t convey real
    domain meaning.
  - **Date/Time columns** with random timestamps (unless you’re specifically
    modeling time-based patterns).

---

## 5. Use Domain Knowledge to Decide

Ultimately, the best way to decide about adding columns is to:

1. **Talk to domain experts** who know the equipment well. Ask, “Does a piece of
   data (column) clearly separate one category from another?”
2. **Test it**: If the column might be valuable, try it in your pipeline. Keep
   careful track of your evaluation metrics (accuracy, F1, etc.) with and
   without that column.

If you see a notable performance gain and the data is reliable, **keep** it. If
it adds complexity but yields little improvement, **discard** it.

---

### Bottom Line

> **Add columns when they provide clear, consistent, and distinguishing
> information about the equipment.**  
> **Remove or avoid columns that are duplicate, mostly empty, or do not
> contribute to differentiating your target categories.**

---

In the current code, **MasterFormat** codes are generated by a hard-coded
dictionary in the function `enhanced_masterformat_mapping()`. This means if you
want the model to return a broader or different list of MasterFormat codes, you
either need to:

1. **Expand the dictionaries** in `enhanced_masterformat_mapping()` with
   additional keys and values for the new categories you want.
2. **Replace the hard-coded dictionaries** with a **lookup from a file or
   database** containing all MasterFormat codes.

Below is an overview of each approach, along with sample code snippets.

---

## 1. Expanding the Hard-Coded Dictionary

If you only have a few more categories to add, the simplest option is to
**extend** the existing dictionaries in your Python code:

```python
primary_mapping = {
    'H': {
        'Chiller Plant': '23 64 00',       # Commercial Water Chillers
        'Cooling Tower Plant': '23 65 00', # Cooling Towers
        # ...
        # ADD MORE KEYS HERE:
        'New HVAC System': '23 99 00',     # Example new MasterFormat
    },
    'P': {
        # ...
    },
    # ...
}

equipment_specific_mapping = {
    # ...
    'New Specialty Equipment': '23 57 19',   # Example new subcategory code
    # ...
}
```

That way, the `enhanced_masterformat_mapping()` function will be aware of your
new MasterFormat codes:

```python
def enhanced_masterformat_mapping(
    uniformat_class: str,
    system_type: str,
    equipment_category: str,
    equipment_subcategory: Optional[str] = None
) -> str:
    # (Same as before)

    # Then you look up uniformat_class -> system_type -> code in primary_mapping
    # or, if you have a subcategory match in equipment_specific_mapping, it returns that code.
    # ...
```

### Pros & Cons

- **Pros:** Easy to do for small additions. No external files required.

- **Cons:** If you have **lots** of MasterFormat entries, your Python dictionary
  can get very large. It’s also more error-prone to maintain in code.

---

## 2. Lookup from an External Resource (CSV, JSON, DB)

If you need to store **all** available MasterFormat categories, or if you want
to maintain them outside your Python code, a better approach is to read from a
file or database. For example:

1. **Store all MasterFormat mappings** in a CSV or JSON file.
2. **Read** that CSV/JSON at runtime into a dictionary or Pandas DataFrame.
3. In `enhanced_masterformat_mapping()`, use your DataFrame/dictionary to do the
   lookup.

### Example: Reading a CSV for MasterFormat Mapping

Suppose you have a CSV file called **`masterformat_mapping.csv`** with columns:

```
Uniformat_Class,System_Type,Equipment_Subcategory,MasterFormat_Code
H,Chiller Plant,,23 64 00
H,Cooling Tower Plant,,23 65 00
P,Domestic Water Plant,,22 11 00
,Heat Exchanger,Heat Exchanger,23 57 00
... etc ...
```

(Or whatever columns make sense to you. The point is: you store your mapping as
data.)

Then, in Python:

```python
import pandas as pd

def load_masterformat_mapping(csv_path="masterformat_mapping.csv"):
    # Example of reading your mapping file
    mapping_df = pd.read_csv(csv_path)
    return mapping_df

def enhanced_masterformat_mapping(
    uniformat_class: str,
    system_type: str,
    equipment_category: str,
    equipment_subcategory: Optional[str] = None
) -> str:
    """
    Looks up the MasterFormat code from an external CSV-based DataFrame.
    """
    # 1. Load the mapping DataFrame (you might do this once at startup rather than every time)
    #    and store it in a global variable or pass it in as an argument
    global masterformat_df

    # 2. Filter the DataFrame by uniformat_class, system_type, subcategory, etc.
    #    This is just an example; you'll need logic that matches your CSV layout
    subset = masterformat_df[
        (masterformat_df['Uniformat_Class'] == uniformat_class) &
        (masterformat_df['System_Type'] == system_type) &
        (masterformat_df['Equipment_Subcategory'] == equipment_subcategory)
    ]

    # 3. If you find a matching row, return its MasterFormat_Code
    if len(subset) > 0:
        return subset['MasterFormat_Code'].iloc[0]

    # 4. Otherwise, fallback
    return '00 00 00'
```

You would then **initialize** `masterformat_df = load_masterformat_mapping()`
**once** when your script starts, so that your `enhanced_masterformat_mapping()`
can reference it.

### Pros & Cons

- **Pros**:

  - You can maintain your MasterFormat data as _data_, rather than code.
  - Easier to keep a large list of MasterFormat lines cleanly in CSV.
  - No code changes needed if you add more lines—just update the CSV.

- **Cons**:
  - Slightly more complex in code because you have to load a CSV or database.
  - You need a consistent “join” or “lookup” logic (and you have to handle
    partial matches, missing fields, etc.).

---

## 3. Hybrid Approach

In many real-world situations, you end up with a **mixed** approach:

- **Big** core list of MasterFormat lines in a CSV or DB.
- A few **custom overrides** or “special cases” kept in Python code (like
  `equipment_specific_mapping`) because they’re less straightforward.

You can combine these approaches by:

1. Loading the main CSV.
2. Checking if the user’s `equipment_subcategory` is in a “special cases”
   dictionary.
3. If not found, then look in the CSV DataFrame.

---

## Summary

- **Right now**, MasterFormat codes are in a short dictionary in
  `enhanced_masterformat_mapping()`.
- **To feed it more or “all” MasterFormat categories**, you either expand that
  dictionary _by hand_, or read from an external resource (CSV, JSON, database)
  and do a **lookup** in your Python code.

Either way, the key point is the model still **predicts** your UniFormat and
Equipment columns from the CSV data, and _then_ you map those predictions to
MasterFormat codes in your `enhanced_masterformat_mapping()` function. All
MasterFormat expansions ultimately flow from that function’s logic.

---

### Short Answer

- **Yes**, the code already deals with **UniFormat** (see the column
  `Uniformat_Class` in the CSV) **and** it uses a **MasterFormat** mapping.

* If you want to **also** use **OmniClass** (or any other classification system)
  in the same pipeline, you can add a **similar mapping** function or a
  **dictionary/CSV** for OmniClass codes, just like we do for MasterFormat.

---

### Longer Explanation

#### 1. The Code Already Uses UniFormat

If you look at the CSV columns:

```text
Uniformat_Class,System_Type, ...
```

and the part of the code that references it:

```python
# e.g. from the CSV
df['Uniformat_Class'] = df['System Type ID']
```

You can see that **UniFormat** is already captured as one of the target columns
in the multi-output model:

```python
y = df[[
    'Equipment_Category',
    'Uniformat_Class',
    'System_Type',
    'Equipment_Type',
    'System_Subtype'
]]
```

So your model is predicting a value for `Uniformat_Class` in addition to
`Equipment_Category` (and a few other fields).

#### 2. The Code Has a Hard-Coded MasterFormat Mapping

Separately, there is a function called `enhanced_masterformat_mapping()` that
takes in
`(uniformat_class, system_type, equipment_category, equipment_subcategory)` and
returns a **MasterFormat** code. That function is just a big dictionary of “if
you see this Uniformat plus that System Type, then the MasterFormat code is X.”

That’s purely a **post-processing** step. It uses the model’s **predicted**
`Uniformat_Class` and `System_Type` (etc.) and **translates** them to a
MasterFormat number.

#### 3. Adding OmniClass

If you also need an **OmniClass** code, you have a couple of options:

- **(A) A Separate Mapping Function**  
  Create a function `enhanced_omniclass_mapping()` that, just like
  `enhanced_masterformat_mapping()`, uses a dictionary or CSV to look up
  `(uniformat_class, system_type, etc.)` → `OmniClass code`.

  ```python
  def enhanced_omniclass_mapping(
      uniformat_class: str,
      system_type: str,
      equipment_category: str,
      equipment_subcategory: Optional[str] = None
  ) -> str:
      """
      Looks up an OmniClass classification code based on the predicted columns.
      """
      # Could be a dictionary just like enhanced_masterformat_mapping()
      # OR read from an external CSV/JSON
      ...
      return omniclass_code
  ```

- **(B) Fold It into the Same Function**  
  You could also make a single function that returns **both** MasterFormat and
  OmniClass codes at once. For example:

  ```python
  def enhanced_classification_mapping(
      uniformat_class: str,
      system_type: str,
      equipment_category: str,
      equipment_subcategory: Optional[str] = None
  ) -> Tuple[str, str]:
      """
      Return both the MasterFormat and OmniClass code as a tuple
      (masterformat_code, omniclass_code).
      """
      masterformat_code = ...
      omniclass_code = ...
      return masterformat_code, omniclass_code
  ```

Either way, you’d store or compute your OmniClass code from the same predicted
columns that drive MasterFormat.

#### 4. Why So Many Classification Systems?

- **UniFormat** is a system primarily for building elements (walls, roofs, MEP
  systems, etc.)—that’s why we have `Uniformat_Class` in the CSV.

* **MasterFormat** is a system for **specifications**, especially for
  construction documents.
* **OmniClass** is a broader classification framework that can include built
  environment elements, phases, properties, etc.

In practice, an asset (like a piece of HVAC equipment) might need to be labeled
with **all** of them if you’re integrating with different
design/construction/maintenance software.

---

### Summary

- You **already** have UniFormat predictions in the code.

* You **already** have MasterFormat output via a mapping function.
* To add **OmniClass**, you can write a **similar mapping** (dictionary, CSV, or
  database) that says: “Given my predicted columns (`Uniformat_Class`, etc.),
  the OmniClass code is X.” Then call it after the model predicts.

---

Below is a **sample template** you can use as a guide for a “clean” CSV that
feeds into your classification pipeline.  
It shows _typical_ columns that work well with your code—and what kind of data
they should contain to align with your **Uniformat** + **MasterFormat**
approach.

---

## 1. Example CSV Column Layout

| Uniformat_Class | System_Type          | Equipment_Category | Equipment_Subcategory | combined_features                                               | Equipment_Size | Unit | Service_Life | ... (any other columns) ... |
| --------------- | -------------------- | ------------------ | --------------------- | --------------------------------------------------------------- | -------------- | ---- | ------------ | --------------------------- |
| H               | Steam Boiler Plant   | Accessory          | Blow Down             | Accessory Blow Down Floor Mounted 100 GPM Steam Boiler ... etc. | 100            | GAL  | 20           | ...                         |
| H               | Steam Boiler Plant   | Boiler             | Packaged Fire Tube    | Boiler Steam Packaged Fire Tube 3000 MBH ... etc.               | 3000           | MBH  | 25           | ...                         |
| P               | Domestic Water Plant | Water Softener     | Duplex System         | Water Softener Duplex ... etc.                                  | 50             | GPM  | 15           | ...                         |
| H               | Cooling Tower Plant  | Accessory          | Basket Strainer       | Accessory Basket Strainer 8 INCHES Cooling Tower ... etc.       | 8              | INCH | 25           | ...                         |

### Column-by-column Explanation

1. **`Uniformat_Class`**

   - A short code that identifies the Uniformat classification for the system.
   - Examples:
     - `H` = HVAC
     - `P` = Plumbing
     - `R` = Refrigeration
     - You can also store longer Uniformat codes if needed (like `D2020`,
       `F1030`, etc.).
   - **Important**: This should truly be a correct Uniformat code, or at least a
     simplified stand-in that you use consistently across your dataset.

2. **`System_Type`**

   - A descriptive field for the major mechanical or plumbing system.
   - Examples: “Steam Boiler Plant,” “Cooling Tower Plant,” “Domestic Water
     Plant,” “Air Handling Units,” etc.
   - Usually a free-text field, but you want to keep it fairly consistent so you
     can map it to MasterFormat or other codes.

3. **`Equipment_Category`**

   - The top-level type of equipment.
   - Examples: “Boiler,” “Chiller,” “Fan,” “Accessory,” “Pump,” “Water
     Softener,” etc.
   - This is used in your pipeline code (like `df['Equipment_Category']`).

4. **`Equipment_Subcategory`**

   - A more detailed subtype of your `Equipment_Category`.
   - Examples: “Blow Down,” “Floor Mounted,” “Packaged Fire Tube,” “Basket
     Strainer,” etc.
   - This is helpful for building hierarchical classification or “Other”
     categories.

5. **`combined_features`**

   - A concatenation of textual fields that describe the equipment.
   - Typically includes columns like “Title,” “Sub System Type,” “Sub System
     ID,” plus any other descriptive fields you want the TF–IDF vectorizer to
     ingest.
   - Example content for a steam blow-down tank might be:
     ```
     "Accessory Blow Down Floor Mounted 100 GPM Steam Boiler Condensate"
     ```
   - The pipeline uses this field for _text-based_ classification signals.

6. **`Equipment_Size`** (or `size_feature`)

   - A numeric or string field for the size/capacity.
   - Examples: 100.0, 25.0, 1000.0, etc.
   - If you store numeric values, you can scale them. If you store “100 GPM,”
     you might want to parse out the numeric portion.

7. **`Unit`**

   - The engineering units for that piece of equipment.
   - Examples: “GPM,” “MBH,” “TONS,” “HP,” “INCHES,” “GAL,” etc.
   - In your code, you might or might not feed this directly into the numeric
     pipeline. Some organizations store it in `combined_features` for text
     usage.

8. **`Service_Life`**

   - The numeric field used in your pipeline’s numeric feature pipeline.
   - Example: 20 (meaning 20 years).
   - If you don’t have a real service-life estimate, you can default to 0 or
     some fallback.

9. **Other Columns**
   - You can keep other columns like `Trade`, `Precon Tag`, `Operations System`,
     etc., if they’re helpful.
   - Some might be fed into `combined_features`, others might be purely
     reference columns.

---

## 2. How This Aligns With Your Code

- **`df['Uniformat_Class']`**: This is what you actually want stored in
  `df['Uniformat_Class']`, rather than `Steam Boiler Plant` or
  `Cooling Tower Plant`.
- **`df['System_Type']`**: Free text describing the system, e.g., “Steam Boiler
  Plant.”
- **`df['Equipment_Category']`**: The top-level category (like “Accessory,”
  “Boiler,” “Pump,” etc.).
- **`df['Equipment_Subcategory']`**: The next-level detail (like “Blow Down,”
  “Floor Mounted,” “Packaged Fire Tube,” “Basket Strainer,” etc.).
- **`df['combined_features']`**: String concatenation of the other text columns
  you want the model to read.

This structure ensures that:

- Your pipeline can train a multi-output classifier for (1)
  `Equipment_Category`, (2) `Uniformat_Class`, (3) `System_Type`, and so on.
- The text-based columns (`combined_features`) still contain enough descriptive
  text to identify “Blow Down vs. Basket Strainer vs. Polishing System,” etc.
- The numeric pipeline can scale `Service_Life` if you have it.

---

## 3. Example CSV Snippet (Very Simplified)

Below is a tiny snippet you could place into a CSV:

```csv
Uniformat_Class,System_Type,Equipment_Category,Equipment_Subcategory,combined_features,Equipment_Size,Unit,Service_Life
H,Steam Boiler Plant,Accessory,Blow Down,"Accessory Blow Down Floor Mounted 100 GAL Steam Boiler Plant",100,GAL,20
H,Cooling Tower Plant,Accessory,Basket Strainer,"Accessory Basket Strainer 8 INCHES Cooling Tower Plant",8,INCH,25
P,Domestic Water Plant,Water Softener,Duplex,"Water Softener Duplex System Domestic Water 50 GPM",50,GPM,15
H,Steam Boiler Plant,Boiler,Packaged Fire Tube,"Boiler Steam Packaged Fire Tube 3000 MBH Some additional text",3000,MBH,25
```

- Notice how **`Uniformat_Class`** is a short code (H or P).
- **`System_Type`** might be “Steam Boiler Plant” or “Cooling Tower Plant,” etc.
- **`Equipment_Category`** is “Accessory,” “Boiler,” “Water Softener,” etc.
- **`Equipment_Subcategory`** is “Blow Down,” “Basket Strainer,” or “Packaged
  Fire Tube.”
- **`combined_features`** is the textual combination of the interesting
  descriptors.
- **`Equipment_Size`** is a numeric value, and **`Unit`** is a string.

---

## 4. Mapping From “Raw” System Names to “Uniformat_Class”

If in your real data you have “Steam Boiler Plant,” “Cooling Tower Plant,” etc.,
you can keep them in **`System_Type`** but do a short dictionary lookup to
assign them to `'H'` or `'P'` for your final **`Uniformat_Class`** column. For
example:

```python
uniformat_mapping = {
    "Steam Boiler Plant": "H",
    "Cooling Tower Plant": "H",
    "Domestic Water Plant": "P",
    "Medical/Lab Gas Plant": "P",
    "Rooftop Unit": "H",
    # ...
}

df["Uniformat_Class"] = df["System_Type"].map(uniformat_mapping)
```

---

## Takeaway

**Give each piece of equipment the correct Uniformat code in a single dedicated
column**, and keep the “friendly name” (like “Steam Boiler Plant”) in its own
`System_Type` column.  
This ensures your classification pipeline can **cleanly** learn from the text
(like “steam” or “boiler” in the combined_features) while also letting you do
direct lookups or mapping for MasterFormat, OmniClass, etc.
