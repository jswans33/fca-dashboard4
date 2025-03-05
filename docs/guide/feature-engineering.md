# Feature Engineering for HVAC Equipment Classification

When classifying HVAC equipment, effective feature engineering is crucial for accurate model performance. Let me explain the most important features and engineering techniques specifically for HVAC equipment classification:

## Key Features for HVAC Equipment Classification

### 1. Equipment Descriptive Features

* **Equipment Type Indicators**: Terms like "pump," "boiler," "chiller," "fan" that directly indicate equipment category
* **Manufacturer-Specific Terminology**: Brand-specific model designations and terminology
* **System Service Designations**: "HVAC," "mechanical," "plumbing" classifications
* **Function Descriptors**: Terms describing what the equipment does (e.g., "cooling," "heating," "ventilation")

### 2. Technical Specifications

* **Capacity Metrics**: Size indicators like BTU, tons, horsepower, CFM
* **Dimensional Features**: Physical dimensions that may indicate equipment class
* **Energy Ratings**: Efficiency metrics like EER, SEER, COP
* **Operating Parameters**: Temperature ranges, pressure ratings
* **Refrigerant Types**: R-410A, R-22, etc., which can indicate equipment generation and type

### 3. System Integration Features

* **Connection Types**: How the equipment connects to other systems
* **Control Interface**: BMS integration capabilities
* **Mounting Configuration**: Ceiling, wall, floor, or roof mounted
* **Service Access Requirements**: Clearance needs that indicate equipment type

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

N-grams are crucial for capturing HVAC terminology that often includes multi-word technical phrases:

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

1. **System Integration**: HVAC equipment rarely operates in isolation, so features capturing system relationships are crucial

2. **Operating Environment**: Features that indicate where and how the equipment is installed provide important classification signals

3. **Energy Source**: The energy source (electric, gas, steam) is a critical differentiator for HVAC equipment

4. **Age-Dependent Features**: Equipment terminology changes over time, so capturing vintage-specific terms helps classification

5. **Regulatory Considerations**: Features that capture compliance with codes and standards can help differentiate equipment classes

By implementing these feature engineering techniques specifically tailored for HVAC equipment, classification models can achieve significantly higher accuracy, especially for specialized equipment that might otherwise fall into "Other" categories.