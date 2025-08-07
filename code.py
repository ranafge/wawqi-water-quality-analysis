# Calculate Weighted Arithmetic Water Quality Index (WAWQI) for Research.csv and output summary
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = 'Research.csv'
df = pd.read_csv(file_path, encoding = 'ascii')

# Strip column names
df.columns = [c.strip() for c in df.columns]

# Define standards (Si) typical WHO guidelines or common limits
standards = {
    'pH (pH)': 8.5,                 # upper permissible limit
    'Conductivity (?S/cm)': 2500,    # µS/cm
    'TDS (ppmTDS)': 1000,           # mg/L
    'Salinity (PSU)': 0.5,          # practical salinity units approximated limit
    'Temperature': 25,              # °C desirable
    'Total Iron (mg/L)': 0.3,       # mg/L
    'Chloride (mg/L)': 250,         # mg/L
    'Turbidity (NTU)': 5,           # NTU
    'Total Hardness (mg/L)': 300,   # mg/L
    'Mg Hardness (mg/L)': 100,      # mg/L
    'Ca Hardness (mg/L)': 200,      # mg/L
    'TABC (CFU)': 500,              # colony forming units /mL
    'TC (CFU)': 1,                  # ideally 0, set 1 to avoid division by zero
    'FC (CFU)': 1,                  # ideally 0
    'E.Coli': 1                     # ideally 0
}

# Ensure all needed columns exist; if not, warn
missing_cols = [col for col in standards if col not in df.columns]
if missing_cols:
    print('Warning: Missing columns in dataset:', missing_cols)

# Compute K constant
import numpy as np
K_denominator = sum([1/standards[col] for col in standards])
K = 1 / K_denominator

# Calculate weight Wi for each parameter
weights = {col: K/standards[col] for col in standards}

# Function to compute Qi per value

def compute_qi(value, col):
    if pd.isna(value):
        return np.nan
    Vi = value
    Si = standards[col]
    if col == 'pH (pH)':
        V0 = 7.0
    else:
        V0 = 0.0
    Qi = (Vi - V0) / (Si - V0) * 100
    if Qi < 0:
        Qi = 0
    return Qi

# Compute WAWQI per sample
wawqi_values = []
for idx, row in df.iterrows():
    Qi_list = []
    WiQi_list = []
    Wi_list = []
    for col in standards:
        if col in df.columns:
            qi = compute_qi(row[col], col)
            wi = weights[col]
            if not np.isnan(qi):
                Qi_list.append(qi)
                WiQi_list.append(wi * qi)
                Wi_list.append(wi)
    if Wi_list:
        wawqi = sum(WiQi_list) / sum(Wi_list)
    else:
        wawqi = np.nan
    wawqi_values.append(wawqi)

# Add to dataframe
df['WAWQI'] = wawqi_values

# Classification
def classify_wawqi(val):
    if pd.isna(val):
        return 'No data'
    if val <= 25:
        return 'Excellent'
    elif val <= 50:
        return 'Good'
    elif val <= 75:
        return 'Poor'
    elif val <= 100:
        return 'Very poor'
    else:
        return 'Unsuitable'

df['Quality Status'] = df['WAWQI'].apply(classify_wawqi)

# Show head of results
print(df[['Sample Serial No ', 'WAWQI', 'Quality Status']].head())

# Bar chart of WAWQI per sample
plt.figure(figsize=(10,6))
sns.barplot(x = 'Sample Serial No ', y = 'WAWQI', data = df, palette = 'viridis')
plt.xticks(rotation = 90)
plt.title('WAWQI per Sample')
plt.ylabel('WAWQI')
plt.xlabel('Sample Serial No')
plt.tight_layout()
plt.show()

# Create summary excel of range quality counts
quality_counts = df['Quality Status'].value_counts().reset_index()
quality_counts.columns = ['Quality Status', 'Count']

output_file = 'WAWQI_quality_summary.xlsx'
quality_counts.to_excel(output_file, index = False)
print('Saved summary to', output_file)





Code Explanation
This code calculates the Weighted Arithmetic Water Quality Index (WAWQI) from water quality data, classifies the quality status, and outputs a summary to an Excel file.
- Load water quality data from a CSV file.
- Define water quality standards based on WHO guidelines.
- Compute a constant K and weights for each parameter.
- Calculate the quality index (Qi) for each parameter in the dataset.
- Compute the WAWQI for each sample and classify the quality status.
- Visualize WAWQI results with a bar chart.



TypeError
Cell In[2], line 69
67 for col in standards:
68
if col in df.columns:
Traceback (most recent call last)
---> 69
70
71
qi = compute_qi (row[col], col)
wi = weights [col]
if not np.isnan(qi):
Cell In[2], line 56, in compute_qi(value, col)
54 else:
VO = 0.0
55
---> 56 Qi
=
(Vi - Vo) / (Si - VØ) * 100
57 if Qi < 0:
58
Qi = 0
TypeError: unsupported operand type(s) for -: 'str' and 'float'
Also, evaluate what went wrong and consider options for how to fix it. Break the execution up into cell blocks




# Clean numeric columns and recompute WAWQI with error fixed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = 'Research.csv'
df = pd.read_csv(file_path, encoding = 'ascii')

# Strip column names
df.columns = [c.strip() for c in df.columns]

# List of parameter columns (intersection with standards)
standards = {
    'pH (pH)': 8.5,
    'Conductivity (?S/cm)': 2500,
    'TDS (ppmTDS)': 1000,
    'Salinity (PSU)': 0.5,
    'Temperature': 25,
    'Total Iron (mg/L)': 0.3,
    'Chloride (mg/L)': 250,
    'Turbidity (NTU)': 5,
    'Total Hardness (mg/L)': 300,
    'Mg Hardness (mg/L)': 100,
    'Ca Hardness (mg/L)': 200,
    'TABC (CFU)': 500,
    'TC (CFU)': 1,
    'FC (CFU)': 1,
    'E.Coli': 1
}

# Convert parameter columns to numeric
for col in standards:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')

# Compute K and weights
K_denominator = sum([1/standards[col] for col in standards])
K = 1 / K_denominator
weights = {col: K/standards[col] for col in standards}

# Function compute_qi

def compute_qi(value, col):
    if pd.isna(value):
        return np.nan
    Vi = value
    Si = standards[col]
    V0 = 7.0 if col == 'pH (pH)' else 0.0
    Qi = (Vi - V0) / (Si - V0) * 100
    if Qi < 0:
        Qi = 0
    return Qi

# Calculate WAWQI per sample
wawqi_vals = []
for _, row in df.iterrows():
    WiQi_sum = 0
    Wi_sum = 0
    for col in standards:
        if col in df.columns:
            qi = compute_qi(row[col], col)
            wi = weights[col]
            if not np.isnan(qi):
                WiQi_sum += wi * qi
                Wi_sum += wi
    wawqi_vals.append(WiQi_sum / Wi_sum if Wi_sum != 0 else np.nan)

df['WAWQI'] = wawqi_vals

# Classification

def classify(val):
    if pd.isna(val):
        return 'No data'
    if val <= 25:
        return 'Excellent'
    elif val <= 50:
        return 'Good'
    elif val <= 75:
        return 'Poor'
    elif val <= 100:
        return 'Very poor'
    else:
        return 'Unsuitable'

df['Quality Status'] = df['WAWQI'].apply(classify)

# Display head
print(df[['Sample Serial No ', 'WAWQI', 'Quality Status']].head())

# Bar chart
plt.figure(figsize = (10,6))
sns.barplot(x = 'Sample Serial No ', y = 'WAWQI', data = df, palette = 'viridis')
plt.xticks(rotation = 90)
plt.title('Weighted Arithmetic Water Quality Index (WAWQI)')
plt.ylabel('WAWQI')
plt.xlabel('Sample Serial No')
plt.tight_layout()
plt.show()

# Summary excel
summary = df['Quality Status'].value_counts().reset_index()
summary.columns = ['Quality Status', 'Count']
summary_file = 'WAWQI_quality_summary.xlsx'
summary.to_excel(summary_file, index = False)
print('Summary saved to', summary_file)


Code Explanation
This code calculates the Weighted Arithmetic Water Quality Index (WAWQI) for water samples based on various parameters and generates a summary of the quality status.
- Load water quality data from a CSV file.
- Clean and convert relevant columns to numeric values.
- Define standards for water quality parameters.
- Compute weights for each parameter based on standards.
- Calculate WAWQI for each sample using a defined formula.
- Classify the quality status based on WAWQI values.
- Visualize WAWQI results with a bar chart.
- Generate and save a summary of quality status counts to an Excel file.


KeyError
Cell In[3], line 91
Traceback (most recent call last)
88 df['Quality Status'] = df['WAWQI'].apply(classify)
90 # Display head
91 print(df[['Sample Serial No', 'WAWQI', 'Quality Status']].head())
93 # Bar chart
94 plt.figure(figsize = (10,6))
File /opt/conda/lib/python3.11/site-packages/pandas/core/frame.py:4108, in DataFrame._getitem__(self, key) 4106
4107
-> 4108
if is_iterator(key):
key = list(key)
indexer = self.columns._get_indexer_strict(key, "columns")[1]
4110 # take() does not accept boolean indexers
4111 if getattr(indexer, "dtype", None) == bool:
File /opt/conda/lib/python3.11/site-packages/pandas/core/indexes/base.py:6200, in Index._get_indexer_strict(self, key, axis_name)
6197 else:
6198 keyarr, indexer, new_indexer = self._reindex_non_unique (keyarr)



# Recompute with corrected column name display
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv('Research.csv', encoding = 'ascii')
df.columns = [c.strip() for c in df.columns]

standards = {'pH (pH)':8.5,'Conductivity (?S/cm)':2500,'TDS (ppmTDS)':1000,'Salinity (PSU)':0.5,'Temperature':25,'Total Iron (mg/L)':0.3,'Chloride (mg/L)':250,'Turbidity (NTU)':5,'Total Hardness (mg/L)':300,'Mg Hardness (mg/L)':100,'Ca Hardness (mg/L)':200,'TABC (CFU)':500,'TC (CFU)':1,'FC (CFU)':1,'E.Coli':1}
for col in standards:
    if col in df.columns:
        df[col]=pd.to_numeric(df[col],errors='coerce')
K=1/sum([1/standards[c] for c in standards])
weights={c:K/standards[c] for c in standards}

def qi(val,col):
    if pd.isna(val):
        return np.nan
    V0=7.0 if col=='pH (pH)' else 0.0
    q=(val-V0)/(standards[col]-V0)*100
    return max(q,0)

wawqi=[]
for _,r in df.iterrows():
    num=0;den=0
    for col in standards:
        if col in df.columns:
            q=qi(r[col],col)
            w=weights[col]
            if not np.isnan(q):
                num+=w*q;den+=w
    wawqi.append(num/den if den else np.nan)

df['WAWQI']=wawqi

def cls(v):
    if pd.isna(v):return 'No data'
    if v<=25:return 'Excellent'
    elif v<=50:return 'Good'
    elif v<=75:return 'Poor'
    elif v<=100:return 'Very poor'
    else:return 'Unsuitable'

df['Quality Status']=df['WAWQI'].apply(cls)

print(df[[ 'Sample Serial No', 'WAWQI','Quality Status']].head())

plt.figure(figsize=(10,6))
sns.barplot(x='Sample Serial No',y='WAWQI',data=df,palette='viridis')
plt.xticks(rotation=90)
plt.ylabel('WAWQI')
plt.title('WAWQI per Sample')
plt.tight_layout()
plt.show()

summary=df['Quality Status'].value_counts().reset_index()
summary.columns=['Quality Status','Count']
summary.to_excel('WAWQI_quality_summary.xlsx',index=False)
print(summary)
print('Saved to WAWQI_quality_summary.xlsx')




Code Explanation
This code calculates the Water Quality Index (WAWQI) for water samples based on various parameters and generates a summary of the quality status.
- Read data from a CSV file and clean column names.
- Define water quality standards for various parameters.
- Convert relevant columns to numeric values.
- Calculate weights based on standards.
- Define a function to compute the quality index for each parameter.
- Iterate through each sample to compute the overall WAWQI.
- Classify the WAWQI into quality status categories.
- Print the results and create a bar plot of WAWQI per sample.
- Summarize the quality status counts and save to an Excel file.





