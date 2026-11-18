# -*- coding: utf-8 -*-
"""analysis.py

Data loading, cleaning, summary statistics and static charts for the
CA4 Capstone Project: Real Estate Property Price Analytics (Bengaluru).
Exported from Google Colab.

Input : Bengaluru_House_Data.csv
        (Kaggle: https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)
Output: Bengaluru_House_Data_Cleaned.xlsx (read by app.py) and chart1-chart4 PNG files

Run   : python analysis.py
"""

import numpy as np
import pandas as pd

# 1. Load data
df = pd.read_csv('Bengaluru_House_Data.csv')
print('Raw dataset:', df.shape)

# 2. Drop duplicates
df = df.drop_duplicates().reset_index(drop=True)
print('After removing duplicates:', len(df))

# 3. Drop unusable/sparse columns and essential nulls
df = df.drop(columns=['society'])
df = df.dropna(subset=['location', 'size']).reset_index(drop=True)
print('After dropping missing location/size:', len(df))

# 4. Extract numeric BHK
df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]))
df = df.drop(columns=['size'])


# 5. Clean total_sqft
def parse_sqft(x):
  tokens = str(x).split('-')
  if len(tokens) == 2:
    try:
      return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2.0
    except:
      return None
  try:
    return float(x)
  except:
    return None


df['total_sqft'] = df['total_sqft'].apply(parse_sqft)
df = df.dropna(subset=['total_sqft']).reset_index(drop=True)
print('After parsing total_sqft and dropping land units:', len(df))

# 6. Impute missing bathrooms and balconies
df['bath'] = df.groupby('bhk')['bath'].transform(lambda x: x.fillna(x.median()))
df['balcony'] = df['balcony'].fillna(df['balcony'].median())

# 7. Calculate price per square foot (Price is in Lakhs, * 100,000)
df['price_per_sqft'] = (df['price'] * 100000.0) / df['total_sqft']

# 8. Clean and consolidate locations (count <= 10 -> 'Other')
df['location'] = df['location'].str.strip()
loc_counts = df['location'].value_counts()
infrequent_locs = loc_counts[loc_counts <= 10].index
df['location'] = df['location'].apply(
    lambda x: 'Other' if x in infrequent_locs else x
)

# 9. Outlier filtering
df = df[df['total_sqft'] / df['bhk'] >= 300].copy()
print('After minimum space filter (sqft/BHK >= 300):', len(df))
df = df[df['bath'] <= (df['bhk'] + 2)].copy()
print('After bathroom plausibility filter (bath <= BHK + 2):', len(df))
df_clean = df.reset_index(drop=True)

# 10. Export to Excel
df_clean.to_excel('Bengaluru_House_Data_Cleaned.xlsx', index=False)
print('Export complete! Bengaluru_House_Data_Cleaned.xlsx is ready.')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

# Load the cleaned dataset
df = pd.read_excel('Bengaluru_House_Data_Cleaned.xlsx')

# -------------------------------------------------------------
# 1. Summary Statistics Table
# -------------------------------------------------------------
print('--- Numerical Summary Statistics ---')
print(
    df[
        ['price', 'total_sqft', 'bhk', 'bath', 'balcony', 'price_per_sqft']
    ].describe()
)

# -------------------------------------------------------------
# 2. Chart 1: Distribution of Property Prices (Univariate)
# -------------------------------------------------------------
plt.figure(figsize=(9, 4))
sns.histplot(df['price'], bins=50, kde=True, color='royalblue')
plt.title('Distribution of Property Prices (in ₹ Lakhs)', fontsize=13)
plt.xlabel('Price (₹ Lakhs)')
plt.ylabel('Listing Count')
plt.xlim(0, 500)  # Zoom in to the bulk distribution for clarity
plt.tight_layout()
plt.savefig('chart1_price_distribution.png')
plt.show()

# -------------------------------------------------------------
# 3. Chart 2: Total Square Feet vs. Price (Bivariate)
# -------------------------------------------------------------
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df[df['total_sqft'] <= 5000],
    x='total_sqft',
    y='price',
    hue='bhk',
    palette='viridis',
    alpha=0.6,
)
plt.title(
    'Property Price vs. Square Footage (Segmented by BHK)', fontsize=13
)
plt.xlabel('Total Square Feet')
plt.ylabel('Price (₹ Lakhs)')
plt.tight_layout()
plt.savefig('chart2_sqft_vs_price.png')
plt.show()

# -------------------------------------------------------------
# 4. Chart 3: Top 10 Localities by Listing Volume & Median Price
# -------------------------------------------------------------
top_locs = df[df['location'] != 'Other']['location'].value_counts().head(10).index
loc_summary = (
    df[df['location'].isin(top_locs)]
    .groupby('location')['price_per_sqft']
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))
sns.barplot(x=loc_summary.values, y=loc_summary.index, palette='mako')
plt.title(
    'Median Price per Sq. Ft. Across Top 10 High-Demand Locations', fontsize=13
)
plt.xlabel('Median Price per Sq. Ft. (₹)')
plt.ylabel('Micro-Market Location')
plt.tight_layout()
plt.savefig('chart3_top_locations_pps.png')
plt.show()

# -------------------------------------------------------------
# 5. Chart 4: Correlation Heatmap (Multivariate)
# -------------------------------------------------------------
plt.figure(figsize=(8, 5))
corr_cols = ['price', 'total_sqft', 'bhk', 'bath', 'balcony', 'price_per_sqft']
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='Blues', fmt='.2f', cbar=True)
plt.title('Correlation Matrix of Housing Attributes', fontsize=13)
plt.tight_layout()
plt.savefig('chart4_correlation_heatmap.png')
plt.show()
