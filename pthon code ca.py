import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv("python dataset.csv")
print(df.head())
# Data Cleaning and Handling
df['latitude'] = df['latitude'] / 10  # example scaling if needed
df['longitude'] = df['longitude'] / 10  # example scaling if needed
df['pollutant_min'] = df['pollutant_min'] / 10  # example scaling if needed
df['pollutant_max'] = df['pollutant_max'] / 10  # example scaling if needed
df['pollutant_avg'] = df['pollutant_avg'] / 10  # example scaling if needed

# Check for missing values
print("Missing values:\n", df.isnull().sum())
# Print cleaned data shape
print("Cleaned Data Shape:", df.shape)

# Print sample of cleaned data
print("Sample Cleaned Data:\n", df.head())
print("Basic Statistics:")
print(df.describe())

# Distribution of pollutant_avg
plt.figure(figsize=(8, 5))
sns.histplot(df['pollutant_avg'], bins=20, kde=True)
plt.title('Pollutant Average Distribution')
plt.xlabel('Pollutant Average')
plt.ylabel('Frequency')
plt.show()
# Distribution of top10 average pollutant level
plt.figure(figsize=(10, 6))
top_cities = df.groupby('city')['pollutant_avg'].mean().sort_values(ascending=False).head(10)
top_cities.plot(kind='bar', color='coral')
plt.title('Top 10 Cities by Average Pollutant Level')
plt.xlabel('City')
plt.ylabel('Average Pollutant Level')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#Distribution of pollutant types
plt.figure(figsize=(5, 5))
pollutant_counts = df['pollutant_id'].value_counts()
plt.pie(pollutant_counts, labels=pollutant_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Pollutant Types')
plt.show()
#Average Pollutant Concentration vs Latitude
plt.figure(figsize=(8, 6))
plt.plot(df['latitude'], df['pollutant_avg'], 'o', alpha=0.5)
plt.title('Average Pollutant Concentration vs Latitude')
plt.xlabel('Latitude')
plt.ylabel('Average Pollutant Concentration')
plt.show()
#Correlation Heatmap of Air Quality Features
plt.figure(figsize=(10, 8))
numeric_cols = ['latitude', 'longitude', 'pollutant_min', 'pollutant_max', 'pollutant_avg']
# Clean data: Remove rows with 'NA' and convert to numeric
df_numeric = df[numeric_cols].replace('NA', pd.NA).dropna()
df_numeric = df_numeric.astype(float)
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Air Quality Features')
plt.show()

#Average Pollutant Concentration
df_clean = df[df['pollutant_avg'] != 'NA'].copy()
df_clean['pollutant_avg'] = df_clean['pollutant_avg'].astype(float)
sns.boxplot(y=df_clean['pollutant_avg'])
plt.title('Box Plot of Average Pollutant Concentration')
plt.ylabel('Average Pollutant Concentration')
plt.show()

#Filter Outliers Using IQR Method
df_clean = df[df['pollutant_avg'] != 'NA'].copy()
df_clean['pollutant_avg'] = df_clean['pollutant_avg'].astype(float)
# Calculate IQR and bounds for outliers
Q1 = df_clean['pollutant_avg'].quantile(0.25)
Q3 = df_clean['pollutant_avg'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_no_outliers = df_clean[(df_clean['pollutant_avg'] >= lower_bound) & (df_clean['pollutant_avg'] <= upper_bound)]
sns.boxplot(y=df_no_outliers['pollutant_avg'])
plt.title('Box Plot of Average Pollutant Concentration (Outliers Removed)')
plt.ylabel('Average Pollutant Concentration')
plt.show()

"""df_clean = df[df['pollutant_avg'] != 'NA'].copy()
df_clean['pollutant_avg'] = df_clean['pollutant_avg'].astype(float)

# Create a pivot table: Average pollutant concentration by state and pollutant_id
pivot_table = df_clean.pivot_table(
    values='pollutant_avg',
    index='state',
    columns='pollutant_id',
    aggfunc='mean'
)

# Create heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='Blues', cbar=True)
plt.title('Average Pollutant Concentration by State and Pollutant Type')
plt.xlabel('Pollutant Type')
plt.ylabel('State')
plt.show()"""
# Clean data: Remove 'NA' and convert to numeric
df_clean = df[df['pollutant_avg'] != 'NA'].copy()
df_clean['pollutant_avg'] = df_clean['pollutant_avg'].astype(float)

# Create a pivot table: Average pollutant concentration by state and pollutant_id
pivot_table = df_clean.pivot_table(
    values='pollutant_avg',
    index='state',
    columns='pollutant_id',
    aggfunc='mean'
)

# Create heatmap with adjusted layout
plt.figure(figsize=(12, 8))  # Increased figure size for better spacing
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='Blues', cbar=True, annot_kws={"size": 8})  # Reduced font size for annotations
plt.title('Average Pollutant Concentration by State and Pollutant Type', fontsize=12)
plt.xlabel('Pollutant Type', fontsize=10)
plt.ylabel('State', fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=8)  # Rotate x-axis labels and adjust font size
plt.yticks(fontsize=8)  # Adjust y-axis label font size
plt.tight_layout()  # Automatically adjust spacing
plt.show()
