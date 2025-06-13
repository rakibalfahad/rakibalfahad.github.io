---
title: "Efficient Data Processing with Pandas"
date: 2025-06-13
categories:
  - data-processing
  - python
tags:
  - pandas
  - data-analysis
  - code-snippet
header:
  image: "/images/fort point1.png"
  teaser: "/images/fort point1.png"
excerpt: "A collection of useful Pandas code snippets for efficient data processing and analysis."
---

# Efficient Data Processing with Pandas

Pandas is a powerful library for data manipulation and analysis in Python. Here are some useful code snippets to help you process data more efficiently.

## Reading Different File Formats

```python
import pandas as pd

# CSV file
df_csv = pd.read_csv('data.csv')

# Excel file
df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON file
df_json = pd.read_json('data.json')

# SQL query
from sqlalchemy import create_engine
engine = create_engine('sqlite:///database.db')
df_sql = pd.read_sql_query('SELECT * FROM table_name', engine)
```

## Data Cleaning Techniques

```python
# Drop missing values
df.dropna(inplace=True)

# Fill missing values
df.fillna(value={'numeric_col': 0, 'text_col': 'Unknown'}, inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Convert data types
df['date_col'] = pd.to_datetime(df['date_col'])
df['numeric_col'] = pd.to_numeric(df['numeric_col'], errors='coerce')
df['category_col'] = df['category_col'].astype('category')

# String manipulations
df['text_col'] = df['text_col'].str.lower()
df['text_col'] = df['text_col'].str.strip()
df['text_col'] = df['text_col'].str.replace('old', 'new')
```

## Efficient Data Transformations

```python
# Apply function to columns
df['new_col'] = df['col'].apply(lambda x: x * 2)

# Apply function to rows
df['row_sum'] = df[['col1', 'col2', 'col3']].apply(sum, axis=1)

# Group by operations
grouped = df.groupby('category')
mean_values = grouped.mean()
count_values = grouped.size()
aggregated = grouped.agg({'numeric_col1': 'mean', 'numeric_col2': 'sum', 'text_col': 'count'})

# Pivot tables
pivot_table = pd.pivot_table(df, 
                              values='value_col', 
                              index=['row_category'], 
                              columns=['column_category'], 
                              aggfunc='mean', 
                              fill_value=0)

# Melt (un-pivot) tables
melted_df = pd.melt(df, 
                    id_vars=['id_col'], 
                    value_vars=['val1', 'val2', 'val3'],
                    var_name='variable',
                    value_name='value')
```

## Advanced Filtering and Selection

```python
# Boolean indexing
filtered_df = df[df['value'] > 100]

# Multiple conditions
filtered_df = df[(df['value'] > 100) & (df['category'] == 'A')]

# Query method (often more readable)
filtered_df = df.query('value > 100 and category == "A"')

# isin for multiple values
filtered_df = df[df['category'].isin(['A', 'B', 'C'])]

# String contains
filtered_df = df[df['text_col'].str.contains('pattern')]

# Get rows where column value is between a range
filtered_df = df[(df['value'] >= 100) & (df['value'] <= 200)]
# Or more concisely:
filtered_df = df[df['value'].between(100, 200)]
```

## Time Series Analysis

```python
# Resample time series data
daily_data = df.resample('D').mean()  # Daily resampling
monthly_data = df.resample('M').sum()  # Monthly resampling

# Rolling window calculations
df['rolling_mean'] = df['value'].rolling(window=7).mean()
df['rolling_std'] = df['value'].rolling(window=7).std()

# Shift values for lag features
df['prev_day'] = df['value'].shift(1)
df['next_day'] = df['value'].shift(-1)

# Time-based indexing
df['2020-01-01':'2020-01-31']  # All rows between these dates

# Difference calculation
df['daily_change'] = df['value'].diff()
df['pct_change'] = df['value'].pct_change() * 100
```

Feel free to adapt these code snippets to your specific data processing needs. Pandas offers many more functions and features, so make sure to check the [official documentation](https://pandas.pydata.org/docs/) for more information.
