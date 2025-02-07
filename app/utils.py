import pandas as pd
import numpy as np
import io
from datetime import datetime
import streamlit as st

def identify_column_type(column_name, sample_data):
    """Identify the type of data in a column based on name and content"""
    name_lower = column_name.lower()
    
    # First check if the column is entirely numeric or convertible to numeric
    try:
        numeric_data = pd.to_numeric(sample_data, errors='coerce')
        if numeric_data.notna().any():  # If any values could be converted to numeric
            # Check if it's a percentage
            if numeric_data.max() <= 100 and numeric_data.min() >= 0:
                return 'percentage'
            return 'numeric'
    except:
        pass
    
    # Then check for date-like values
    try:
        date_test = pd.to_datetime(sample_data, errors='coerce')
        if date_test.notna().any():
            return 'date'
    except:
        pass
    
    # If not numeric or date, check keywords
    if any(k in name_lower for k in ['date', 'day', 'month', 'year', 'time']):
        return 'date'
    elif any(k in name_lower for k in ['quantity', 'sold', 'sales', 'units', 'volume']):
        return 'quantity'
    elif any(k in name_lower for k in ['price', 'cost', 'rate', 'inr', 'rs']):
        return 'price'
    elif any(k in name_lower for k in ['product', 'model', 'item', 'vehicle', 'sku']):
        return 'product'
    elif any(k in name_lower for k in ['stock', 'inventory', 'balance']):
        return 'stock'
    
    return 'other'

def map_columns(df):
    """Map DataFrame columns to standard names used by the model with duplicate handling"""
    column_mapping = {}
    used_mappings = set()
    
    # Get sample data for each column
    sample_data = df.head()
    
    for col in df.columns:
        col_type = identify_column_type(col, df[col])
        mapped_name = None
        
        if col_type == 'date':
            mapped_name = 'Date'
        elif col_type == 'quantity':
            mapped_name = 'Quantity_Sold'
        elif col_type == 'price':
            # Handle multiple price columns
            if 'fuel' in col.lower():
                mapped_name = 'Fuel_Price'
            elif 'interest' in col.lower():
                mapped_name = 'Interest_Rate'
            elif 'car' in col.lower() or 'vehicle' in col.lower():
                mapped_name = 'Price (INR)'
        elif col_type == 'product':
            mapped_name = 'Vehicle_Model'
        elif col_type == 'stock':
            mapped_name = 'Current_Stock'
            
        # If we have a mapping and it's not already used
        if mapped_name:
            # Handle duplicates by appending a suffix
            if mapped_name in used_mappings:
                base_name = mapped_name
                counter = 1
                while mapped_name in used_mappings:
                    mapped_name = f"{base_name}_{counter}"
                    counter += 1
            used_mappings.add(mapped_name)
            column_mapping[col] = mapped_name
    
    return column_mapping

def standardize_dataframe(df):
    """Standardize dataframe with improved type handling and duplicate resolution"""
    if df is None or df.empty:
        return None
        
    try:
        df_clean = df.copy()
        
        # Remove any completely empty columns or rows
        df_clean = df_clean.dropna(axis=1, how='all')
        df_clean = df_clean.dropna(how='all')
        
        # Print original columns and types for debugging
        print("Original columns and types:")
        for col in df_clean.columns:
            print(f"  • {col}: {df_clean[col].dtype}")
        
        # Clean column names and handle duplicates before mapping
        clean_names = []
        name_counts = {}
        
        for col in df_clean.columns:
            # Clean the name
            clean_name = col.strip().replace(' ', '_').replace('(', '').replace(')', '')
            
            # Handle duplicates by adding suffix
            if clean_name in name_counts:
                name_counts[clean_name] += 1
                clean_name = f"{clean_name}_{name_counts[clean_name]}"
            else:
                name_counts[clean_name] = 0
                
            clean_names.append(clean_name)
        
        # Assign cleaned column names
        df_clean.columns = clean_names
        
        # Convert numeric columns
        for col in df_clean.columns:
            try:
                # Try to convert to numeric if possible
                if df_clean[col].dtype == 'object':
                    numeric_data = pd.to_numeric(df_clean[col], errors='coerce')
                    if numeric_data.notna().any():  # If any values could be converted
                        df_clean[col] = numeric_data
            except Exception as e:
                print(f"Could not convert {col} to numeric: {str(e)}")
        
        # Map columns to standard names
        column_mapping = map_columns(df_clean)
        
        if column_mapping:
            # Create reversed mapping to check for duplicates
            rev_mapping = {}
            final_mapping = {}
            
            # Handle duplicate mappings
            for orig, new in column_mapping.items():
                if new in rev_mapping:
                    # If mapping exists, create numbered version
                    count = 1
                    while f"{new}_{count}" in rev_mapping:
                        count += 1
                    new_name = f"{new}_{count}"
                    final_mapping[orig] = new_name
                    rev_mapping[new_name] = orig
                else:
                    final_mapping[orig] = new
                    rev_mapping[new] = orig
            
            # Apply the final mapping
            df_clean = df_clean.rename(columns=final_mapping)
            
            print("Column mapping:")
            for orig, new in final_mapping.items():
                print(f"  • {orig} → {new}")
        
        # Handle date columns last
        date_cols = [col for col in df_clean.columns 
                    if any(k in col.lower() for k in ['date', 'day', 'month', 'year', 'time'])]
        
        for col in date_cols:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                print(f"Converted {col} to datetime")
            except Exception as e:
                print(f"Failed to convert {col} to datetime: {str(e)}")
        
        # Final check for duplicate columns
        if len(df_clean.columns) != len(set(df_clean.columns)):
            print("Still found duplicate columns after cleaning, keeping first occurrence")
            df_clean = df_clean.loc[:, ~df_clean.columns.duplicated(keep='first')]
        
        return df_clean
        
    except Exception as e:
        print(f"Error standardizing dataframe: {str(e)}")
        print("Column names:", list(df.columns))
        print("Column types:", df.dtypes)
        return None

def read_file(f):
    """Read a CSV file robustly with improved encoding and format handling"""
    try:
        # Reset file pointer and read content
        f.seek(0)
        try:
            # Try reading as string first
            content = f.read()
            if isinstance(content, str):
                # Convert string to bytes if needed
                content = content.encode('utf-8')
        except UnicodeDecodeError:
            # If string reading fails, read as bytes
            f.seek(0)
            content = f.read()
            if not isinstance(content, bytes):
                content = content.encode('utf-8')

        # Try reading with pandas directly first
        try:
            df = pd.read_csv(io.BytesIO(content))
            if not df.empty:
                st.success(f"Successfully read file using default pandas reader")
                return standardize_dataframe(df)
        except:
            pass

        # If direct reading fails, try different encodings and delimiters
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
        delimiters = [',', ';', '\t', '|']

        for encoding in encodings:
            try:
                # Try to decode the content
                decoded_content = content.decode(encoding)
                
                # Try each delimiter
                for delimiter in delimiters:
                    try:
                        df = pd.read_csv(
                            io.StringIO(decoded_content),
                            sep=delimiter,
                            engine='python',
                            on_bad_lines='skip'
                        )
                        if not df.empty:
                            st.success(f"Successfully read file using {encoding} encoding and {delimiter} delimiter")
                            return standardize_dataframe(df)
                    except:
                        continue
            except:
                continue

        # If all attempts fail
        st.error(f"Could not read {f.name} with any encoding/delimiter combination")
        return None

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def detect_file_type(df):
    """
    Detect file type based on column names.
    Returns one of: 'Sales', 'Inventory', 'Promotional', 'External', or 'Unknown'
    """
    cols = set(df.columns)
    
    # Count relevant columns for each type
    sales_cols = {'Quantity_Sold', 'Price (INR)'}.intersection(cols)
    inventory_cols = {'Current_Stock', 'Inventory_Turnover_Ratio'}.intersection(cols)
    
    # Determine type based on column presence
    if len(sales_cols) > 0:
        return "Sales"
    elif len(inventory_cols) > 0:
        return "Inventory"
    elif 'Date' in cols and any('price' in col.lower() for col in cols):
        return "Sales"
    elif 'Date' in cols and any('stock' in col.lower() for col in cols):
        return "Inventory"
    
    return "Unknown"

def prepare_time_features(df):
    """Extract time-based features from Date column with improved date handling"""
    if 'Date' not in df.columns:
        return df
    
    df_copy = df.copy()
    
    # Ensure Date column is datetime
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_copy['Date']):
            df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
        
        # Check if we have valid dates after conversion
        if df_copy['Date'].notna().any():
            df_copy['Year'] = df_copy['Date'].dt.year
            df_copy['Month'] = df_copy['Date'].dt.month
            df_copy['DayOfWeek'] = df_copy['Date'].dt.dayofweek
            df_copy['Quarter'] = df_copy['Date'].dt.quarter
        else:
            st.error("No valid dates found after conversion")
            return df
            
    except Exception as e:
        st.error(f"Error preparing time features: {str(e)}")
        return df
    
    return df_copy

def safe_mode(series):
    """Safely get mode or default value of a series"""
    if series.empty:
        return series.iloc[0] if not series.empty else None
    mode_vals = series.mode()
    return mode_vals.iloc[0] if not mode_vals.empty else series.iloc[0]

def clean_merged_data(df):
    """Clean merged dataframe by handling NaN values"""
    if df is None or df.empty:
        return None
        
    df_clean = df.copy()
    
    # Fill NaN values appropriately for each type of column
    numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    
    for col in numeric_columns:
        if col == 'Quantity_Sold':
            # Don't fill NaN in target variable, remove those rows
            df_clean = df_clean.dropna(subset=[col])
        elif 'Price' in col or 'Stock' in col:
            # Use forward fill then backward fill for time-series related columns
            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
            # If still has NaN, fill with median
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            # Use median for other numeric columns
            median_val = df_clean[col].median()
            if pd.isna(median_val):
                median_val = 0
            df_clean[col] = df_clean[col].fillna(median_val)
    
    for col in categorical_columns:
        if col != 'Date':  # Skip date column
            try:
                mode_val = safe_mode(df_clean[col])
                if mode_val is not None:
                    df_clean[col] = df_clean[col].fillna(mode_val)
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
            except Exception as e:
                st.warning(f"Error filling NaN values for column {col}: {str(e)}")
                df_clean[col] = df_clean[col].fillna('Unknown')
    
    return df_clean
