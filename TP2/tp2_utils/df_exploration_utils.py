# utils/df_exploration_utils.py

import pandas as pd
import re

def extract_comparison_values(df, column_name):
    """
    Extract numeric values from strings with '>x' or '<x' patterns in a DataFrame column.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of the column to process

    Returns:
        pd.DataFrame: Copy of original DataFrame with two new columns:
            - '[column_name]_comparison': contains '>', '<', or None
            - '[column_name]_value': contains the extracted numeric value or None
    """
    # Create new df to store the values of interest
    columnas = [f'{column_name}_comparison', f'{column_name}_value']
    result_df = pd.DataFrame(columns=columnas)

    # Regular expression to match patterns like >x, <x, >x.y, <x.y
    pattern = re.compile(r'^([<>])\s*(\d+\.?\d*)$')

    for value in df[column_name]:
        if pd.isna(value) or not isinstance(value, str):
            continue

        match = pattern.match(str(value).strip())
        if match:
            comparison_op = match.group(1)  # Either '>' or '<'
            numeric_value = float(match.group(2))  # Convert to float

            # Añadimos nuevo registro al result_df
            nuevo_registro = {
                f'{column_name}_comparison': comparison_op,  # Operador de comparación
                f'{column_name}_value': numeric_value        # Valor numérico
            }
            result_df.loc[len(result_df)] = nuevo_registro

    return result_df

def compare_comparison_values(df, col):
    print(col, "\n")
    result_df = extract_comparison_values(df, col)
    print(result_df.value_counts())
    numric = pd.to_numeric(df[col], errors='coerce')
    print(f"Mínimo: {numric.min()}, Máximo: {numric.max()}")
    print("\n---------------------------------------------------")
