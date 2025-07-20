# tp2_utils/df_exploration_utils.py

import pandas as pd
import re

def extract_comparison_values(df, column_name):
    """
    Extract numeric values from strings with '>x' or '<x' patterns in a dataframe´s column, where x represents a number.

    Parameters:
    ---------
    df : pandas.DataFrame

    column_name : str
        The name of the column.

    Returns:
    ---------
    result_df : pandas.DataFrame
        A dataframe with two columns: column_name+"_comparison" and column_name+"_value". Each file in result_df saves each value of the form '<x' or
        '>x'. The '<' or '>' symbol is saved in the first column, while the numerical value is saved in the second.
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

def numerical_col_data(df, col):
    """
    Prints the minimum, maximum, mean, median and standar desviation of a column. Additionally, indicates if there are values in the format '<x' or '>x',
    with x a number, and if they exist, prints those values and their count.

    No return is given.

    Parameters:
    ---------
    df : pandas.DataFrame

    col : str
        The name of the column.
    """
    print(col, "\n")
    result_df = extract_comparison_values(df, col)
    if len(result_df) != 0:
        print(result_df.value_counts())
    else:
        print("No presenta valores con '<' 0 '>'")
    # Imprimimos información de los valores numéricos de la columna
    numric = pd.to_numeric(df[col], errors='coerce')
    print(f"Mínimo: {numric.min()}, Máximo: {numric.max()}, Media: {numric.mean()}, Mediana: {numric.median()}, Desviación Estándar: {numric.std()}")
    print("\n---------------------------------------------------")

def col_interval(df, col, minlim=None, maxlim=None):
    """
    Indicate how many values in a dataframe´s column are in a interval defined by minlim and maxlim.

    If minlim is None and maxlim a numerical value, indicates how many values are in (-inf, maxlim).
    If minlim is a numerical value and maxlim is None, indicates how many values are in (minlim, inf).
    If both minlim and maxlim are numerical values, return how many values are in (minlim, maxlim).

    No return is given.

    Parameters:
    ---------
    df : pandas.DataFrame

    col : str
        The name of the column.

    minlim : None | float | int, default == None
        Open lower bound of the interval. If set to None, the lower bound is -inf.

    maxlim : None | float | int, default == None
        Open upper bound of the interval. If set to None, the upper bound is inf.
    """
    df_filt = df.copy(deep=True)
    df_filt[col] = pd.to_numeric(df_filt[col], errors='coerce')
    
    max_str = ""
    min_str = ""    

    if minlim:
        df_filt = df_filt[df_filt[col] > minlim]
        min_str = f' mayores a {minlim} '
    if maxlim:
        df_filt = df_filt[df_filt[col] < maxlim]
        max_str = f' menores a {maxlim} '
    conector_str = 'y' if minlim and maxlim else ""
    
    number_of_values = len(df_filt)
    print(f"Valores de {col}{min_str}{conector_str}{max_str}: {number_of_values} ({number_of_values*100/613:2.2f}%)")
