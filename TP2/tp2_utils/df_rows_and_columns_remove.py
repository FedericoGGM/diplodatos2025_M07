# tp2_utils/df_rows_and_columns_remove.py

import numpy as np

from .variables_classification import *

def remove_null_rows(df):
    """
    Creates a copy of df without its rows that have all values in continuous numerical and binomial categorical columns as null (except for 'latitud' and 'longitud').

    This function doesn´t modify the dataframe in place.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df without the null rows in the columns mentioned above.
    """
    classifier = VariablesClassification()
    cont_bin= list(classifier.continuas + classifier.nominales_binarias)
    important_cols = [col for col in cont_bin if not col in ['latitud', 'longitud']]

    df1 = df.dropna(subset=important_cols, how='all')
    print(f"Se eliminaron {len(df)-len(df1)} registros nulos.")

    return df1

def remove_columns_with_too_many_sull(df, p):
    """
    Creates a copy of df without columns that exceed a missing data percentage threshold p. Additionally, visualize the percentage of missing data in each column.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    p : float
        The percentage threshold

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df without the mentioned columns.
    """
    # Compute missing values percentage by column
    null_percentage = df.isnull().mean() * 100
    print("Columna\t\t\tPorcentaje de Nulos")
    print(null_percentage, "\n")

    # Filter columns by missing data percentage
    columns_with_many_null = null_percentage[null_percentage > p].index.tolist()
    df1 = df.drop(columns_with_many_null, axis=1)

    # Print removed columns
    remove_cols_str = ", ".join(columns_with_many_null)
    print(f"Columnas eliminadas (con >{p}% de datos faltantes): {remove_cols_str}.")
    print(f"Hemos eliminado {len(columns_with_many_null)} de {len(df.columns)} columnas.")

    return df1

def remove_correlated_cols(df, corr_threshold):
    """
    Creates a copy of df without strongly correlated colums. This function Makes groups of strongly correlated columns and remove all columns in each group but one member.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    corr_threshold : float
        Threshold to decide how should be the r coeficient between two columns to decide they´re strongly correlated.

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df without the strongly correlated columns.
    """
    corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()

    # Creates a mask to only conserve the superior triangular correlation matrix
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_sup = corr_matrix.where(mask)

    # Identify strongly correlated columns
    correlated_columns = [col for col in corr_sup.columns if any(corr_sup[col] > corr_threshold)]

    # Print removed columns
    remove_cols_str = ", ".join(correlated_columns)
    print("Columnas eliminadas por alta correlación:", remove_cols_str)

    df1 = df.drop(columns=correlated_columns)

    return df1
