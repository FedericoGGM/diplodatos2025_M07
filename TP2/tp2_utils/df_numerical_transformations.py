# tp2_utils/df_numerical_transformations.py

import pandas as pd
import numpy as np

from .variables_classification import VariablesClassification

def set_to_numerical(df):
    """
    Set to numeric all columns classified as numericals in VariablesClassification (except for 'latitud' and 'longitud') in a pandas DataFrame df, imputing non-numerical values as NaN.

    This function doesn´t modify the dataframe in place, but returns a copy of the dataframe with the modified column.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df with the numerical columns (classified in VariablesClassification) modified as described above.
    """
    df1 = df.copy()
    classifier = VariablesClassification()
    cols = [col for col in classifier.numericas if not col in ["latitud", "longitud"]]
    for col in cols:
        df1[col] = pd.to_numeric(df1[col], errors='coerce')
    print("Imputación de valores no numéricos de variables numéricas completada.")
    return df1

def impute_outliers(df, col, maxlim=None, minlim=None):
    """
    Assign outlier values as NaN: Upper outliers if maxlim isn´t None, lower outliers if minlim isn´t None.

    This function doesn´t modify the dataframe in place, but returns a copy of the dataframe with the modified column.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    col : str
        The name of the column to modify.

    maxlim : None | float | int, default == None
        If provided, indicates the maximun to the values in col. Each value above maxlim is considered an outlier.

    minlim : None | float | int, default == None
        If provided, indicates the minimun to the values in col. Each value below minlim is considered an outlier.

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df with the col column modified as described above.
    """
    df1 = df.copy()
    
    if maxlim:
        mask = pd.to_numeric(df1[col], errors='coerce') > maxlim
        df1.loc[mask, col] = np.nan
        print(f"Imputación de valores outliers de columna {col} -mayores a {maxlim}- completada)")
    if minlim:
        mask = pd.to_numeric(df1[col], errors='coerce') < minlim
        df1.loc[mask, col] = np.nan
        print(f"Imputación de valores outliers de columna {col} -menores a {minlim}- completada)")
    return df1

def imput_hidr_01(df):
    """
    Impute '<0.10', '\xa0<0.10', '\xa00.10' y '\xa00.20' values in hidr_deriv_petr_ug_l column of df to zero.

    This function doesn´t modify the dataframe in place, but returns a copy of the dataframe with the modified column.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df with the hidr_deriv_petr_ug_l column modified as described above.
    """
    df1 = df.copy()
    mask = df['hidr_deriv_petr_ug_l'].isin(['<0.10', '\xa0<0.10', '\xa00.10', '\xa00.20'])
    df1.loc[mask, 'hidr_deriv_petr_ug_l'] = 0
    print("Imputación de valores '<0.10', '\xa0<0.10', '\xa00.10' y '\xa00.20' en columna hidr_deriv_petr_ug_l completada")
    return df1

def remove_dots_in_populations(df):
    """
    Remove the "." patterns in each value of Poblacion_partido and Personas_con_cloacas columns of df.

    This function doesn´t modify the dataframe in place, but returns a copy of the dataframe with the modified column.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df with the Poblacion_partido and Personas_con_cloacas columns modified as described above.
    """
    df1 = df.copy()
    columns = ['Poblacion_partido', 'Personas_con_cloacas']
    for col in columns:
        df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    print("Eliminación de puntos en los valores de 'Poblacion_partido' y 'Personas_con_cloacas' completada")
    return df
