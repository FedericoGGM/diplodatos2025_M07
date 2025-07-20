# tp2_utils/df_categorical_transformations_utils.py

# Import libraries
import pandas as pd
import re

from datetime import datetime, timedelta

from .variables_classification import VariablesClassification

def _convert_excel_serial_date(value):  # Internal use only
    """
    Convert a date in Excel serial format to dd/mm/yyyy format.

    Parameters:
    ---------
    valuer : str | int
        Date in excel serial format.

    Returns
    ---------:
        Date in dd/mm/yyyy format.
    """
    if pd.isna(value):
        return value
    elif value == "31/10/0202":
        return "31/10/2022"
    elif "/" not in value and "no" not in value:
        int(value)
        base_date = datetime(1899, 12, 30)
        after_value = (base_date + timedelta(days=int(value))).strftime("%d/%m/%Y")
        return after_value
    return value

def apply_date_correction_to_df(df):
    """
    Apply all the corrections to 'fecha' column in the dataframe:
        - Convert a date in Excel serial format to dd/mm/yyyy format.
        - Convert the time data into pandas datetime objects, imputing non-valid values (strings like 'no se midió') as NaT.

    This function doesn't modify the dataframe in place, but returns a copy of the dataframe with the modified column.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df with 'fecha' column modified as described above.
    """
    df1 = df.copy(deep=True)
    df1["fecha"] = df1["fecha"].apply(_convert_excel_serial_date)
    df1['fecha'] = pd.to_datetime(df1['fecha'], format='%d/%m/%Y', errors='coerce')
    print("Columna 'fecha' corregida")
    return df1

def correct_campana(df):
    """
    Apply all the corrections to 'campaña' column in the dataframe:
        - Impute all valid values to lowercase values ('verano', 'otoño', 'invierno', 'primavera').
        - Impute all non-valid values to None.

    This function doesn't modify the dataframe in place, but returns a copy of the dataframe with the modified column.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df with 'campaña' column modified as described above.
    """
    df1 = df.copy(deep=True)
    # Pasamos todos los valores a minúscula
    df1['campaña'] = df1['campaña'].str.lower().where(df1['campaña'].notna(), None)

    # Imputamos como NaN los valores que no correspondan a 'verano', 'otoño', 'invierno', 'primavera'
    patron = re.compile(r'^(verano|otoño|invierno|primavera)', flags=re.IGNORECASE)
    df1['campaña'] = df1['campaña'].where(
        df1['campaña'].astype(str).str.contains(patron, na=False)
    )
    print("Columna 'camapaña' corregida")
    return df1

def correct_binary_categorical(df):
    '''
    Correct values in binary categorical columns. set to None those values that not correspond to 'Ausencia' or 'Presencia' (or similar options), and set
    the possible values to 0 ('Ausencia') and 1 ('Presencia').

    This function doesn't modify the dataframe in place, but returns a copy of the dataframe with the modified columns.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df with the binary columns modified as described above.
    '''
    # Create a copy of the dataframe
    df1 = df.copy(deep=True)

    # Select the columns to process using the VariablesClassification class
    classifier = VariablesClassification()
    cols_a_p = classifier.nominales_binarias

    # Pattern to detect "ausencia/presencia" (ignoring uppercase letters and variations)
    patron = re.compile(r'^(ausen|presen)', flags=re.IGNORECASE)

    for col in cols_a_p:
        # Extract root (first 5-6 letters) from the value
        raiz_valor = df1[col].astype(str).str.extract(patron)[0]

        # Map to 0 or 1 (or NaN if there's no coincidence)
        df1[col] = raiz_valor.str.lower().map(
            {
                'ausen': 0,
                'presen': 1
            }
        )

    print(f"Columnas {', '.join(cols_a_p)} corregidas")

    return df1

def correct_calidad_del_agua(df):
    """
    Set to None the non-valid values in column 'calidad_del_agua'.

    This function doesn't modify the dataframe in place, but returns a copy of the dataframe with the modified column.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df with 'caldad_del_agua' column modified as described above.
    """
    opciones_validas_calidad = [
        'Apta',
        'Levemente deteriorada',
        'Deteriorada',
        'Muy deteriorada',
        'Extremadamente deteriorada'
    ]
    df1 = df.copy(deep=True)
    df1['calidad_de_agua'] = df1['calidad_de_agua'].where(df1['calidad_de_agua'].isin(opciones_validas_calidad))
    print("Columna 'calidad_de_agua' corregida")
    return df1

def correct_year(df):
    """
    Set to NaN all the non-valid values in 'año' column.

    This function doesn't modify the dataframe in place, but returns a copy of the dataframe with the modified column.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df with 'año' column modified as described above.
    """
    df1 = df.copy(deep=True)
    df1['año'] =  pd.to_numeric(df1['año'], errors='coerce')
    print("Columna 'año' corregida")
    return df1

def imputation_per_muni(df):
    """
    Impute missing data in variables related with the 2022 census and Programa de Estudios del Conurbano using the 'codigo' variable, only using the two first characters
    of the values in 'codigo'. Variables 'latitud' and 'longitud' are not imputed by this function.

    This function doesn't modify the dataframe in place, but returns a copy of the dataframe with the modified columns.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to modify.

    Returns:
    ---------
    df1 : pandas.DataFrame
        A copy of df with the censo columns modified as described above (see tp2_utils/variables_classification.VariablesClassification).
    """
    # Create a copy of the dataframe
    df1 = df.copy(deep=True)

    # Get a list of the columns whose valuesimpute
    classifier = VariablesClassification()
    col_list = [col for col in classifier.censo if not col in ["latitud", "longitud"]]

    # Add code prefix column in df1
    df1['prefijo'] = df1['codigo'].str[:2]

    # For each column and prefix,fill NaN values with the not null value in the group
    for col in col_list:
        # Agrupar por prefijo y aplicar transformación
        df1[col] = df1.groupby('prefijo')[col].transform(
            lambda x: x.fillna(x.dropna().iloc[0]) if not x.dropna().empty else x # Assume there is a unique value for each column in col_list for each code prefix (municipality)
        )

    # Eliminate 'prefijo' column
    df1 = df1.drop(columns=['prefijo'])
    print("Imputación de variables de Censo y Programa de Conurbano -excepto longitud y latitud- usando código realizada")

    return df1
