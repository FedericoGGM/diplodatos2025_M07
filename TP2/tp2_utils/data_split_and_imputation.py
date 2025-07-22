# tp2_utils/data_split_and_imputation.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

def split_in_tt(df, target_col, test_fraction, random_seed=None):
    """
    Splits a pandas dataframe in four datasets: X and y training sets, and X and y test sets.

    Parameters:
    ---------
    df : pandas.DataFrame
        The dataframe to split.

    target_col : str
        Target column.

    test_fraction : float
        Fraction of data to use in test set.

    random_seed : None | float, default = None
        seed to use in the random split.

    Returns:
    ---------
    X_train, X_test, y_train, y_test : pandas.DataFrame
        Train and test dataframes.
        """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=random_seed
    )

    total = len(df)
    print(f"Train: {len(X_train)}, muestras ({len(X_train)/total:.2%})")
    print(f"Test: {len(X_test)}, muestras ({len(X_test)/total:.2%})")

    return X_train, X_test, y_train, y_test

def numerical_imputation_exploration(X_train, numerical_cols):
    """
    Visualizes some options to input missing numerical values in a dataframe. No return is given.

    Parameters:
    ---------
    X_train : pandas.DataFrame
        The dataframe of the training set.
    numerical_cols : list of str
        List of the numerical columns.
    """
    knn_num_imputer_3 = KNNImputer(n_neighbors=3)
    knn_num_imputer_7 = KNNImputer(n_neighbors=7)
    bayes_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
    X_train_num_imputed_3 = knn_num_imputer_3.fit_transform(X_train[numerical_cols])
    X_train_num_imputed_7 = knn_num_imputer_7.fit_transform(X_train[numerical_cols])
    X_train_num_imputed_bay = bayes_imputer.fit_transform(X_train[numerical_cols])

    # Convert the imputed array back to DataFrame for easier handling
    X_train_num_imputed_3_df = pd.DataFrame(X_train_num_imputed_3,
                                        columns=numerical_cols,
                                        index=X_train.index)
    X_train_num_imputed_7_df = pd.DataFrame(X_train_num_imputed_7,
                                        columns=numerical_cols,
                                        index=X_train.index)
    X_train_num_imputed_bay_df = pd.DataFrame(X_train_num_imputed_bay,
                                     columns=numerical_cols,
                                     index=X_train.index)

    # DataFrames and titles for eacj subgrafic
    dataframes = [X_train, X_train_num_imputed_3_df, X_train_num_imputed_7_df, X_train_num_imputed_bay_df]
    titles = ['Sin imputar', 'KNN con 3 vecinos', 'KNN con 7 vecinos', 'Imputación Iterativa con regresión\nde cresta bayesiana']

    for col in numerical_cols:
        fig, axes = plt.subplots(1, 4, figsize=(18, 6))
        plt.suptitle(f"Distribución de {col}", fontsize=16)

        for i, (data, ax, title) in enumerate(zip(dataframes, axes, titles)):
            data[col].hist(bins=30, ax=ax)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Valores")
            if i == 0:
                ax.set_ylabel("Frecuencia")

        plt.tight_layout()
        plt.show()

def one_hot_encoding(X_train, X_test, categorical_cols):
    """
    Makes a One-Hot encoding over categorical variables in X_train.

    This function doesn´t make a transformation in place on X_train, but return a modified copy.

    Parameters:
    ---------
    X_train : pandas.DataFrame
        The dataframe of the training set.
    X_train : pandas.DataFrame
        The dataframe of the testing set.
    categorical_cols : list of str
        List of the categorical columns.

    Returns:
    ---------
    df_train_final, df_test_final : pandas.DataFrame
        Encoded datasets.
    feature_names : list
        Names of new colums
    """
    ### Creates an encoder. Note that:
        # drop='first' evoid multicollinearity (avoiding liear dependence between new categorical variables).
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])

    # Names of new columns
    feature_names = encoder.get_feature_names_out(categorical_cols)

    # Creates dataframes with new columns
    df_train_encoded = pd.DataFrame(X_train_encoded, columns=feature_names)
    df_test_encoded = pd.DataFrame(X_test_encoded, columns=feature_names)

    # Concatenates with the original dataframes
    df_train_final = pd.concat([X_train.drop(categorical_cols, axis=1), df_train_encoded], axis=1)
    df_test_final = pd.concat([X_test.drop(categorical_cols, axis=1), df_test_encoded], axis=1)

    print(f"\nCantidad de columnas después del One-Hot Encoding: {len(df_train_final.columns)}")
    return df_train_final, df_test_final, feature_names

def categorical_imputation_exploration(X_train, categorical_cols):
    """
    Visualizes some options to input missing categorical values in a dataframe. No return is given.

    Parameters:
    ---------
    X_train : pandas.DataFrame
        The dataframe to split.
    categorical_cols : list of str
        List of the categorical columns.
    """
    knn_cat_imputer_3 = KNNImputer(n_neighbors=3)
    knn_cat_imputer_7 = KNNImputer(n_neighbors=7)
    X_train_cat_imputed_3 = knn_cat_imputer_3.fit_transform(X_train[categorical_cols])
    X_train_cat_imputed_7 = knn_cat_imputer_7.fit_transform(X_train[categorical_cols])

    # Convert the imputed array back to DataFrame for easier handling
    X_train_cat_imputed_3_df = pd.DataFrame(X_train_cat_imputed_3,
                                        columns=categorical_cols,
                                        index=X_train.index)
    X_train_cat_imputed_7_df = pd.DataFrame(X_train_cat_imputed_7,
                                        columns=categorical_cols,
                                        index=X_train.index)

    # DataFrames and titles for eacj subgrafic
    dataframes = [X_train, X_train_cat_imputed_3_df, X_train_cat_imputed_7_df]
    titles = ['Sin imputar', 'KNN con 3 vecinos', 'KNN con 7 vecinos']

    for col in categorical_cols:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.suptitle(f"Distribución de {col}", fontsize=16)

        for i, (data, ax, title) in enumerate(zip(dataframes, axes, titles)):
            data[col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
            data[col].hist(bins=30, ax=ax)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Valores")
            if i == 0:
                ax.set_ylabel("Frecuencia")

        plt.tight_layout()
        plt.show()

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

def normalize_and_standardize(X_train, X_test, num_cols):
    """
    Aplica normalización (MinMax) y estandarización (StandardScaler) solo a columnas numéricas especificadas,
    y retorna DataFrames de pandas con todas las columnas (transformadas y no transformadas).

    Args:
        X_train: DataFrame de pandas con datos de entrenamiento.
        X_test: DataFrame de pandas con datos de prueba.
        num_cols: Lista de columnas numéricas a transformar (deben existir en los DataFrames).

    Returns:
        Tuple: (df_train_norm, df_test_norm, df_train_std, df_test_std)
        Todos los elementos son DataFrames de pandas.
    """
    # Verificar si las columnas existen
    missing_cols = [col for col in num_cols if col not in X_train.columns]
    if missing_cols:
        raise ValueError(f"Columnas no encontradas en DataFrame: {missing_cols}")

    # Copiar los DataFrames originales (manteniendo índices y columnas no numéricas)
    df_train_norm = X_train.copy()
    df_test_norm = X_test.copy()
    df_train_std = X_train.copy()
    df_test_std = X_test.copy()

    # 1. Normalización (MinMaxScaler)
    if num_cols:
        minmax_scaler = MinMaxScaler()
        df_train_norm[num_cols] = minmax_scaler.fit_transform(X_train[num_cols])
        df_test_norm[num_cols] = minmax_scaler.transform(X_test[num_cols])

    # 2. Estandarización (StandardScaler)
    if num_cols:
        std_scaler = StandardScaler()
        df_train_std[num_cols] = std_scaler.fit_transform(X_train[num_cols])
        df_test_std[num_cols] = std_scaler.transform(X_test[num_cols])

    return (
        df_train_norm,
        df_test_norm,
        df_train_std,
        df_test_std
    )
