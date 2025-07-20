# tp2_utils/__init__.py

from .df_categorical_transformations_utils import *
from .variables_classification import *
from .df_exploration_utils import *
from .df_numerical_transformations import *

__all__ = [
    # Functions from df_categorical_transformations_utils
    'apply_date_correction_to_df',
    'correct_campana',
    'correct_binary_categorical',
    'correct_calidad_del_agua',
    'correct_ano',
    'imputation_per_muni',

    # Functions of df_exploration_utils
    'extract_comparison_values',
    'numerical_col_data',
    'col_interval',

    # Functions of df_numerical_transformations
    'set_to_numerical',
    'impute_outliers',
    'imput_hidr_01',
    'remove_dots_in_populations',

    # Clsses from variables_classification
    'VariablesClassification'
]


__version__ = '1.0.0'
