# tp2_utils/variables_classification.py

class VariablesClassification:
    """
    A centralized repository for variable classification to data analysis of the project 'El impacto de las condiciones ambientales en la calidad del agua del Río de La
    Plata'.
    
    This class provides organized access to different categories of the variables provided by the project´s database. Variables are classified as continuous, discrete,
    ordinal, nominal (binary and non-binary), and census-related variables. All variable lists are immutable to maintain data integrity.

    The class provides property decorators for safe access to these classifications without modification risks. The classifications include:

    - Continuous variables (physical/chemical measurements)
    - Discrete variables (countable quantities)
    - Ordinal variables (ordered categories)
    - Nominal variables (categorical data, both binary and non-binary)
    - Census-derived variables (demographic and economic indicators)

    Example:
        >>> var_class = VariablesClassification()
        >>> continuous_vars = var_class.continuas
        >>> numeric_vars = var_class.numericas
        >>> 'ph' in continuous_vars
        True

    Note:
        All returned collections are immutable (tuples) or copies to prevent accidental modification of the master lists. For census variables, some overlap exists with
        other categories.
    """
    _continuas = (
        'tem_agua',
        'tem_aire',
        'od',
        'ph',
        'colif_fecales_ufc_100ml',
        'escher_coli_ufc_100ml',
        'enteroc_ufc_100ml',
        'nitrato_mg_l',
        'nh4_mg_l',
        'p_total_l_mg_l',
        'fosf_ortofos_mg_l',
        'dbo_mg_l',
        'dqo_mg_l',
        'turbiedad_ntu',
        'hidr_deriv_petr_ug_l',
        'cr_total_mg_l',
        'cd_total_mg_l',
        'clorofila_a_ug_l',
        'microcistina_ug_l',
        'ica',
        'latitud',
        'longitud'
    )

    _discretas = (
        'Poblacion_partido',
        'Personas_con_cloacas'
    )

    _ordinales = (
        'orden',
        'calidad_de_agua',
        'Agricultura, ganadería, caza y silvicultura',
        'Pesca explotación de criaderos de peces y granjas piscícolas y servicios conexos',
        'Explotación de minas y canteras',
        'Industria Manufacturera',
        'Electricidad, gas y agua',
        'Construcción',
        'Servicios',
        'fecha',
        'año'
    )

    _nominales_binarias = (
        'olores',
        'color',
        'espumas',
        'mat_susp'
    )

    _nominales_no_binarias = (
        'codigo',
        'sitios',
        'campaña',
        'gobierno_local',
        'sitio',
        'Actividad_principal'
    )

    _censo = (
        'gobierno_local',
        'latitud',
        'longitud',
        'Poblacion_partido',
        'Personas_con_cloacas',
        'Actividad_principal',
        'Agricultura, ganadería, caza y silvicultura',
        'Pesca explotación de criaderos de peces y granjas piscícolas y servicios conexos',
        'Explotación de minas y canteras',
        'Industria Manufacturera',
        'Electricidad, gas y agua',
        'Construcción',
        'Servicios'
    )

    @property
    def continuas(self):
        """Tuple: Continuous variables list."""
        return self._continuas

    @property
    def discretas(self):
        """Tuple: Discrete variables."""
        return self._discretas

    @property
    def numericas(self):
        """Tuple: Numerical variables tuple."""
        return self._continuas + self._discretas

    @property
    def ordinales(self):
        """Tuple: Ordinal variables."""
        return self._ordinales

    @property
    def nominales_binarias(self):
        """Tuple: Binary nominal variables."""
        return self._nominales_binarias

    @property
    def nominales_no_binarias(self):
        """Tuple: Non-binary nominal variables."""
        return self._nominales_no_binarias

    @property
    def todas_las_nominales(self):
        """Tuple: All nominal variables."""
        return self._nominales_binarias + self._nominales_no_binarias

    @property
    def categoricas(self):
        """Tuple: All categorical variables."""
        return self._ordinales + self._nominales_binarias + self._nominales_no_binarias

    @property
    def censo(self):
        """Tuple: 2022 census and Programa de Estudios del Conurbano related variables."""
        return self._censo
