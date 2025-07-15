# utils/variable_classification.py

class VariablesClassification:
    """
    This class provides access to variable classification lists through getter methods.
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

        # Define getters for each attribute
    @property
    def continuas(self):
        """Returns a copy of the continuous variables list to prevent modification."""
        return self._continuas

    @property
    def discretas(self):
        """Returns a copy of the discrete variables tuple (already immutable)."""
        return self._discretas

    @property
    def numericas(self):
        """Returns a copy of the numerical variables tuple (already immutable)."""
        return self._continuas + self._discretas

    @property
    def ordinales(self):
        """Returns a copy of the ordinal variables tuple."""
        return self._ordinales

    @property
    def nominales_binarias(self):
        """Returns a copy of the binary nominal variables tuple."""
        return self._nominales_binarias

    @property
    def nominales_no_binarias(self):
        """Returns a copy of the non-binary nominal variables tuple."""
        return self._nominales_no_binarias

    @property
    def todas_las_nominales(self):
        """Returns a copy of the non-binary nominal variables tuple."""
        return self._nominales_binarias + self._nominales_no_binarias

    @property
    def categoricas(self):
        """Returns a copy of the non-binary nominal variables tuple."""
        return self._ordinales + self._nominales_binarias + self._nominales_no_binarias

    @property
    def censo(self):
        """Returns a copy of the list of variables got from the 2022 census and Programa de Estudios del Conurbano."""
        return self._censo
