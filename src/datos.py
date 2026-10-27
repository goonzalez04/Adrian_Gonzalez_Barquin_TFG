from dataclasses import dataclass
import numpy as np
import pathlib
import pandas as pd

@dataclass
class Cultivos:
    nombres: list
    beneficio_ha: np.ndarray
    es_cereal: np.ndarray
    es_leguminosa: np.ndarray
    dosis: np.ndarray
    precio_semilla: np.ndarray
    stock_semilla: np.ndarray
    abono_kg_ha: np.ndarray
    cosecha_kg_ha: np.ndarray
    


    @property
    def n(self):
        return len(self.nombres)

@dataclass
class Parcelas:
    ids: list
    superficie: np.ndarray
    historial: np.ndarray

    @property
    def n(self):
        return len(self.ids)


@dataclass
class Tipo_Anio:
    estados: list
    probabilidades: np.ndarray


@dataclass
class Ayudas:
    basica_anual: float
    redistributiva_anual: float
    ecorregimen_anual: float
 

@dataclass
class Datos:
    cultivos: Cultivos
    parcelas: Parcelas
    tipo_anio: Tipo_Anio
    reduccion_abono: float
    precio_abono: float
    ayudas: Ayudas
    n_campañas: int = 3
    riesgo: float = 0.5


    
    

def cargar_datos(carpeta=pathlib.Path(__file__).resolve().parent.parent / "Datos"):

    #Cultivos
    tabla_cultivos = pd.read_csv(carpeta / "Cultivos.csv", index_col=0)

    nombres = tabla_cultivos.index.to_list()
    rendimiento = tabla_cultivos["rendimiento_toneladas_ha"].to_numpy()
    precio = tabla_cultivos["precio_tonelada"].to_numpy()
    ayuda = tabla_cultivos["ayuda_asociada_PAC"].to_numpy()
    coste = tabla_cultivos["coste_ha"].to_numpy()
    factor_bueno = tabla_cultivos["factor_bueno"].to_numpy()
    factor_malo = tabla_cultivos["factor_malo"].to_numpy()
    dosis = tabla_cultivos["dosis_kg_ha"].to_numpy()
    precio_semilla = tabla_cultivos["precio_semilla_kg"].to_numpy()
    stock = tabla_cultivos["stock_semilla_kg"].to_numpy()
    abono_kg_ha = tabla_cultivos["abono_kg_ha"].to_numpy()
    

    #Parametros
    tabla_parametro = pd.read_csv(carpeta / "Parametros.csv", index_col=0)
    precio_abono = tabla_parametro.loc["precio_abono_kg", "valor"]
    reduccion_abono = tabla_parametro.loc["reduccion_abono_tras_leguminosa", "valor"]
    basica = tabla_parametro.loc["ayuda_basica_anual", "valor"]
    redistributiva = tabla_parametro.loc["ayuda_redistributiva_anual", "valor"]
    ecorregimen = tabla_parametro.loc["ecorregimen_anual", "valor"]
    
    ayudas = Ayudas(basica, redistributiva, ecorregimen)


    #Factores de rendimiento por tipo de año (filas: bueno, mediocre, malo)
    factores = np.vstack([factor_bueno, np.ones(len(nombres)), factor_malo])
    cosecha_kg_ha = rendimiento*factores*1000

    #Beneficio por hectárea: una fila por tipo de año, una columna por cultivo
    beneficio_ha = rendimiento * factores * precio + ayuda - coste




    es_cereal = (tabla_cultivos["familia"] == "cereal").to_numpy()
    es_leguminosa = (tabla_cultivos["familia"] == "leguminosa").to_numpy()

    cultivos = Cultivos(nombres, beneficio_ha, es_cereal, es_leguminosa,
                    dosis, precio_semilla, stock, abono_kg_ha, cosecha_kg_ha)


    #Diccionario nombre -> numero, segun el orden en la tabla de cultivos
    nombre_a_numero = {nombre: i for i, nombre in enumerate(nombres)}

    #Parcelas
    tabla_parcelas = pd.read_csv(carpeta / "Parcelas.csv", index_col=0)

    ids = tabla_parcelas.index.to_list()
    superficie = tabla_parcelas["superficie_ha"].to_numpy()

    columnas_historial = ["anio_-3", "anio_-2", "anio_-1"]
    historial = np.array([[nombre_a_numero[nombre] for nombre in fila] for fila in tabla_parcelas[columnas_historial].to_numpy()])

    parcelas = Parcelas(ids, superficie, historial)


    #Tipo de año
    estados = ["bueno", "mediocre", "malo"]
    prior = np.array([1, 1, 1])

    tabla_campania = pd.read_csv(carpeta / "Campanias.csv", index_col=0)
    recuento = tabla_campania["estado"].value_counts().reindex(estados, fill_value=0).to_numpy()

    probabilidades = (prior + recuento) / (prior.sum() + recuento.sum())


    tipo_anio = Tipo_Anio(estados, probabilidades)

    return Datos(cultivos, parcelas, tipo_anio, reduccion_abono, precio_abono, ayudas)

    