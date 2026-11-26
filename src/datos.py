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
    retorno_min: np.ndarray


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
    descuento_anual: float
    n_campañas: int = 3
    riesgo: float = 0.5


    
def coste_por_labores(carpeta, familias, precio_gasoil, mant_por_litro):
    # Coste de una pasada de cada labor
    labores = pd.read_csv(carpeta / "Labores.csv")
    maquinaria = pd.read_csv(carpeta / "Maquinaria.csv")
    por_familia = pd.read_csv(carpeta / "Labores_familia.csv")

    labores = pd.merge(labores, maquinaria, on='maquina')
    propia = labores["propia"] == "si"
    litros = labores["litros_gasoil_ha"]
    coste_propia = litros * precio_gasoil + litros * mant_por_litro + labores["desgaste_apero_eur_ha"]

    labores["coste_pasada"] = np.where(propia, coste_propia, labores["contrato_eur_ha"])

    tabla = pd.merge(por_familia, labores[["labor", "coste_pasada"]], on='labor')
    tabla["coste"] = tabla["pasadas"] * (tabla["coste_pasada"] + tabla["producto_eur_ha"])

    coste_familia = tabla.groupby("familia")["coste"].sum()
    return familias.map(coste_familia).to_numpy()



def cargar_datos(carpeta=pathlib.Path(__file__).resolve().parent.parent / "Datos", archivo_parcelas = None):


    #Parametros
    tabla_parametro = pd.read_csv(carpeta / "Parametros.csv", index_col=0)
    precio_abono = tabla_parametro.loc["precio_abono_kg", "valor"]
    reduccion_abono = tabla_parametro.loc["reduccion_abono_tras_leguminosa", "valor"]
    basica = tabla_parametro.loc["ayuda_basica_anual", "valor"]
    redistributiva = tabla_parametro.loc["ayuda_redistributiva_anual", "valor"]
    ecorregimen = tabla_parametro.loc["ecorregimen_anual", "valor"]
    precio_gasoil = tabla_parametro.loc["precio_gasoil", "valor"]
    mant_por_litro = tabla_parametro.loc["mantenimiento_eur_litro", "valor"]

    ayudas = Ayudas(basica, redistributiva, ecorregimen)
    descuento = tabla_parametro.loc["descuento_anual", "valor"]
    

    #Cultivos
    tabla_cultivos = pd.read_csv(carpeta / "Cultivos.csv", index_col=0)

    nombres = tabla_cultivos.index.to_list()
    rendimiento = tabla_cultivos["rendimiento_toneladas_ha"].to_numpy()
    precio = tabla_cultivos["precio_tonelada"].to_numpy()
    ayuda = tabla_cultivos["ayuda_asociada_PAC"].to_numpy()
    coste = coste_por_labores(carpeta / "Costes", tabla_cultivos["familia"], precio_gasoil, mant_por_litro) \
        + tabla_cultivos["seguro_eur_ha"].to_numpy()
    factor_bueno = tabla_cultivos["factor_bueno"].to_numpy()
    factor_malo = tabla_cultivos["factor_malo"].to_numpy()
    dosis = tabla_cultivos["dosis_kg_ha"].to_numpy()
    precio_semilla = tabla_cultivos["precio_semilla_kg"].to_numpy()
    stock = tabla_cultivos["stock_semilla_kg"].to_numpy()
    abono_kg_ha = tabla_cultivos["abono_kg_ha"].to_numpy()
    retorno_min = tabla_cultivos["retorno_min_anios"].to_numpy()
    

    


    #Factores de rendimiento por tipo de año (filas: bueno, mediocre, malo)
    factores = np.vstack([factor_bueno, np.ones(len(nombres)), factor_malo])
    cosecha_kg_ha = rendimiento*factores*1000

    #Beneficio por hectárea: una fila por tipo de año, una columna por cultivo
    beneficio_ha = rendimiento * factores * precio + ayuda - coste




    es_cereal = (tabla_cultivos["familia"] == "cereal").to_numpy()
    es_leguminosa = (tabla_cultivos["familia"] == "leguminosa").to_numpy()

    cultivos = Cultivos(nombres, beneficio_ha, es_cereal, es_leguminosa,
                    dosis, precio_semilla, stock, abono_kg_ha, cosecha_kg_ha, retorno_min)


    #Diccionario nombre -> numero, segun el orden en la tabla de cultivos
    nombre_a_numero = {nombre: i for i, nombre in enumerate(nombres)}

    #Parcelas
    if archivo_parcelas is None:
        archivo_parcelas = carpeta / "Parcelas.csv"
    tabla_parcelas = pd.read_csv(archivo_parcelas, index_col=0)


    ids = tabla_parcelas.index.to_list()
    superficie = tabla_parcelas["superficie_ha"].to_numpy()

    columnas_historial = sorted((col for col in tabla_parcelas.columns if col.startswith("anio_")),
                                key=lambda col: int(col.split("_")[1]))

    historial = np.array([[nombre_a_numero[nombre] for nombre in fila] for fila in tabla_parcelas[columnas_historial].to_numpy()])

    parcelas = Parcelas(ids, superficie, historial)


    #Tipo de año
    estados = ["bueno", "mediocre", "malo"]
    prior = np.array([1, 1, 1])

    tabla_campania = pd.read_csv(carpeta / "Campanias.csv", index_col=0)
    recuento = tabla_campania["estado"].value_counts().reindex(estados, fill_value=0).to_numpy()

    probabilidades = (prior + recuento) / (prior.sum() + recuento.sum())


    tipo_anio = Tipo_Anio(estados, probabilidades)

    return Datos(cultivos, parcelas, tipo_anio, reduccion_abono, precio_abono, ayudas, descuento)

