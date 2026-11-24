import numpy as np
import pathlib
import time
import pandas as pd
from restricciones import repetir_familia_cultivo, diversificacion_pac, retorno_minimo, rotacion_cultivos
from datos import cargar_datos
from multiprocessing import Pool






def plan_aleatorio(datos):
    return np.random.randint(0, datos.cultivos.n, size=(datos.parcelas.n, datos.n_campañas))


def poblacion_inicial(datos, n_individuos):
    return np.array([plan_aleatorio(datos) for _ in range(n_individuos)])

def ayudas_fijas(datos):
    a = datos.ayudas

    return a.basica_anual + a.redistributiva_anual + a.ecorregimen_anual


def margenes_poblacion(poblacion, datos):
    
    c = datos.cultivos
    superficie = datos.parcelas.superficie
    historial = datos.parcelas.historial

    # Hectáreas de cada cultivo en cada campaña, para todos los planes  -> (I, T, C)
    es_cultivo = poblacion[..., None] == np.arange(c.n)                  # (I, P, T, C) True/False
    ha = (es_cultivo * superficie[None, :, None, None]).sum(axis=1)      # suma sobre parcelas

    # Semilla: se compra lo que no cubre el stock                        -> (I, T)
    comprar = np.maximum(0, c.dosis * ha - c.stock_semilla)
    coste = (comprar * c.precio_semilla).sum(axis=-1)

    # Abono: cultivo anterior de cada casilla (año 1 -> último del historial)
    ultimo = np.broadcast_to(historial[:, -1], poblacion.shape[:2])[..., None]   # (I, P, 1)
    anterior = np.concatenate([ultimo, poblacion[:, :, :-1]], axis=-1)           # (I, P, T)
    factor = np.where(c.es_leguminosa[anterior], 1 - datos.reduccion_abono, 1)
    abono_kg = c.abono_kg_ha[poblacion] * factor * superficie[None, :, None]
    coste = coste + abono_kg.sum(axis=1) * datos.precio_abono

    # Beneficio por tipo de año                                         -> (I, T, 3)
    beneficio = ha @ c.beneficio_ha.T
    tabla = beneficio - coste[..., None] + ayudas_fijas(datos)
    return tabla.transpose(0, 2, 1)                                      # (I, 3, T)


def margenes_por_campaña(plan, datos):
    return margenes_poblacion(plan[None], datos)[0]



def evaluar_plan(plan, datos):
    secuencia = np.hstack([datos.parcelas.historial, plan])
    n_historial = datos.parcelas.historial.shape[1]

    
    pesos = datos.descuento_anual ** np.arange(datos.n_campañas)
    margenes = margenes_por_campaña(plan, datos) @ pesos


    violaciones = (repetir_familia_cultivo(secuencia, datos.cultivos.es_cereal, 3, n_historial)
                   + repetir_familia_cultivo(secuencia, datos.cultivos.es_leguminosa, 1, n_historial))

    # BCAM 7 de la PAC (modalidad A): rotación de cultivos y diversificación
    violaciones += rotacion_cultivos(secuencia, datos.parcelas.superficie, datos.cultivos.n, max_anios=2, desde=n_historial)
    violaciones += retorno_minimo(secuencia, datos.cultivos.retorno_min, desde=n_historial)
    violaciones += diversificacion_pac(plan, datos.parcelas.superficie, datos.cultivos.n)

    return margenes, violaciones



def aptitud(plan, datos):
    """Aptitud de un plan (parcelas, campañas) -> un número,
    o de una población (individuos, parcelas, campañas) -> un array (individuos,).

    aptitud = margen esperado - riesgo * (esperado - año malo) - 10 000 * reglas incumplidas
    """
    # Un solo plan: lo trato como una población de 1
    if plan.ndim == 2:
        return aptitud(plan[None], datos)[0]

    poblacion = plan
    c = datos.cultivos
    superficie = datos.parcelas.superficie
    historial = datos.parcelas.historial
    n_historial = historial.shape[1]

    # Historial + plan de cada individuo                               -> (I, P, H+T)
    historial = np.broadcast_to(historial, (len(poblacion),) + historial.shape)
    secuencia = np.concatenate([historial, poblacion], axis=-1)

    # Margen de cada plan en cada tipo de año, con descuento           -> (I, 3)
    pesos = datos.descuento_anual ** np.arange(datos.n_campañas)
    margenes = margenes_poblacion(poblacion, datos) @ pesos

    # Reglas incumplidas de cada plan                                   -> (I,)
    violaciones = (repetir_familia_cultivo(secuencia, c.es_cereal, 3, n_historial)
                   + repetir_familia_cultivo(secuencia, c.es_leguminosa, 1, n_historial)
                   + rotacion_cultivos(secuencia, superficie, c.n, max_anios=2, desde=n_historial)
                   + retorno_minimo(secuencia, c.retorno_min, desde=n_historial)
                   + diversificacion_pac(poblacion, superficie, c.n))

    # Margen esperado, riesgo y penalización                            -> (I,)
    esperado = margenes @ datos.tipo_anio.probabilidades
    malo = margenes[:, datos.tipo_anio.estados.index("malo")]
    return esperado - datos.riesgo * (esperado - malo) - 10000 * violaciones


def torneo(poblacion, aptitudes, n, k=3):
    
    n_individuos = len(poblacion)
    indices = np.random.randint(0, n_individuos, size=(n, k))
    columna = aptitudes[indices].argmax(axis = 1)
    ganadores = indices[np.arange(n), columna]
    return poblacion[ganadores]

def cruza(padre, madre):
    mascara = np.random.rand(*padre.shape[:-1]) < 0.5
    return np.where(mascara[..., None], padre, madre)


def mutar(plan, tasa, datos):
    mascara = np.random.rand(*plan.shape) < tasa
    nuevos = np.random.randint(0, datos.cultivos.n, size=plan.shape)
    return np.where(mascara, nuevos, plan)

def parametros_GA(datos):
    casillas= datos.parcelas.n*datos.n_campañas
    tasa = 1/casillas
    n_individuos = int(np.clip(10 * casillas, 50, 300))
    return n_individuos, tasa


def algoritmo_genetico(datos, n_elite=2, paciencia=100, max_generaciones=2000):
    n_individuos, tasa = parametros_GA(datos)
    poblacion = poblacion_inicial(datos, n_individuos)
    mejores = []

    mejor_hasta_ahora = -np.inf
    sin_mejora = 0
    generacion = 0


    while sin_mejora < paciencia and generacion < max_generaciones:
        aptitudes = aptitud(poblacion, datos)
        mejor_de_esta = aptitudes.max()
        mejores.append(mejor_de_esta)

        if mejor_de_esta > mejor_hasta_ahora:
            mejor_hasta_ahora = mejor_de_esta
            sin_mejora = 0
        else:
            sin_mejora += 1

        orden = np.argsort(aptitudes)[::-1]

        elite = poblacion[orden[:n_elite]]
        nhijos = n_individuos - n_elite

        padres = torneo(poblacion, aptitudes, nhijos)
        madres = torneo(poblacion, aptitudes, nhijos)

        hijos = mutar(cruza(padres, madres), tasa, datos)

        poblacion = np.concatenate([elite, hijos])

        generacion += 1

    aptitudes = aptitud(poblacion, datos)
    mejor = poblacion[np.argmax(aptitudes)]
    return mejor, mejores


def un_reinicio(argumentos):
    datos, semilla = argumentos
    np.random.seed(semilla)
    return algoritmo_genetico(datos)

def optimizar(datos, n_reinicios=5, semilla=None):
    
    semillas = np.random.default_rng(semilla).integers(0, 2**31 - 1, n_reinicios)

    
    with Pool() as pool:
        resultados = pool.map(un_reinicio, [(datos, int(s)) for s in semillas])

    return max(resultados, key=lambda r: aptitud(r[0], datos))





if __name__ == "__main__":
    inicio = time.perf_counter()

    archivo = pathlib.Path(__file__).resolve().parent.parent / "Datos" / "Parcelas" / "Parcelas_real.csv"
    datos = cargar_datos(archivo_parcelas=archivo)
    mejor, evolucion = optimizar(datos, n_reinicios=5, semilla = 0)

    nombres = datos.cultivos.nombres
    plan = pd.DataFrame({
        "recinto": datos.parcelas.ids,
        "ha": datos.parcelas.superficie,
        "año 1": [nombres[c] for c in mejor[:, 0]],
        "año 2": [nombres[c] for c in mejor[:, 1]],
        "año 3": [nombres[c] for c in mejor[:, 2]],
    })
    print(plan.to_string(index=False))

    print("\nHectáreas del año 1 por cultivo:")
    print(plan.groupby("año 1")["ha"].sum().round(2))

    margenes, violaciones = evaluar_plan(mejor, datos)
    print("\nViolaciones:", violaciones)
    tabla = margenes_por_campaña(mejor, datos)
    año1 = tabla[:, 0]
    print("Año 1 (bueno, mediocre, malo):", año1.round(2))
    print("Año 1 esperado:", np.dot(datos.tipo_anio.probabilidades, año1).round(2))
    print(f"Tiempo de optimizar: {time.perf_counter() - inicio:.1f} s")


