import numpy as np
import pandas as pd

def repetir_familia_cultivo(secuencia, familia, maximo, desde=0):
    es_familia = familia[secuencia]
    contador = np.zeros(secuencia.shape[:-1], dtype=int)
    violaciones = 0
    for año in range(secuencia.shape[-1]):
        contador = np.where(es_familia[..., año], contador + 1, 0)
        if año >= desde:
            violaciones += (contador > maximo).sum(axis=-1)
    return violaciones

   
    


def rotacion_cultivos(secuencia, superficie, n_cultivos, max_anios=3, desde=0):
    # BCAM 7 (modalidad A): un mismo cultivo no puede ocupar una parcela más de
    # max_anios años seguidos. Las explotaciones de 10 ha o menos están exentas.
    if superficie.sum() <= 10:
        return 0

    violaciones = 0
    for c in range(n_cultivos):
        es_c = np.arange(n_cultivos) == c
        violaciones += repetir_familia_cultivo(secuencia, es_c, max_anios, desde)
    return violaciones


def diversificacion_pac(plan, superficie, n_cultivos):
    # BCAM 7 (modalidad A): diversificación anual de cultivos según el
    # anexo II del RD 1049/2022. Sirve para un plan (P, T) o para la
    # población entera (I, P, T).
    total = superficie.sum()

    # Las explotaciones de 10 ha o menos están exentas
    if total <= 10:
        return 0

    if total > 30:
        min_cultivos, max_principal, max_dos = 3, 0.70, 0.90
    elif total > 20:
        min_cultivos, max_principal, max_dos = 2, 0.70, None
    else:
        min_cultivos, max_principal, max_dos = 2, 0.75, None

    # Hectáreas de cada cultivo en cada campaña: (T, C) con un plan, (I, T, C) con la población
    es_cultivo = plan[..., None] == np.arange(n_cultivos)
    por_cultivo = (es_cultivo * superficie[:, None, None]).sum(axis=-3)
    ordenado = np.sort(por_cultivo, axis=-1)[..., ::-1]      # de mayor a menor

    # Reglas incumplidas en cada campaña (0 a 3)
    malas = ((por_cultivo > 0).sum(axis=-1) < min_cultivos).astype(int)
    malas += ordenado[..., 0] / total > max_principal
    if max_dos is not None:
        malas += ordenado[..., :2].sum(axis=-1) / total > max_dos

    # Suma de las campañas: un número por plan
    return malas.sum(axis=-1)


def retorno_minimo(secuencia, retorno_min, desde=0):
    # Un cultivo no puede volver a la misma parcela hasta que pasen
    # retorno_min años (titarro 4: si se siembra en 2026, no vuelve hasta 2030)
    violaciones = 0
    for anio in range(desde, secuencia.shape[-1]):
        cultivo = secuencia[..., anio]
        espera = retorno_min[cultivo]
        for s in range(1, retorno_min.max()):
            if anio - s < 0:
                break
            repetido = secuencia[..., anio - s] == cultivo
            violaciones += (repetido & (s < espera)).sum(axis=-1)
    return violaciones