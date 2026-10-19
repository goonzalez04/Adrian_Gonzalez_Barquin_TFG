import numpy as np
import pandas as pd

def repetir_familia_cultivo(secuencia, familia, maximo, desde=0):
   
    violaciones = 0
    for parcela in range(secuencia.shape[0]):
        count = 0
        for año in range(secuencia.shape[1]):
            if familia[secuencia[parcela, año]]:
                count += 1
                if count > maximo and año >= desde:
                    violaciones += 1

            else:
                count = 0
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
    # anexo II del RD 1049/2022
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

    violaciones = 0
    for campaña in range(plan.shape[1]):
        por_cultivo = np.bincount(plan[:, campaña], weights=superficie,
                                  minlength=n_cultivos)
        ordenado = np.sort(por_cultivo)[::-1]

        if (por_cultivo > 0).sum() < min_cultivos:
            violaciones += 1
        if ordenado[0] / total > max_principal:
            violaciones += 1
        if max_dos is not None and ordenado[:2].sum() / total > max_dos:
            violaciones += 1
    return violaciones
