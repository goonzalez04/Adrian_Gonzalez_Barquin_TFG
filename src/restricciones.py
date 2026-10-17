import numpy as np
import pandas as pd

def repetir_familia_cultivo(secuencia, familia, maximo):

    violaciones = 0
    for parcela in range(secuencia.shape[0]):
        count = 0
        for año in range(secuencia.shape[1]):
            if familia[secuencia[parcela, año]]:
                count += 1
                if count > maximo:
                    violaciones += 1

            else:
                count = 0
    return violaciones


def diversificacion_pac(plan, superficie, n_cultivos):
    total = superficie.sum()

    if total > 30:
        min_cultivos, max_principal, max_dos = 3, 0.75, 0.95
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
