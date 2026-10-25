import numpy as np

from restricciones import repetir_familia_cultivo, diversificacion_pac, rotacion_cultivos
from datos import cargar_datos




def plan_aleatorio(datos):
    return np.random.randint(0, datos.cultivos.n, size=(datos.parcelas.n, datos.n_campañas))


def poblacion_inicial(datos, n_individuos):
    return np.array([plan_aleatorio(datos) for _ in range(n_individuos)])

def evaluar_plan(plan, datos):
    secuencia = np.hstack([datos.parcelas.historial, plan])
    n_historial = datos.parcelas.historial.shape[1]

    c = datos.cultivos
    superficie = datos.parcelas.superficie
    n_estados = len(datos.tipo_anio.estados)
    margenes = np.zeros(n_estados)

    for s in range(n_estados):
        stock = c.stock_semilla.copy()
        for t in range(datos.n_campañas):
            ha = np.bincount(plan[:,t], weights=superficie, minlength=c.n)
            margenes[s] += (c.beneficio_ha[s]*ha).sum()
            comprar = np.maximum(0, c.dosis*ha-stock)
            margenes[s] -= (comprar * c.precio_semilla).sum()
            stock = np.zeros(c.n)

            # Abono: un cultivo sembrado tras una leguminosa necesita menos
            anterior = secuencia[:, n_historial + t - 1]
            tras_leguminosa = c.es_leguminosa[anterior]
            factor = np.where(tras_leguminosa, 1 - datos.reduccion_abono, 1)
            abono_kg = c.abono_kg_ha[plan[:, t]] * factor * superficie
            margenes[s] -= abono_kg.sum() * datos.precio_abono

    violaciones = (repetir_familia_cultivo(secuencia, datos.cultivos.es_cereal, 2, n_historial)
                   + repetir_familia_cultivo(secuencia, datos.cultivos.es_leguminosa, 1, n_historial))

    # BCAM 7 de la PAC (modalidad A): rotación de cultivos y diversificación
    violaciones += rotacion_cultivos(secuencia, datos.parcelas.superficie, datos.cultivos.n, desde=n_historial)
    violaciones += diversificacion_pac(plan, datos.parcelas.superficie, datos.cultivos.n)

    return margenes, violaciones



def aptitud(plan, datos):
    margenes, violaciones = evaluar_plan(plan, datos)
    esperado = np.dot(datos.tipo_anio.probabilidades, margenes)
    malo = margenes[datos.tipo_anio.estados.index("malo")]
    riesgo = esperado - malo
    return esperado - datos.riesgo * riesgo - 10000 * violaciones


def torneo(poblacion, aptitudes, k=3):
    n_individuos = len(poblacion)
    indices = np.random.choice(n_individuos, size=k, replace=False)
    mejor = indices[0]
    for i in indices[1:]:
        if aptitudes[i] > aptitudes[mejor]:
            mejor = i
    return poblacion[mejor]

def cruza(padre, madre):
    mascara = np.random.rand(padre.shape[0])< 0.5
    return np.where(mascara[:, None], padre, madre)


def mutar(plan, tasa, datos):
    mascara = np.random.rand(*plan.shape) < tasa
    nuevos = np.random.randint(0, datos.cultivos.n, size=plan.shape)
    return np.where(mascara, nuevos, plan)


def algoritmo_genetico(datos, n_individuos, generaciones, tasa, n_elite):
    poblacion = poblacion_inicial(datos, n_individuos)
    mejores = []

    for g in range(generaciones):
        aptitudes = [aptitud(plan, datos) for plan in poblacion]
        mejores.append(max(aptitudes))
        
        orden = np.argsort(aptitudes)[::-1]
        nueva = [poblacion[i] for i in orden[:n_elite]]

        while len(nueva) < n_individuos:
            padre = torneo(poblacion, aptitudes)
            madre = torneo(poblacion, aptitudes)
            hijo = cruza(padre, madre)
            hijo = mutar(hijo, tasa, datos)
            nueva.append(hijo)

        poblacion = np.array(nueva)

    aptitudes = [aptitud(plan, datos) for plan in poblacion]
    mejor = poblacion[np.argmax(aptitudes)]
    return mejor, mejores




if __name__ == "__main__":
    datos = cargar_datos()
    mejor, evolucion = algoritmo_genetico(datos, 100, 200, 0.1, 2)
    print(mejor)
    margenes, violaciones = evaluar_plan(mejor, datos)
    print("Márgenes (bueno, mediocre, malo):", margenes)
    print("Margen esperado:", np.dot(datos.tipo_anio.probabilidades, margenes))

