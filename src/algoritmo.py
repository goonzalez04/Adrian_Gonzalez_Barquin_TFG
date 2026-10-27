import numpy as np

from restricciones import repetir_familia_cultivo, diversificacion_pac, rotacion_cultivos
from datos import cargar_datos




def plan_aleatorio(datos):
    return np.random.randint(0, datos.cultivos.n, size=(datos.parcelas.n, datos.n_campañas))


def poblacion_inicial(datos, n_individuos):
    return np.array([plan_aleatorio(datos) for _ in range(n_individuos)])

def ayudas_fijas(datos):
    a = datos.ayudas

    return a.basica_anual + a.redistributiva_anual + a.ecorregimen_anual

def margenes_por_campaña(plan, datos):
    secuencia = np.hstack([datos.parcelas.historial, plan])
    n_historial = datos.parcelas.historial.shape[1]

    c = datos.cultivos
    superficie = datos.parcelas.superficie

    # Una fila por tipo de año (bueno, mediocre, malo), una columna por campaña
    tabla = np.zeros((len(datos.tipo_anio.estados), datos.n_campañas))      # ←
   

    for t in range(datos.n_campañas):
        stock = c.stock_semilla.copy()
        ha = np.bincount(plan[:, t], weights=superficie, minlength=c.n)

        comprar = np.maximum(0, c.dosis * ha - stock)
        coste = (comprar * c.precio_semilla).sum()                         # ← coste de esta campaña
        

        # Abono: un cultivo sembrado tras una leguminosa necesita menos
        anterior = secuencia[:, n_historial + t - 1]
        tras_leguminosa = c.es_leguminosa[anterior]
        factor = np.where(tras_leguminosa, 1 - datos.reduccion_abono, 1)
        abono_kg = c.abono_kg_ha[plan[:, t]] * factor * superficie
        coste += abono_kg.sum() * datos.precio_abono

        tabla[:, t] = c.beneficio_ha @ ha - coste + ayudas_fijas(datos)    # ←

    return tabla


def evaluar_plan(plan, datos):
    secuencia = np.hstack([datos.parcelas.historial, plan])
    n_historial = datos.parcelas.historial.shape[1]

    
    pesos = datos.descuento_anual ** np.arange(datos.n_campañas)
    margenes = margenes_por_campaña(plan, datos) @ pesos


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
        aptitudes = [aptitud(plan, datos) for plan in poblacion]
        mejor_de_esta = max(aptitudes)
        mejores.append(mejor_de_esta)

        if mejor_de_esta > mejor_hasta_ahora:
            mejor_hasta_ahora = mejor_de_esta
            sin_mejora = 0
        else:
            sin_mejora += 1

        orden = np.argsort(aptitudes)[::-1]
        nueva = [poblacion[i] for i in orden[:n_elite]]

        while len(nueva) < n_individuos:
            padre = torneo(poblacion, aptitudes)
            madre = torneo(poblacion, aptitudes)
            hijo = cruza(padre, madre)
            hijo = mutar(hijo, tasa, datos)
            nueva.append(hijo)

        poblacion = np.array(nueva)
        generacion += 1

    aptitudes = [aptitud(plan, datos) for plan in poblacion]
    mejor = poblacion[np.argmax(aptitudes)]
    return mejor, mejores


def optimizar(datos, n_reinicios=5):
    mejor_plan = None
    mejor_aptitud = -np.inf
    mejor_evolucion = None

    for i in range(n_reinicios):
        plan, evolucion = algoritmo_genetico(datos)
        a= aptitud(plan, datos)
        if a > mejor_aptitud:
            mejor_plan = plan
            mejor_aptitud = a
            mejor_evolucion = evolucion
    
    return mejor_plan, mejor_evolucion




if __name__ == "__main__":
    datos = cargar_datos()
    mejor, evolucion = optimizar(datos)
    print(mejor)
    margenes, violaciones = evaluar_plan(mejor, datos)
    tabla = margenes_por_campaña(mejor, datos)
    print("Margen por campaña (filas: bueno, mediocre, malo):")
    print(tabla.round(2))
    año1 = tabla[:, 0]
    print("Año 1 esperado:", np.dot(datos.tipo_anio.probabilidades, año1).round(2))


