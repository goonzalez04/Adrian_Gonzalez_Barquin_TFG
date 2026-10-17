import numpy as np
import pandas as pd


from restricciones import repetir_familia_cultivo, diversificacion_pac

parcelas = pd.read_csv("Datos/Parcelas.csv")
cultivos = pd.read_csv("Datos/Cultivos.csv")

cultivos["beneficio_ha"] = (cultivos["rendimiento_toneladas_ha"] * cultivos["precio_tonelada"]+cultivos["ayuda_PAC_ha"]
- cultivos["coste_ha"])

beneficio_ha = cultivos["beneficio_ha"].to_numpy()

es_cereal = (cultivos["familia"] == "cereal").to_numpy()
es_leguminosa = (cultivos["familia"] == "leguminosa").to_numpy()


n_cultivos = len(cultivos)
n_parcelas = len(parcelas)
n_campañas = 3

referencia = {nombre: i for i, nombre in enumerate(cultivos["cultivo"])}
historial = []
for p in range(n_parcelas):
    historial.append([referencia[c] for c in parcelas.loc[p, ["anio_-3", "anio_-2", "anio_-1"]]])

historial = np.array(historial)

superficie = parcelas["superficie_ha"].to_numpy()


def plan_aleatorio():
    return np.random.randint(0, n_cultivos, size=(n_parcelas, n_campañas))


def poblacion_inicial(n_individuos):
    return np.array([plan_aleatorio() for _ in range(n_individuos)])


def aptitud(plan):
    margen = (beneficio_ha[plan] * superficie[:, None]).sum()

    secuencia = np.hstack([historial, plan])

    #Restriccion de no repetir cultivos de la misma familia más de un número máximo de veces consecutivas
    violaciones = (repetir_familia_cultivo(secuencia, es_cereal, 2)
                   + repetir_familia_cultivo(secuencia, es_leguminosa, 1))

    # Diversificación de la PAC
    violaciones += diversificacion_pac(plan, superficie, n_cultivos)

    return margen - 10000 * violaciones

   



def torneo(poblacion, aptitudes, k=3):
    n_individuos = len(poblacion)
    indices = np.random.choice(n_individuos, size=k, replace=False)
    mejor = indices[0]
    for i in indices[1:]:
        if aptitudes[i] > aptitudes[mejor]:
            mejor = i
    return poblacion[mejor]

def cruza(padre, madre):
    mascara = np.random.rand(n_parcelas)< 0.5
    return np.where(mascara[:, None], padre, madre)


def mutar(plan, tasa):
    mascara = np.random.rand(n_parcelas, n_campañas) < tasa
    nuevos = np.random.randint(0, n_cultivos, size=(n_parcelas, n_campañas))
    return np.where(mascara, nuevos, plan)


def algoritmo_genetico(n_individuos, generaciones, tasa, n_elite):
    poblacion = poblacion_inicial(n_individuos)
    mejores = []

    for g in range(generaciones):
        aptitudes = [aptitud(plan) for plan in poblacion]
        mejores.append(max(aptitudes))
        
        orden = np.argsort(aptitudes)[::-1]
        nueva = [poblacion[i] for i in orden[:n_elite]]

        while len(nueva) < n_individuos:
            padre = torneo(poblacion, aptitudes)
            madre = torneo(poblacion, aptitudes)
            hijo = cruza(padre, madre)
            hijo = mutar(hijo, tasa)
            nueva.append(hijo)

        poblacion = np.array(nueva)

    aptitudes = [aptitud(plan) for plan in poblacion]
    mejor = poblacion[np.argmax(aptitudes)]
    return mejor, mejores


mejor, evolucion = algoritmo_genetico(100, 200, 0.1, 2)
print(mejor)
print("Aptitud final:", aptitud(mejor))
print("Primera generación:", evolucion[0], "Última:", evolucion[-1])
