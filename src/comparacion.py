import numpy as np
import matplotlib.pyplot as plt
from algoritmo import (algoritmo_genetico, plan_aleatorio, evaluar_plan, aptitud, n_parcelas, n_campañas, historial)
from datos import cargar_datos

def busqueda_aleatoria(datos, n_evaluaciones):
    mejor_plan = None
    mejor_aptitud = -np.inf
    for i in range(n_evaluaciones):
        plan = plan_aleatorio(datos)
        apt = aptitud(plan, datos)
        if mejor_plan is None or apt > mejor_aptitud:
            mejor_plan = plan
            mejor_aptitud = apt
    return mejor_plan

def repetir_con_semilla(datos, n_semillas):
    aptitudes_ga = []
    aptitudes_aleatorio = []

    for i in range(n_semillas):
        np.random.seed(i)
        mejor_ga, _ = algoritmo_genetico(datos, 100, 200, 0.1, 2)
        aptitudes_ga.append(aptitud(mejor_ga, datos))

        np.random.seed(i)
        mejor_aleatorio = busqueda_aleatoria(datos, 20000)
        aptitudes_aleatorio.append(aptitud(mejor_aleatorio, datos))

    return np.array(aptitudes_ga), np.array(aptitudes_aleatorio)




if __name__ == "__main__":
    datos = cargar_datos()
    todo_trigo = np.zeros((datos.parcelas.n, datos.n_campañas), dtype=int)
    rotacion_pasada = datos.parcelas.historial
    mejor_aleatorio = busqueda_aleatoria(datos, 20000)
    mejor_ga, evolucion = algoritmo_genetico(datos, 100, 200, 0.1, 2)

    # Repetir con varias semillas para ver si el resultado es estable
    aptitudes_ga, aptitudes_aleatorio = repetir_con_semilla(datos, 2)

    print(f"{'método':<22}{'media':>12}{'mínimo':>12}{'máximo':>12}{'desv.':>10}")
    for nombre, aptitudes in [("Algoritmo genético", aptitudes_ga),
                              ("Búsqueda aleatoria", aptitudes_aleatorio)]:
        print(f"{nombre:<22}{aptitudes.mean():>12.2f}{aptitudes.min():>12.2f}"
              f"{aptitudes.max():>12.2f}{aptitudes.std():>10.2f}")

    plt.figure(figsize=(9, 5))
    plt.plot(evolucion, label="Algoritmo genético (mejor de cada generación)")
    plt.axhline(aptitud(mejor_aleatorio, datos), color="gray", linestyle="--",
                label="Mejor de 20 000 planes aleatorios")
    plt.axhline(aptitud(rotacion_pasada, datos), color="orange", linestyle=":",
                label="Rotación del año pasado")
    plt.xlabel("Generación")
    plt.ylabel("Aptitud")
    plt.title("Evolución de la mejor aptitud")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
