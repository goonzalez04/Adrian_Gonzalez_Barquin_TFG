import numpy as np
import matplotlib.pyplot as plt
from algoritmo import (algoritmo_genetico, plan_aleatorio, evaluar_plan, aptitud, n_parcelas, n_campañas, historial )


def busqueda_aleatoria(n_evaluaciones):
    mejor_plan = None
    mejor_aptitud = -np.inf
    for i in range(n_evaluaciones):
        plan = plan_aleatorio()
        apt = aptitud(plan)
        if mejor_plan is None or apt > mejor_aptitud:
            mejor_plan = plan
            mejor_aptitud = apt
    return mejor_plan




if __name__ == "__main__":
    todo_trigo = np.zeros((n_parcelas, n_campañas), dtype=int)
    rotacion_pasada = historial
    mejor_aleatorio = busqueda_aleatoria(20000)
    mejor_ga, evolucion = algoritmo_genetico(100, 200, 0.1, 2)



    
    

    plt.figure(figsize=(9, 5))
    plt.plot(evolucion, label="Algoritmo genético (mejor de cada generación)")
    plt.axhline(aptitud(mejor_aleatorio), color="gray", linestyle="--",
                label="Mejor de 20 000 planes aleatorios")
    plt.axhline(aptitud(rotacion_pasada), color="orange", linestyle=":",
                label="Rotación del año pasado")
    plt.xlabel("Generación")
    plt.ylabel("Aptitud")
    plt.title("Evolución de la mejor aptitud")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.show()
