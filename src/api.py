import pathlib
import uuid
import numpy as np

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from datos import cargar_datos
from algoritmo import optimizar, margenes_por_campaña, evaluar_plan

app = FastAPI()
RUTA_WEB = pathlib.Path(__file__).resolve().parent.parent / "web" 
RUTA_DATOS = pathlib.Path(__file__).resolve().parent.parent / "Datos"

trabajos = {}



@app.get("/cultivos")
def cultivos():
    datos = cargar_datos()
    c = datos.cultivos
    resultado = []
    for i in range(c.n):
        resultado.append({
            "nombre": c.nombres[i],
            "dosis_kg_ha": float(c.dosis[i]),
            "stock_semilla_kg": float(c.stock_semilla[i]),
            "retorno_min_anios": int(c.retorno_min[i])
        })

    return resultado


@app.get("/parcelas")
def parcelas():
    datos = cargar_datos()
    p = datos.parcelas
    nombres = datos.cultivos.nombres
    resultado = []
    for i in range(p.n):
        resultado.append({
            "id" : p.ids[i],
            "superficie_ha": float(p.superficie[i]),
            "historial": [nombres[c] for c in p.historial[i]]
        })

    return resultado



@app.get("/")
def web():
    return FileResponse(RUTA_WEB / "index.html")


def calcular_plan(id_trabajo):
    try:
        datos = cargar_datos()
        mejor, _ = optimizar(datos, n_reinicios=5)

        nombres = datos.cultivos.nombres
        superficie = datos.parcelas.superficie

        # El plan, parcela a parcela: cultivo de los años 1, 2 y 3
        plan = []
        for i in range(datos.parcelas.n):
            plan.append({
                "id": datos.parcelas.ids[i],
                "superficie_ha": float(superficie[i]),
                "cultivos": [nombres[c] for c in mejor[i]]
            })

        # Hectáreas de cada cultivo en el año 1
        ha = np.bincount(mejor[:, 0], weights=superficie, minlength=datos.cultivos.n)
        hectareas = {nombres[j]: round(float(ha[j]), 2) for j in range(datos.cultivos.n)}

        # Margen esperado del año 1
        tabla = margenes_por_campaña(mejor, datos)
        margen = float(datos.tipo_anio.probabilidades @ tabla[:, 0])

        _, violaciones = evaluar_plan(mejor, datos)

        trabajos[id_trabajo] = {
            "estado": "terminado",
            "plan": plan,
            "hectareas": hectareas,
            "margen_esperado": round(margen, 2),
            "violaciones": int(violaciones)
        }
    except Exception as error:
        trabajos[id_trabajo] = {"estado": "error", "detalle": str(error)}


@app.post("/optimizar")
def optimizar_plan(tareas: BackgroundTasks):
    id_trabajo = str(uuid.uuid4())
    trabajos[id_trabajo] = {"estado": "calculando"}
    tareas.add_task(calcular_plan, id_trabajo)
    return {"id": id_trabajo}

@app.get("/optimizar/{id_trabajo}")
def estado_plan(id_trabajo: str):
    if id_trabajo not in trabajos:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    return trabajos[id_trabajo]



@app.get("/recintos")
def recintos():
    return FileResponse(RUTA_DATOS / "Parcelas.geojson",
                        media_type="application/geo+json")