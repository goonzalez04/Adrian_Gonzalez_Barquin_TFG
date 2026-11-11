from fastapi import FastAPI
from datos import cargar_datos

app = FastAPI()

@app.get("/saludo")
async def saludo():
    return {"message": "¡Hola! Bienvenido a la API."}


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