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