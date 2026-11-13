from fastapi.testclient import TestClient
from api import app
from datos import cargar_datos


cliente = TestClient(app)

def test_cultivos_devuelve_todos_los_cultivos():
    datos = cargar_datos()
    lista = cliente.get("/cultivos").json()
    nombres = [fila["nombre"] for fila in lista]
    assert len(lista) == datos.cultivos.n
    assert nombres == datos.cultivos.nombres


def test_parcelas_tiene_superficie_e_historial():
    lista = cliente.get("/parcelas").json()
    datos = cargar_datos()
    assert len(lista)==datos.parcelas.n
    for p in lista:
        assert len(p["historial"]) == datos.parcelas.historial.shape[1]