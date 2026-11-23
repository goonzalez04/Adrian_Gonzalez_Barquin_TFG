import pytest

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


def test_pagina_pricipal():
    respuesta = cliente.get("/")
    assert respuesta.status_code == 200
    assert 'mapa' in respuesta.text


def test_optimizar_devuelve_un_plan_sin_violaciones():
    id_trabajo = cliente.post("/optimizar").json()["id"]
    resultado = cliente.get(f"/optimizar/{id_trabajo}").json()
    datos = cargar_datos()

    assert resultado["estado"] == "terminado"
    assert resultado["violaciones"] == 0
    assert len(resultado["plan"]) == datos.parcelas.n
    assert sum(resultado["hectareas"].values()) == pytest.approx(datos.parcelas.superficie.sum(), abs=0.1)

def test_optimizar_con_id_que_no_existe_da_404():
    respuesta = cliente.get("/optimizar/no-existe")
    assert respuesta.status_code == 404


def test_recintos_toene_un_contorno_por_parcela():
    respuesta = cliente.get("/recintos")
    assert respuesta.status_code == 200

    geojson = respuesta.json()
    datos = cargar_datos()
    ids_recintos = [f["properties"]["id"] for f in geojson["features"]]
    assert len(ids_recintos) == datos.parcelas.n
    assert set(ids_recintos) == set(datos.parcelas.ids)