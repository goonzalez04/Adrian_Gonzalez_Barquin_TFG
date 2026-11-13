import numpy as np
from numpy.testing import temppath
import pytest
from datos import cargar_datos

def test_carga_las_5_parcelas_de_prueba():
    datos = cargar_datos()
    assert datos.parcelas.n == 20
    assert datos.parcelas.historial.shape == (20,3)

def test_las_probabilidades_de_tipo_anio_suman_1():
    datos = cargar_datos()
    suma =datos.tipo_anio.probabilidades.sum()
    assert  suma == pytest.approx(1)


def test_beneficio_tiene_una_fila_por_tipo_anio_y_columna_por_cultivo():
    datos = cargar_datos()
    assert datos.cultivos.beneficio_ha.shape == (len(datos.tipo_anio.estados), datos.cultivos.n)

def test_el_historial_solo_tiene_cultivos_validos():
    datos = cargar_datos()
    assert datos.parcelas.historial.min() >= 0
    assert datos.parcelas.historial.max() < datos.cultivos.n


def test_acepta_historial_de_una_sola_campaña(tmp_path):
    archivo = tmp_path / "parcelas.csv"
    archivo.write_text("id,superficie_ha,anio_-1\n" \
    "A,10,trigo\n" \
    "B,5,titarro\n")
    datos = cargar_datos(archivo_parcelas=archivo)
    assert datos.parcelas.historial.shape == (2,1)


def test_ordena_el_historial_del_año_mas_antiguo_al_mas_reciente(tmp_path):
    archivo = tmp_path / "parcelas.csv"
    archivo.write_text("id,superficie_ha,anio_-1,anio_-3,anio_-2\n" \
    "A,10,trigo,cebada,titarro\n")
    datos = cargar_datos(archivo_parcelas=archivo)
    nombres = datos.cultivos.nombres
    fila = [nombres[i] for i in datos.parcelas.historial[0]]
    assert fila == ["cebada", "titarro", "trigo"]
