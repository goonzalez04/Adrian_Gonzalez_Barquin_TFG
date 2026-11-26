import numpy as np
from numpy.testing import temppath
import pandas as pd
import pytest
from datos import cargar_datos, coste_por_labores

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


def test_coste_por_labores_suma_maquina_contrato_y_producto(tmp_path):
    (tmp_path / "Labores.csv").write_text(
        "labor,maquina,litros_gasoil_ha,desgaste_apero_eur_ha,contrato_eur_ha\n"
        "propia,maq_propia,10,5,0\n"
        "contratada,maq_contratada,0,0,40\n")
    (tmp_path / "Maquinaria.csv").write_text(
        "maquina,propia\n"
        "maq_propia,si\n"
        "maq_contratada,no\n")
    (tmp_path / "Labores_familia.csv").write_text(
        "familia,labor,pasadas,producto_eur_ha\n"
        "A,propia,2,0\n"
        "A,contratada,1,10\n"
        "B,propia,1,0\n")

    coste = coste_por_labores(tmp_path, pd.Series(["A", "B", "A"]), precio_gasoil=1, mant_por_litro=0.5)

    
    assert coste == pytest.approx([90, 20, 90])
