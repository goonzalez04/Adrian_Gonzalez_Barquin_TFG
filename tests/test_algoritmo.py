import numpy as np
import pytest

from algoritmo import margenes_por_campaña, evaluar_plan, aptitud, cruza, mutar, optimizar
from datos import Cultivos, Parcelas, Tipo_Anio, Ayudas, Datos, cargar_datos

TRIGO, TITARRO = 0, 1


def datos_a_mano(superficie, historial, n_campañas=1, beneficio_trigo=(0, 0, 0),
                 stock_trigo=1000, abono_trigo=0, ayudas=(0, 0, 0), descuento=1.0, riesgo=0.5):
    # Datos pequeños con números redondos, para comprobar los resultados de cabeza.
    # Dos cultivos: trigo (cereal) y titarro (leguminosa). Por defecto no hay
    # beneficio, abono ni ayudas, así que el margen solo recoge lo que se prueba.
    beneficio_ha = np.array([[beneficio_trigo[s], 0.0] for s in range(3)])
    cultivos = Cultivos(
        nombres=["trigo", "titarro"],
        beneficio_ha=beneficio_ha,
        es_cereal=np.array([True, False]),
        es_leguminosa=np.array([False, True]),
        dosis=np.array([250.0, 200.0]),
        precio_semilla=np.array([1.0, 1.0]),
        stock_semilla=np.array([float(stock_trigo), 10000.0]),
        abono_kg_ha=np.array([float(abono_trigo), 0.0]),
        cosecha_kg_ha=np.zeros((3, 2)),
        retorno_min=np.array([1, 4]),
    )
    parcelas = Parcelas(
        ids=[f"P{i}" for i in range(len(superficie))],
        superficie=np.array(superficie, dtype=float),
        historial=np.array(historial),
    )
    tipo_anio = Tipo_Anio(["bueno", "mediocre", "malo"], np.array([0.25, 0.5, 0.25]))
    return Datos(cultivos, parcelas, tipo_anio, reduccion_abono=0.3, precio_abono=1.0,
                 ayudas=Ayudas(*ayudas), descuento_anual=descuento,
                 n_campañas=n_campañas, riesgo=riesgo)


# --- Semilla ---

def test_semilla_del_stock_no_se_paga():
    # 4 ha × 250 kg/ha = 1000 kg, justo lo que hay en el stock
    datos = datos_a_mano(superficie=[4], historial=[[TITARRO]], stock_trigo=1000)
    plan = np.array([[TRIGO]])
    assert margenes_por_campaña(plan, datos)[:, 0] == pytest.approx([0, 0, 0])


def test_semilla_que_falta_se_compra():
    # 6 ha × 250 kg/ha = 1500 kg; hay 1000 en stock, se compran 500 a 1 €/kg
    datos = datos_a_mano(superficie=[6], historial=[[TITARRO]], stock_trigo=1000)
    plan = np.array([[TRIGO]])
    assert margenes_por_campaña(plan, datos)[:, 0] == pytest.approx([-500, -500, -500])


def test_cada_campaña_empieza_con_el_mismo_stock():
    # Trigo, titarro, trigo: en la campaña 3 vuelve a haber 1000 kg de stock
    datos = datos_a_mano(superficie=[6], historial=[[TITARRO]], n_campañas=3, stock_trigo=1000)
    plan = np.array([[TRIGO, TITARRO, TRIGO]])
    tabla = margenes_por_campaña(plan, datos)
    assert tabla[0, 0] == pytest.approx(-500)
    assert tabla[0, 2] == pytest.approx(-500)


# --- Abono ---

def test_cereal_tras_titarro_gasta_un_30_por_ciento_menos_de_abono():
    # 1 ha de trigo con 100 kg de abono a 1 €/kg; stock de semilla de sobra
    tras_titarro = datos_a_mano(superficie=[1], historial=[[TITARRO]], abono_trigo=100)
    tras_trigo = datos_a_mano(superficie=[1], historial=[[TRIGO]], abono_trigo=100)
    plan = np.array([[TRIGO]])
    assert margenes_por_campaña(plan, tras_titarro)[0, 0] == pytest.approx(-70)
    assert margenes_por_campaña(plan, tras_trigo)[0, 0] == pytest.approx(-100)


# --- Ayudas y descuento ---

def test_ayudas_fijas_se_suman_una_vez_en_cada_campaña():
    datos = datos_a_mano(superficie=[1], historial=[[TRIGO]], n_campañas=2, ayudas=(100, 20, 30))
    plan = np.array([[TITARRO, TRIGO]])
    assert margenes_por_campaña(plan, datos) == pytest.approx(np.full((3, 2), 150))


def test_descuento_hace_que_los_años_lejanos_cuenten_menos():
    # 150 € en cada campaña; con descuento 0,5: 150 + 150 × 0,5 = 225
    datos = datos_a_mano(superficie=[1], historial=[[TRIGO]], n_campañas=2,
                         ayudas=(100, 20, 30), descuento=0.5)
    plan = np.array([[TITARRO, TRIGO]])
    margenes, _ = evaluar_plan(plan, datos)
    assert margenes == pytest.approx([225, 225, 225])


def test_con_descuento_1_el_total_es_la_suma_de_las_campañas():
    datos = cargar_datos()
    datos.descuento_anual = 1.0
    np.random.seed(0)
    plan = np.random.randint(0, datos.cultivos.n, size=(datos.parcelas.n, datos.n_campañas))
    margenes, _ = evaluar_plan(plan, datos)
    assert margenes == pytest.approx(margenes_por_campaña(plan, datos).sum(axis=1))


# --- Aptitud ---

def test_cada_violacion_resta_10000():
    # Titarro el año pasado y titarro otra vez: dos violaciones (leguminosa
    # 2 años seguidos y retorno del titarro) y margen 0, así que la aptitud es -20 000
    datos = datos_a_mano(superficie=[1], historial=[[TITARRO]])
    plan = np.array([[TITARRO]])
    assert evaluar_plan(plan, datos)[1] == 2
    assert aptitud(plan, datos) == pytest.approx(-20000)


def test_titarro_antes_de_4_años_es_una_violacion():
    # Titarro hace 3 años y titarro otra vez en el plan: no han pasado 4 años
    datos = datos_a_mano(superficie=[1], historial=[[TITARRO, TRIGO, TRIGO]])
    assert evaluar_plan(np.array([[TITARRO]]), datos)[1] == 1


def test_titarro_a_los_4_años_no_es_violacion():
    datos = datos_a_mano(superficie=[1], historial=[[TITARRO, TRIGO, TRIGO, TRIGO]])
    assert evaluar_plan(np.array([[TITARRO]]), datos)[1] == 0


def test_sin_riesgo_la_aptitud_es_el_margen_esperado():
    # Bueno 300, mediocre 100, malo -100, con probabilidades 0,25 / 0,5 / 0,25: esperado 100
    datos = datos_a_mano(superficie=[1], historial=[[TITARRO]], beneficio_trigo=(300, 100, -100), riesgo=0)
    assert aptitud(np.array([[TRIGO]]), datos) == pytest.approx(100)


def test_con_riesgo_maximo_la_aptitud_es_el_margen_del_año_malo():
    datos = datos_a_mano(superficie=[1], historial=[[TITARRO]], beneficio_trigo=(300, 100, -100), riesgo=1)
    assert aptitud(np.array([[TRIGO]]), datos) == pytest.approx(-100)


# --- Operadores del algoritmo genético ---

def test_mutar_con_tasa_0_no_cambia_el_plan():
    datos = cargar_datos()
    plan = np.zeros((datos.parcelas.n, datos.n_campañas), dtype=int)
    assert (mutar(plan, 0, datos) == plan).all()


def test_mutar_solo_pone_cultivos_que_existen():
    datos = cargar_datos()
    plan = np.zeros((datos.parcelas.n, datos.n_campañas), dtype=int)
    mutado = mutar(plan, 1, datos)
    assert mutado.min() >= 0
    assert mutado.max() < datos.cultivos.n


def test_cruza_cada_parcela_viene_entera_del_padre_o_de_la_madre():
    padre = np.zeros((20, 3), dtype=int)
    madre = np.ones((20, 3), dtype=int)
    hijo = cruza(padre, madre)
    for fila in hijo:
        assert (fila == 0).all() or (fila == 1).all()


# --- Algoritmo completo (más lentas: ejecutan el GA de verdad) ---

def test_misma_semilla_da_el_mismo_plan():
    datos = cargar_datos()
    plan_1, _ = optimizar(datos, n_reinicios=2, semilla=0)
    plan_2, _ = optimizar(datos, n_reinicios=2, semilla=0)
    assert (plan_1 == plan_2).all()


def test_el_plan_optimo_cumple_todas_las_reglas():
    datos = cargar_datos()
    plan, _ = optimizar(datos, n_reinicios=2, semilla=0)
    assert evaluar_plan(plan, datos)[1] == 0
