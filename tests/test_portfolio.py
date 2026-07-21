import numpy as np
import polars as pl
import pytest

from lifeflow import Decrement, Inforce, Portfolio, Timeline


def _df():
    return pl.DataFrame(
        {
            "NPOL": [101, 102, 103],
            "duration": [12, 12, 12],
            "capital": [1000.0, 2000.0, 3000.0],
        }
    )


def test_cols_expone_cada_columna_como_array():
    """cols es de donde @grid y Decrement leen las variables por nombre."""
    p = Portfolio(_df(), id_col="NPOL")

    assert set(p.cols) == {"NPOL", "duration", "capital"}
    for nombre, valores in p.cols.items():
        assert isinstance(valores, np.ndarray), f"{nombre} no es un array"
    assert np.array_equal(p.cols["capital"], np.array([1000.0, 2000.0, 3000.0]))


def test_ids_devuelve_los_identificadores():
    p = Portfolio(_df(), id_col="NPOL")
    assert np.array_equal(p.ids, np.array([101, 102, 103]))


def test_selectid_aisla_la_poliza_y_sigue_siendo_proyectable():
    p = Portfolio(_df(), id_col="NPOL")
    sub = p.selectid(102)

    assert isinstance(sub, Portfolio)
    assert np.array_equal(sub.ids, np.array([102]))
    assert np.array_equal(sub.cols["capital"], np.array([2000.0]))

    # lo que de verdad importa: el sub-portfolio se puede proyectar
    tl = Timeline("duration")
    inforce = Inforce(sub, [Decrement(np.full(13, 0.01), tl)])
    assert inforce.inf.shape == (1, 12)
    assert np.isclose(inforce.inf[0, 0], 0.99)


def test_selectid_conserva_el_id_col():
    """El sub-portfolio debe seguir sabiendo cuál es su columna de id."""
    p = Portfolio(_df(), id_col="NPOL")
    assert p.selectid(102).selectid(102).ids == np.array([102])


def test_selectid_con_id_inexistente_devuelve_cartera_vacia():
    """Comportamiento deliberado: filtrar sin coincidencias da vacío, como en
    polars, en lugar de lanzar. Si algún día se prefiere que avise, este test
    es el que hay que cambiar."""
    p = Portfolio(_df(), id_col="NPOL")
    vacio = p.selectid(999)

    assert isinstance(vacio, Portfolio)
    assert len(vacio.df) == 0
    assert vacio.cols["NPOL"].size == 0


def test_ids_sin_id_col_lanza():
    p = Portfolio(_df())
    with pytest.raises(ValueError, match="id_col"):
        p.ids


def test_selectid_sin_id_col_lanza():
    p = Portfolio(_df())
    with pytest.raises(ValueError, match="id_col"):
        p.selectid(101)
