import numpy as np
from src.data.feeds import OHLCVArrays, _concat_ohlcv_arrays, _sort_and_dedup_ohlcv


def test_concat_sorts_and_deduplicates():
    # Datos simulando concatenación de ficheros con timestamps desordenados y duplicados
    arr1 = OHLCVArrays(
        ts=np.array([3, 1], dtype=np.int64),
        o=np.array([30.0, 10.0]),
        h=np.array([31.0, 11.0]),
        low=np.array([29.0, 9.0]),
        c=np.array([30.5, 10.5]),
        v=np.array([300, 100]),
    )
    arr2 = OHLCVArrays(
        ts=np.array([2, 3], dtype=np.int64),
        o=np.array([20.0, 300.0]),  # valor alternativo para el timestamp duplicado
        h=np.array([21.0, 310.0]),
        low=np.array([19.0, 290.0]),
        c=np.array([20.5, 305.0]),
        v=np.array([200, 3000]),
    )

    merged = _concat_ohlcv_arrays([arr1, arr2])

    # Los timestamps deben quedar ordenados y sin duplicados, conservando la última barra de cada ts
    np.testing.assert_array_equal(merged.ts, np.array([1, 2, 3], dtype=np.int64))

    # Para ts=3 debe conservarse la última barra llegada (del segundo array)
    np.testing.assert_allclose(merged.o, np.array([10.0, 20.0, 300.0]))
    np.testing.assert_allclose(merged.h, np.array([11.0, 21.0, 310.0]))
    np.testing.assert_allclose(merged.low, np.array([9.0, 19.0, 290.0]))
    np.testing.assert_allclose(merged.c, np.array([10.5, 20.5, 305.0]))
    np.testing.assert_allclose(merged.v, np.array([100, 200, 3000]))


def test_sort_and_dedup_preserves_alignment():
    data = OHLCVArrays(
        ts=np.array([5, 5, 4, 6], dtype=np.int64),
        o=np.array([1, 2, 3, 4], dtype=float),
        h=np.array([10, 20, 30, 40], dtype=float),
        low=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
        c=np.array([11, 22, 33, 44], dtype=float),
        v=np.array([100, 200, 300, 400], dtype=float),
    )

    cleaned = _sort_and_dedup_ohlcv(data)

    # Índice ordenado y sin duplicados, conservando el último de cada timestamp
    np.testing.assert_array_equal(cleaned.ts, np.array([4, 5, 6], dtype=np.int64))
    np.testing.assert_allclose(cleaned.o, np.array([3, 2, 4], dtype=float))
    np.testing.assert_allclose(cleaned.h, np.array([30, 20, 40], dtype=float))
    np.testing.assert_allclose(cleaned.low, np.array([0.3, 0.2, 0.4], dtype=float))
    np.testing.assert_allclose(cleaned.c, np.array([33, 22, 44], dtype=float))
    np.testing.assert_allclose(cleaned.v, np.array([300, 200, 400], dtype=float))
