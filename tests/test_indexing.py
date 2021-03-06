import numpy
import pytest

import rasterio
from rasterio import (
    get_data_window, window_intersection, window_union, windows_intersect
)


DATA_WINDOW = ((3, 5), (2, 6))


def test_index():
    with rasterio.open('tests/data/RGB.byte.tif') as src:
        left, bottom, right, top = src.bounds
        assert src.index(left, top) == (0, 0)
        assert src.index(right, top) == (0, src.width)
        assert src.index(right, bottom) == (src.height, src.width)
        assert src.index(left, bottom) == (src.height, 0)


def test_full_window():
    with rasterio.open('tests/data/RGB.byte.tif') as src:
        left, bottom, right, top = src.bounds
        assert src.window(left, bottom, right, top) == tuple(zip((0, 0), src.shape))


def test_window_no_exception():
    with rasterio.open('tests/data/RGB.byte.tif') as src:
        left, bottom, right, top = src.bounds
        left -= 1000.0
        assert src.window(left, bottom, right, top, boundless=True) == (
                (0, src.height), (-4, src.width))


def test_index_values():
    with rasterio.open('tests/data/RGB.byte.tif') as src:
        assert src.index(101985.0, 2826915.0) == (0, 0)
        assert src.index(101985.0+400.0, 2826915.0) == (0, 1)
        assert src.index(101985.0+400.0, 2826915.0-700.0) == (2, 1)


def test_window():
    with rasterio.open('tests/data/RGB.byte.tif') as src:
        left, bottom, right, top = src.bounds
        dx, dy = src.res
        eps = 1.0e-8
        assert src.window(
            left+eps, bottom+eps, right-eps, top-eps) == ((0, src.height),
                                                          (0, src.width))
        assert src.index(left+400, top-400) == (1, 1)
        assert src.index(left+dx+eps, top-dy-eps) == (1, 1)
        assert src.window(left, top-400, left+400, top) == ((0, 2), (0, 2))
        assert src.window(left, top-2*dy-eps, left+2*dx-eps, top) == ((0, 2), (0, 2))


def test_window_bounds_roundtrip():
    with rasterio.open('tests/data/RGB.byte.tif') as src:
        assert ((100, 200), (100, 200)) == src.window(
            *src.window_bounds(((100, 200), (100, 200))))


def test_window_full_cover():

    def bound_covers(bounds1, bounds2):
        """Does bounds1 cover bounds2?
        """
        return (bounds1[0] <= bounds2[0] and bounds1[1] <= bounds2[1] and
                bounds1[2] >= bounds2[2] and bounds1[3] >= bounds2[3])

    with rasterio.open('tests/data/RGB.byte.tif') as src:
        bounds = list(src.window_bounds(((100, 200), (100, 200))))
        bounds[1] = bounds[1] - 10.0  # extend south
        bounds[2] = bounds[2] + 10.0  # extend east

        win = src.window(*bounds)
        bounds_calc = list(src.window_bounds(win))
        assert bound_covers(bounds_calc, bounds)


@pytest.fixture
def data():
    data = numpy.zeros((10, 10), dtype='uint8')
    data[slice(*DATA_WINDOW[0]), slice(*DATA_WINDOW[1])] = 1
    return data


def test_data_window_unmasked(data):
    window = get_data_window(data)
    assert window == ((0, data.shape[0]), (0, data.shape[1]))


def test_data_window_masked(data):
    data = numpy.ma.masked_array(data, data == 0)
    window = get_data_window(data)
    assert window == DATA_WINDOW


def test_data_window_nodata(data):
    window = get_data_window(data, nodata=0)
    assert window == DATA_WINDOW

    window = get_data_window(numpy.ones_like(data), nodata=0)
    assert window == ((0, data.shape[0]), (0, data.shape[1]))


def test_data_window_nodata_disjunct():
    data = numpy.zeros((3, 10, 10), dtype='uint8')
    data[0, :4, 1:4] = 1
    data[1, 2:5, 2:8] = 1
    data[2, 1:6, 1:6] = 1
    window = get_data_window(data, nodata=0)
    assert window == ((0, 6), (1, 8))


def test_data_window_empty_result():
    data = numpy.zeros((3, 10, 10), dtype='uint8')
    window = get_data_window(data, nodata=0)
    assert window == ((0, 0), (0, 0))


def test_data_window_masked_file():
    with rasterio.open('tests/data/RGB.byte.tif') as src:
        window = get_data_window(src.read(1, masked=True))
        assert window == ((3, 714), (13, 770))

        window = get_data_window(src.read(masked=True))
        assert window == ((3, 714), (13, 770))


def test_window_union():
    assert window_union([
        ((0, 6), (3, 6)),
        ((2, 4), (1, 5))
    ]) == ((0, 6), (1, 6))


def test_window_intersection():
    assert window_intersection([
        ((0, 6), (3, 6)),
        ((2, 4), (1, 5))
    ]) == ((2, 4), (3, 5))

    assert window_intersection([
        ((0, 6), (3, 6)),
        ((6, 10), (1, 5))
    ]) == ((6, 6), (3, 5))

    assert window_intersection([
        ((0, 6), (3, 6)),
        ((2, 4), (1, 5)),
        ((3, 6), (0, 6))
    ]) == ((3, 4), (3, 5))


def test_window_intersection_disjunct():
    with pytest.raises(ValueError):
        window_intersection([
            ((0, 6), (3, 6)),
            ((100, 200), (0, 12)),
            ((7, 12), (7, 12))
        ])


def test_windows_intersect():
    assert windows_intersect([
        ((0, 6), (3, 6)),
        ((2, 4), (1, 5))
    ]) == True

    assert windows_intersect([
        ((0, 6), (3, 6)),
        ((2, 4), (1, 5)),
        ((3, 6), (0, 6))
    ]) == True


def test_windows_intersect_disjunct():
    assert windows_intersect([
        ((0, 6), (3, 6)),
        ((10, 20), (0, 6))
    ]) == False

    assert windows_intersect([
        ((0, 6), (3, 6)),
        ((2, 4), (1, 5)),
        ((5, 6), (0, 6))
    ]) == False

    assert windows_intersect([
        ((0, 6), (3, 6)),
        ((2, 4), (1, 3)),
        ((3, 6), (4, 6))
    ]) == False