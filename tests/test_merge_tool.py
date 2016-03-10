import sys
import os
import logging
import numpy
from pytest import fixture

import rasterio
from rasterio.tools.merge import merge as merge_tool
from rasterio.transform import Affine, array_bounds


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


# Fixture to create test datasets within temporary directory
@fixture(scope='function')
def test_data_dir_1(tmpdir):
    kwargs = {
        "crs": {'init': 'epsg:4326'},
        "transform": (-114, 0.2, 0, 46, 0, -0.2),
        "count": 1,
        "dtype": rasterio.uint8,
        "driver": "GTiff",
        "width": 10,
        "height": 10,
        "nodata": 1
    }

    with rasterio.drivers():

        with rasterio.open(str(tmpdir.join('a.tif')), 'w', **kwargs) as dst:
            data = numpy.ones((10, 10), dtype=rasterio.uint8)
            data[0:6, 0:6] = 255
            dst.write_band(1, data)

        with rasterio.open(str(tmpdir.join('b.tif')), 'w', **kwargs) as dst:
            data = numpy.ones((10, 10), dtype=rasterio.uint8)
            data[4:8, 4:8] = 254
            dst.write_band(1, data)

    return tmpdir


@fixture(scope='function')
def test_data_dir_2(tmpdir):
    kwargs = {
        "crs": {'init': 'epsg:4326'},
        "transform": (-114, 0.2, 0, 46, 0, -0.1),
        "count": 1,
        "dtype": rasterio.uint8,
        "driver": "GTiff",
        "width": 10,
        "height": 10
        # these files have undefined nodata.
    }

    with rasterio.drivers():

        with rasterio.open(str(tmpdir.join('a.tif')), 'w', **kwargs) as dst:
            data = numpy.zeros((10, 10), dtype=rasterio.uint8)
            data[0:6, 0:6] = 255
            dst.write_band(1, data)

        with rasterio.open(str(tmpdir.join('b.tif')), 'w', **kwargs) as dst:
            data = numpy.zeros((10, 10), dtype=rasterio.uint8)
            data[4:8, 4:8] = 254
            dst.write_band(1, data)

    return tmpdir


def test_merge_with_nodata(test_data_dir_1):
    files = [str(x) for x in test_data_dir_1.listdir()]
    files.sort()
    inputs = [rasterio.open(path) for path in files]
    result, transform = merge_tool(inputs)
    expected = numpy.ones((10, 10), dtype=rasterio.uint8)
    expected[0:6, 0:6] = 255
    expected[4:8, 4:8] = 254
    assert numpy.all(result == expected)


def test_merge_warn(test_data_dir_1, recwarn):
    files = [str(x) for x in test_data_dir_1.listdir()]
    files.sort()
    inputs = [rasterio.open(path) for path in files]
    result, transform = merge_tool(inputs, nodata=-1)
    w = recwarn.pop()
    assert "using the --nodata option for better results" in str(w.message)


def test_merge_without_nodata(test_data_dir_2):
    files = [str(x) for x in test_data_dir_2.listdir()]
    files.sort()
    inputs = [rasterio.open(path) for path in files]
    result, transform = merge_tool(inputs)
    expected = numpy.zeros((10, 10), dtype=rasterio.uint8)
    expected[0:6, 0:6] = 255
    expected[4:8, 4:8] = 254
    assert numpy.all(result == expected)


# Non-coincident datasets test fixture.
# Two overlapping GeoTIFFs, one to the NW and one to the SE.
@fixture(scope='function')
def test_data_dir_overlapping(tmpdir):
    kwargs = {
        "crs": {'init': 'epsg:4326'},
        "transform": (-114, 0.2, 0, 46, 0, -0.2),
        "count": 1,
        "dtype": rasterio.uint8,
        "driver": "GTiff",
        "width": 10,
        "height": 10,
        "nodata": 0
    }

    with rasterio.drivers():
        with rasterio.open(str(tmpdir.join('nw.tif')), 'w', **kwargs) as dst:
            data = numpy.ones((10, 10), dtype=rasterio.uint8)
            dst.write_band(1, data)

        kwargs['transform'] = (-113, 0.2, 0, 45, 0, -0.2)
        with rasterio.open(str(tmpdir.join('se.tif')), 'w', **kwargs) as dst:
            data = numpy.ones((10, 10), dtype=rasterio.uint8) * 2
            dst.write_band(1, data)

    return tmpdir


def test_merge_overlapping(test_data_dir_overlapping):
    files = [str(x) for x in test_data_dir_overlapping.listdir()]
    files.sort()
    inputs = [rasterio.open(path) for path in files]
    result, transform = merge_tool(inputs)
    _, h, w = result.shape
    assert h == w == 15
    assert array_bounds(w, h, transform) == (-114, 43, -111, 46)
    expected = numpy.zeros((15, 15), dtype=rasterio.uint8)
    expected[0:10, 0:10] = 1
    expected[5:, 5:] = 2
    assert numpy.all(result == expected)


# Fixture to create test datasets within temporary directory
@fixture(scope='function')
def test_data_dir_float(tmpdir):
    kwargs = {
        "crs": {'init': 'epsg:4326'},
        "transform": (-114, 0.2, 0, 46, 0, -0.2),
        "count": 1,
        "dtype": rasterio.float64,
        "driver": "GTiff",
        "width": 10,
        "height": 10,
        "nodata": 0
    }

    with rasterio.drivers():
        with rasterio.open(str(tmpdir.join('one.tif')), 'w', **kwargs) as dst:
            data = numpy.zeros((10, 10), dtype=rasterio.float64)
            data[0:6, 0:6] = 255
            dst.write_band(1, data)

        with rasterio.open(str(tmpdir.join('two.tif')), 'w', **kwargs) as dst:
            data = numpy.zeros((10, 10), dtype=rasterio.float64)
            data[4:8, 4:8] = 254
            dst.write_band(1, data)
    return tmpdir


def test_merge_float(test_data_dir_float):
    files = [str(x) for x in test_data_dir_float.listdir()]
    files.sort()
    inputs = [rasterio.open(path) for path in files]
    result, transform = merge_tool(inputs, nodata=-1.5)
    expected = numpy.ones((10, 10), dtype=rasterio.float64) * -1.5
    expected[0:6, 0:6] = 255
    expected[4:8, 4:8] = 254
    assert numpy.all(result == expected)


# Test below comes from issue #288. There was an off-by-one error in
# pasting image data into the canvas array.

@fixture(scope='function')
def tiffs(tmpdir):

    data = numpy.ones((1, 1, 1), 'uint8')

    kwargs = {'count': '1',
              'driver': 'GTiff',
              'dtype': 'uint8',
              'height': 1,
              'width': 1}

    kwargs['transform'] = Affine( 1, 0, 1,
                                  0,-1, 1)
    with rasterio.open(str(tmpdir.join('a-sw.tif')), 'w', **kwargs) as r:
        r.write(data * 40)

    kwargs['transform'] = Affine( 1, 0, 2,
                                  0,-1, 2)
    with rasterio.open(str(tmpdir.join('b-ct.tif')), 'w', **kwargs) as r:
        r.write(data * 60)

    kwargs['transform'] = Affine( 2, 0, 3,
                                  0,-2, 4)
    with rasterio.open(str(tmpdir.join('c-ne.tif')), 'w', **kwargs) as r:
        r.write(data * 90)

    kwargs['transform'] = Affine( 2, 0, 2,
                                  0,-2, 4)
    with rasterio.open(str(tmpdir.join('d-ne.tif')), 'w', **kwargs) as r:
        r.write(data * 120)

    return tmpdir


def test_merge_tiny(tiffs):
    files = [str(x) for x in tiffs.listdir()]
    files.sort()
    inputs = [rasterio.open(path) for path in files]
    data, transform = merge_tool(inputs)

    # Output should be
    #
    # [[  0 120 120  90]
    #  [  0 120 120  90]
    #  [  0  60   0   0]
    #  [ 40   0   0   0]]

    assert (data[0][0:2,1:3] == 120).all()
    assert (data[0][0:2,3] == 90).all()
    assert data[0][2][1] == 60
    assert data[0][3][0] == 40


def test_merge_tiny_res(tiffs):
    files = [str(x) for x in tiffs.listdir()]
    files.sort()
    inputs = [rasterio.open(path) for path in files]
    data, transform = merge_tool(inputs, res=(2.0, 2.0))

    # Output should be
    # [[[120  90]
    #   [  0   0]]]

    assert data[0, 0, 0] == 120
    assert data[0, 0, 1] == 90
    assert (data[0, 1, 0:1] == 0).all()


def test_merge_tiny_res_single_val(tiffs):
    files = [str(x) for x in tiffs.listdir()]
    files.sort()
    inputs = [rasterio.open(path) for path in files]
    data, transform = merge_tool(inputs, res=2.0)

    # Output should be
    # [[[120  90]
    #   [  0   0]]]

    assert data[0, 0, 0] == 120
    assert data[0, 0, 1] == 90
    import pdb; pdb.set_trace()
    assert (data[0, 1, 0:1] == 0).all()
