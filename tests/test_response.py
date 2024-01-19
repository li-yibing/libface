import libface
import pytest


@pytest.mark.integration
@pytest.mark.response
def test_type(response):
    assert isinstance(response, libface.datastruct.ImageData) or isinstance(
        response, libface.datastruct.Response
    )


@pytest.mark.integration
@pytest.mark.response
def test_location_type(response):
    for face in response.faces:
        assert isinstance(face.loc, libface.datastruct.Location)


@pytest.mark.integration
@pytest.mark.response
def test_dims_type(response):
    for face in response.faces:
        assert isinstance(face.dims, libface.datastruct.Dimensions)


@pytest.mark.integration
@pytest.mark.response
def test_preds_type(response):
    for face in response.faces:
        assert isinstance(face.preds, dict)


@pytest.mark.integration
@pytest.mark.response
def test_preds_value_type(response):
    for face in response.faces:
        for pred in face.preds.values():
            assert isinstance(pred, libface.datastruct.Prediction)
