import pytest
from unittest.mock import patch
from ui.app import predict, interface


@pytest.fixture
def app_interface():
    """Fixture to provide the Gradio interface instance."""
    return interface


@pytest.mark.parametrize("features, expected_result", [
    (("near_bay", -122.0, 37.0, 15, 1000, 300, 800, 300, 8.5), 12345.0),
    (("inland", -120.0, 35.0, 20, 2000, 500, 1000, 400, 6.0), 12345.0),
])
def test_predict(features, expected_result):
    """Test the predict function with various inputs."""
    with patch("ui.app.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"prediction": 12345.0}
        result = predict(*features)
        assert result == expected_result


def test_interface_launch():
    """Test that the interface launches correctly."""
    with patch("gradio.Interface.launch") as mock_launch:
        interface.launch(server_name="0.0.0.0", server_port=8080)
        mock_launch.assert_called_once_with(server_name="0.0.0.0", server_port=8080)
