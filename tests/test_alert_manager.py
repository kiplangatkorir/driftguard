import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytest
import json
from unittest.mock import patch, mock_open
from driftmonitor.alert_manager import AlertManager

@pytest.fixture
def alert_manager():
    """Fixture to create an AlertManager instance with mocked files."""
    with patch("builtins.open", mock_open(read_data="{}")):
        return AlertManager()

@patch("smtplib.SMTP")
@patch("builtins.open", new_callable=mock_open)
def test_send_alert(mock_file, mock_smtp, alert_manager):
    """Test sending an alert email."""
    alert_manager.set_recipient_email("korirg543@gmail.com")

    # Ensure alert history starts empty
    alert_manager.alert_history = []

    # Mock SMTP server
    mock_server = mock_smtp.return_value
    mock_server.starttls.return_value = True
    mock_server.login.return_value = True
    mock_server.sendmail.return_value = True

    # Send an alert
    result = alert_manager.send_alert("Test alert message", drift_score=0.8)

    # Assertions
    assert result is True
    assert len(alert_manager.alert_history) == 1
    assert alert_manager.alert_history[0]["status"] == "success"
    assert alert_manager.alert_history[0]["recipient"] == "korirg543@gmail.com"

@patch("builtins.open", new_callable=mock_open)
def test_set_recipient_email(mock_file, alert_manager):
    """Test setting the recipient email."""
    result = alert_manager.set_recipient_email("korirg543@gmail.com", name="Korir")

    assert result is True
    assert alert_manager.recipient_config["email"] == "korirg543@gmail.com"
    assert alert_manager.recipient_config["name"] == "Korir"

@patch("builtins.open", new_callable=mock_open, read_data='{"email": "test@example.com"}')
def test_get_recipient_config(mock_file, alert_manager):
    """Test getting recipient configuration."""
    recipient = alert_manager.get_recipient_config()

    assert recipient["email"] == "test@example.com"

@patch("builtins.open", new_callable=mock_open)
def test_check_and_alert(mock_file, alert_manager):
    """Test checking drift score and triggering an alert."""
    alert_manager.set_recipient_email("korirg543@gmail.com")

    # Ensure alert history starts empty
    alert_manager.alert_history = []

    with patch.object(alert_manager, "send_alert", return_value=True) as mock_send_alert:
        result = alert_manager.check_and_alert(0.9)  # Above threshold

    assert result is True
    mock_send_alert.assert_called_once()

