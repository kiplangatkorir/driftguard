import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytest
import json
import os
from unittest.mock import patch, mock_open
from datetime import datetime
from driftmonitor.alert_manager import AlertManager

@pytest.fixture
def alert_manager():
    """Fixture to create an instance of AlertManager for testing."""
    return AlertManager()

@patch("builtins.open", new_callable=mock_open, read_data='{}')
@patch("os.path.exists", return_value=True)
def test_set_recipient_email(mock_exists, mock_file, alert_manager):
    """Test setting recipient email."""
    email = "korirg543@gmail.com"
    
    # Set recipient email
    assert alert_manager.set_recipient_email(email) is True

    # Check if the recipient was updated
    assert alert_manager.recipient_config["email"] == email

@patch("smtplib.SMTP")
@patch("builtins.open", new_callable=mock_open)
@patch.object(AlertManager, "_load_alert_history", return_value=[])  # Mock history to start empty
def test_send_alert(mock_history, mock_file, mock_smtp, alert_manager):
    """Test sending an alert email."""
    
    # Set recipient email
    alert_manager.set_recipient_email("korirg543@gmail.com")

    # Mock SMTP server behavior
    mock_server = mock_smtp.return_value
    mock_server.starttls.return_value = True
    mock_server.login.return_value = True
    mock_server.sendmail.return_value = True

    # Send an alert
    result = alert_manager.send_alert("Test alert message", drift_score=0.8)
    
    # Check that the alert was successfully sent
    assert result is True
    
    # Check if the alert history is updated
    print("Alert history:", alert_manager.alert_history)  # Debugging line to check what's in history
    assert len(alert_manager.alert_history) == 1  # Expect only 1 new alert
    
    # Ensure that the history is being saved correctly
    mock_file.assert_called_with('alert_history.json', 'w')  # Ensure file is being written


@patch("builtins.open", new_callable=mock_open)
def test_check_and_alert(mock_file, alert_manager):
    """Test drift check triggering an alert."""
    alert_manager.set_recipient_email("korirg543@gmail.com")
    
    # Test below threshold (no alert should be sent)
    assert alert_manager.check_and_alert(0.4) is False

    # Test above threshold (alert should be sent)
    with patch.object(alert_manager, "send_alert", return_value=True) as mock_alert:
        assert alert_manager.check_and_alert(0.6) is True
        mock_alert.assert_called_once()

@patch("builtins.open", new_callable=mock_open)
def test_alert_statistics(mock_file, alert_manager):
    """Test alert statistics retrieval."""
    alert_manager.set_recipient_email("korirg543@gmail.com")

    # Simulate sending two alerts
    alert_manager.alert_history = [
        {"status": "success", "timestamp": datetime.now().isoformat()},
        {"status": "failed", "timestamp": datetime.now().isoformat()}
    ]

    stats = alert_manager.get_alert_statistics()
    
    assert stats["total_alerts"] == 2
    assert stats["successful_alerts"] == 1
    assert stats["failed_alerts"] == 1




