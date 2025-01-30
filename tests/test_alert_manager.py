import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytest
import json
import os
from unittest.mock import patch, MagicMock
from datetime import datetime
from driftmonitor.alert_manager import AlertManager  

@pytest.fixture
def alert_manager():
    """Fixture to create an AlertManager instance with temporary files."""
    return AlertManager(
        threshold=0.5,
        alert_history_file="test_alert_history.json",
        recipient_config_file="test_recipient_config.json"
    )

def test_email_validation(alert_manager):
    """Test valid and invalid email formats."""
    assert alert_manager._validate_email("test@example.com") is True
    assert alert_manager._validate_email("invalid-email") is False
    assert alert_manager._validate_email("user@domain") is False
    assert alert_manager._validate_email("user@domain..com") is False

def test_set_recipient_email(alert_manager):
    """Test setting a valid recipient email."""
    assert alert_manager.set_recipient_email("valid@example.com", "Test User") is True
    config = alert_manager.get_recipient_config()
    assert config["email"] == "valid@example.com"
    assert config["name"] == "Test User"

def test_set_recipient_email_invalid(alert_manager):
    """Test handling of invalid email formats."""
    with pytest.raises(ValueError, match="Invalid email format"):
        alert_manager.set_recipient_email("invalid-email")

@patch("smtplib.SMTP")
def test_send_alert(mock_smtp, alert_manager):
    """Test sending an alert successfully (mocked SMTP)."""
    alert_manager.set_recipient_email("korirg543@gmail.com")
    
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server
    
    assert alert_manager.send_alert("Test Alert", drift_score=0.8) is True
    
    alert_history = alert_manager.get_alert_statistics()
    assert alert_history["total_alerts"] == 1
    assert alert_history["successful_alerts"] == 1
    assert alert_history["failed_alerts"] == 0

@patch("smtplib.SMTP", side_effect=Exception("SMTP connection error"))
def test_send_alert_failure(mock_smtp, alert_manager):
    """Test failure scenario when sending an alert."""
    alert_manager.set_recipient_email("korirg543@gmail.com")
    assert alert_manager.send_alert("Test Alert", drift_score=0.8) is False
    
    alert_history = alert_manager.get_alert_statistics()
    assert alert_history["total_alerts"] == 1
    assert alert_history["failed_alerts"] == 1

def test_check_and_alert(alert_manager):
    """Test that alerts trigger only when drift exceeds the threshold."""
    alert_manager.set_recipient_email("korirg543@gmail.com.com")
    
    with patch.object(alert_manager, "send_alert", return_value=True) as mock_send:
        assert alert_manager.check_and_alert(0.6) is True  # Should trigger alert
        mock_send.assert_called_once()
    
    with patch.object(alert_manager, "send_alert", return_value=False) as mock_send:
        assert alert_manager.check_and_alert(0.4) is False  # Should not trigger alert
        mock_send.assert_not_called()

def test_alert_statistics(alert_manager):
    """Test alert statistics tracking."""
    alert_manager.set_recipient_email("korirg543@gmail.com")
    alert_manager.send_alert("Test Alert", drift_score=0.8)

    stats = alert_manager.get_alert_statistics()
    assert stats["total_alerts"] == 1
    assert stats["successful_alerts"] == 1
    assert stats["failed_alerts"] == 0
    assert stats["alert_count_today"] == 1
    assert stats["last_alert_time"] is not None

@pytest.fixture(scope="function", autouse=True)
def cleanup_test_files():
    """Cleanup test files after each test."""
    yield
    for file in ["test_alert_history.json", "test_recipient_config.json"]:
        if os.path.exists(file):
            os.remove(file)
