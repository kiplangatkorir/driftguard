import pytest
import os
import json
from alert_manager import AlertManager  # Adjust the import if the filename is different
from unittest.mock import patch, MagicMock

# Test files (mocked for testing)
TEST_ALERT_HISTORY_FILE = "alert_history.json"
TEST_RECIPIENT_CONFIG_FILE = "recipient_config.json"

@pytest.fixture
def alert_manager():
    """Fixture to create an instance of AlertManager with test files."""
    manager = AlertManager(
        threshold=0.5,
        alert_history_file=TEST_ALERT_HISTORY_FILE,
        recipient_config_file=TEST_RECIPIENT_CONFIG_FILE
    )
    yield manager
    # Cleanup test files after the test runs
    for file in [TEST_ALERT_HISTORY_FILE, TEST_RECIPIENT_CONFIG_FILE]:
        if os.path.exists(file):
            os.remove(file)

def test_email_validation(alert_manager):
    """Test valid and invalid email formats."""
    valid_emails = ["test@example.com", "user.name@domain.co", "my-email@sub.domain.org"]
    invalid_emails = ["invalid-email", "user@com", "user@.com", "@missing.com"]

    for email in valid_emails:
        assert alert_manager._validate_email(email) is True

    for email in invalid_emails:
        assert alert_manager._validate_email(email) is False

def test_set_recipient_email(alert_manager):
    """Test setting a recipient email address."""
    assert alert_manager.set_recipient_email("korirg543@gmail.com", "Test User") is True

    # Verify the email was stored correctly
    config = alert_manager.get_recipient_config()
    assert config["email"] == "korirg543@gmail.com"
    assert config["name"] == "Test User"

def test_set_recipient_email_invalid(alert_manager):
    """Test setting an invalid email address."""
    with pytest.raises(ValueError, match="Invalid email format"):
        alert_manager.set_recipient_email("invalid-email")

def test_load_recipient_config(alert_manager):
    """Test loading a recipient configuration from file."""
    test_data = {"email": "test@example.com", "name": "Test User"}
    with open(TEST_RECIPIENT_CONFIG_FILE, "w") as f:
        json.dump(test_data, f)

    alert_manager = AlertManager(
        alert_history_file=TEST_ALERT_HISTORY_FILE,
        recipient_config_file=TEST_RECIPIENT_CONFIG_FILE
    )
    
    config = alert_manager.get_recipient_config()
    assert config["email"] == "korirg543@gmail.com"
    assert config["name"] == "Test User"

def test_send_alert(alert_manager):
    """Test sending an alert (mocked email sending)."""
    alert_manager.set_recipient_email("recipient@example.com")

    with patch("smtplib.SMTP") as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        success = alert_manager.send_alert("Test alert", drift_score=0.7)
        assert success is True

        # Verify an email was sent
        mock_server.sendmail.assert_called_once()

def test_send_alert_without_recipient(alert_manager):
    """Test sending an alert without a recipient email set."""
    with pytest.raises(ValueError, match="No recipient email configured"):
        alert_manager.send_alert("Test alert", drift_score=0.8)

def test_check_and_alert(alert_manager):
    """Test checking drift score and triggering alert."""
    alert_manager.set_recipient_email("recipient@example.com")

    with patch.object(alert_manager, "send_alert", return_value=True) as mock_send_alert:
        result = alert_manager.check_and_alert(drift_score=0.6)
        assert result is True
        mock_send_alert.assert_called_once()

def test_check_and_alert_below_threshold(alert_manager):
    """Test when drift score is below threshold (no alert)."""
    alert_manager.set_recipient_email("recipient@example.com")

    with patch.object(alert_manager, "send_alert", return_value=True) as mock_send_alert:
        result = alert_manager.check_and_alert(drift_score=0.4)
        assert result is False
        mock_send_alert.assert_not_called()

def test_alert_history(alert_manager):
    """Test saving and loading alert history."""
    alert_manager.alert_history.append({
        "timestamp": "2024-01-30T12:00:00",
        "message": "Test alert",
        "drift_score": 0.75,
        "status": "success"
    })
    alert_manager._save_alert_history()

    new_manager = AlertManager(
        alert_history_file=TEST_ALERT_HISTORY_FILE,
        recipient_config_file=TEST_RECIPIENT_CONFIG_FILE
    )

    assert len(new_manager.alert_history) == 1
    assert new_manager.alert_history[0]["message"] == "Test alert"

def test_get_alert_statistics(alert_manager):
    """Test fetching alert statistics."""
    alert_manager.alert_history.extend([
        {"status": "success"},
        {"status": "failed"},
        {"status": "success"},
    ])

    stats = alert_manager.get_alert_statistics()
    assert stats["total_alerts"] == 3
    assert stats["successful_alerts"] == 2
    assert stats["failed_alerts"] == 1
