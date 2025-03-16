"""Tests for the alert management module."""
import pytest
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import Mock, patch

from driftguard.core.alerts import AlertManager, Alert, AlertSeverity
from driftguard.core.config import AlertConfig

@pytest.fixture
def alert_config():
    """Create test alert configuration"""
    return AlertConfig(
        email_settings={
            'smtp_server': 'smtp.test.com',
            'smtp_port': 587,
            'sender_email': 'test@example.com',
            'recipients': ['admin@example.com'],
            'username': 'test_user',
            'password': 'test_pass'
        },
        severity_thresholds={
            'critical': 0.3,
            'warning': 0.1,
            'info': 0.05
        },
        notification_cooldown=timedelta(hours=1)
    )

@pytest.fixture
def alert_manager(alert_config):
    """Create alert manager instance"""
    return AlertManager(alert_config)

def test_alert_creation():
    """Test alert object creation"""
    alert = Alert(
        message="Test alert",
        severity=AlertSeverity.WARNING,
        source="test_source",
        timestamp=datetime.now(),
        metadata={"test_key": "test_value"}
    )
    
    assert alert.message == "Test alert"
    assert alert.severity == AlertSeverity.WARNING
    assert alert.source == "test_source"
    assert isinstance(alert.timestamp, datetime)
    assert alert.metadata == {"test_key": "test_value"}

def test_alert_manager_initialization(alert_manager):
    """Test alert manager initialization"""
    assert alert_manager.config is not None
    assert len(alert_manager.alert_history) == 0

@patch('smtplib.SMTP')
def test_email_notification(mock_smtp, alert_manager):
    """Test email notification sending"""
    # Add alert that should trigger notification
    alert_manager.add_alert(
        message="Critical drift detected",
        severity=AlertSeverity.CRITICAL,
        source="drift_detector",
        metadata={"metric": "accuracy", "value": 0.6}
    )
    
    # Verify SMTP calls
    mock_smtp.assert_called_once()
    mock_smtp_instance = mock_smtp.return_value.__enter__.return_value
    assert mock_smtp_instance.starttls.called
    assert mock_smtp_instance.login.called
    assert mock_smtp_instance.send_message.called

def test_alert_filtering(alert_manager):
    """Test alert filtering functionality"""
    # Add multiple alerts
    alert_manager.add_alert(
        "Warning alert",
        AlertSeverity.WARNING,
        "test_source"
    )
    alert_manager.add_alert(
        "Critical alert",
        AlertSeverity.CRITICAL,
        "drift_detector"
    )
    alert_manager.add_alert(
        "Info alert",
        AlertSeverity.INFO,
        "performance_monitor"
    )
    
    # Filter by severity
    critical_alerts = alert_manager.get_alerts(
        severity=AlertSeverity.CRITICAL
    )
    assert len(critical_alerts) == 1
    assert critical_alerts[0].severity == AlertSeverity.CRITICAL
    
    # Filter by source
    drift_alerts = alert_manager.get_alerts(source="drift_detector")
    assert len(drift_alerts) == 1
    assert drift_alerts[0].source == "drift_detector"
    
    # Filter by time range
    now = datetime.now()
    recent_alerts = alert_manager.get_alerts(
        start_time=now - timedelta(minutes=5),
        end_time=now
    )
    assert len(recent_alerts) == 3

def test_alert_deduplication(alert_manager):
    """Test alert deduplication logic"""
    # Add similar alerts within cooldown period
    alert_manager.add_alert(
        "Duplicate alert",
        AlertSeverity.WARNING,
        "test_source"
    )
    
    # This should be deduplicated
    alert_manager.add_alert(
        "Duplicate alert",
        AlertSeverity.WARNING,
        "test_source"
    )
    
    alerts = alert_manager.get_alerts()
    assert len(alerts) == 1

def test_alert_aggregation(alert_manager):
    """Test alert aggregation functionality"""
    # Add multiple related alerts
    for i in range(5):
        alert_manager.add_alert(
            f"Performance degradation {i}",
            AlertSeverity.WARNING,
            "performance_monitor",
            metadata={"metric": "accuracy"}
        )
    
    # Get aggregated alerts
    aggregated = alert_manager.get_aggregated_alerts(
        group_by=["source", "severity"],
        time_window=timedelta(minutes=5)
    )
    
    assert len(aggregated) == 1
    assert aggregated[0]["count"] == 5
    assert aggregated[0]["source"] == "performance_monitor"
    assert aggregated[0]["severity"] == AlertSeverity.WARNING

def test_severity_escalation(alert_manager):
    """Test alert severity escalation"""
    # Add multiple warnings
    for _ in range(3):
        alert_manager.add_alert(
            "Warning alert",
            AlertSeverity.WARNING,
            "test_source"
        )
    
    # Check if severity was escalated
    alerts = alert_manager.get_alerts()
    assert any(alert.severity == AlertSeverity.CRITICAL for alert in alerts)

@patch('requests.post')
def test_webhook_notification(mock_post, alert_manager):
    """Test webhook notification sending"""
    # Configure webhook
    alert_manager.config.webhook_url = "https://test.webhook.com"
    
    # Add alert
    alert_manager.add_alert(
        "Test webhook",
        AlertSeverity.CRITICAL,
        "test_source"
    )
    
    # Verify webhook call
    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == "https://test.webhook.com"

def test_error_handling(alert_manager):
    """Test error handling in alert manager"""
    # Test invalid severity
    with pytest.raises(ValueError):
        alert_manager.add_alert(
            "Test alert",
            "INVALID_SEVERITY",
            "test_source"
        )
    
    # Test invalid time range
    with pytest.raises(ValueError):
        end_time = datetime.now()
        start_time = end_time + timedelta(hours=1)
        alert_manager.get_alerts(
            start_time=start_time,
            end_time=end_time
        )
