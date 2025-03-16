"""
Alert management module for DriftGuard.
"""
from typing import Dict, List, Optional, Union
import smtplib
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .interfaces import IAlertManager
from .config import AlertConfig

logger = logging.getLogger(__name__)

class Alert:
    """Alert data model"""
    def __init__(
        self,
        message: str,
        severity: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ):
        self.message = message
        self.severity = severity.upper()
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            'message': self.message,
            'severity': self.severity,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Alert':
        """Create alert from dictionary"""
        return cls(
            message=data['message'],
            severity=data['severity'],
            source=data['source'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )

class EmailNotifier:
    """Email notification handler"""
    def __init__(self, config: AlertConfig):
        """Initialize email notifier"""
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate email configuration"""
        required = ['smtp_host', 'smtp_port', 'smtp_user', 'smtp_password']
        missing = [f for f in required if not self.config.email.get(f)]
        if missing:
            raise ValueError(
                f"Missing required email configuration: {', '.join(missing)}"
            )
    
    def send_email(
        self,
        subject: str,
        body: str,
        recipients: Optional[List[str]] = None
    ) -> None:
        """Send email notification"""
        if not recipients:
            recipients = self.config.email.get('default_recipients', [])
        
        if not recipients:
            logger.warning("No recipients specified for email notification")
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.email['smtp_user']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to SMTP server
            with smtplib.SMTP(
                self.config.email['smtp_host'],
                self.config.email['smtp_port']
            ) as server:
                server.starttls()
                server.login(
                    self.config.email['smtp_user'],
                    self.config.email['smtp_password']
                )
                server.send_message(msg)
            
            logger.info(f"Email notification sent to {len(recipients)} recipients")
        
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            raise

class AlertManager(IAlertManager):
    """Manages alerts and notifications"""
    
    def __init__(self, config: Optional[AlertConfig] = None):
        """Initialize alert manager"""
        self.config = config or AlertConfig()
        self.alerts: List[Alert] = []
        self.email_notifier = None
        
        if self.config.email and self.config.email.get('enabled', False):
            self.email_notifier = EmailNotifier(self.config)
    
    def add_alert(
        self,
        message: str,
        severity: str,
        source: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add new alert"""
        # Validate severity
        severity = severity.upper()
        if severity not in self.config.severity_levels:
            raise ValueError(
                f"Invalid severity level. Must be one of: "
                f"{', '.join(self.config.severity_levels)}"
            )
        
        # Create and store alert
        alert = Alert(
            message=message,
            severity=severity,
            source=source,
            metadata=metadata
        )
        self.alerts.append(alert)
        
        # Log alert
        log_level = getattr(
            logging,
            severity,
            logging.INFO
        )
        logger.log(log_level, f"[{source}] {message}")
        
        # Send notifications if needed
        self._handle_notifications(alert)
    
    def get_alerts(
        self,
        severity: Optional[Union[str, List[str]]] = None,
        source: Optional[Union[str, List[str]]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Alert]:
        """Get filtered alerts"""
        filtered = self.alerts
        
        # Filter by severity
        if severity:
            if isinstance(severity, str):
                severity = [severity.upper()]
            else:
                severity = [s.upper() for s in severity]
            filtered = [a for a in filtered if a.severity in severity]
        
        # Filter by source
        if source:
            if isinstance(source, str):
                source = [source]
            filtered = [a for a in filtered if a.source in source]
        
        # Filter by time range
        if start_time:
            filtered = [a for a in filtered if a.timestamp >= start_time]
        if end_time:
            filtered = [a for a in filtered if a.timestamp <= end_time]
        
        return filtered
    
    def clear_alerts(
        self,
        severity: Optional[Union[str, List[str]]] = None,
        source: Optional[Union[str, List[str]]] = None
    ) -> None:
        """Clear alerts matching filters"""
        if severity:
            if isinstance(severity, str):
                severity = [severity.upper()]
            else:
                severity = [s.upper() for s in severity]
        
        if source:
            if isinstance(source, str):
                source = [source]
        
        self.alerts = [
            alert for alert in self.alerts
            if (severity and alert.severity not in severity) or
               (source and alert.source not in source)
        ]
    
    def _handle_notifications(self, alert: Alert) -> None:
        """Handle notifications for alert"""
        # Check if severity requires notification
        if alert.severity not in self.config.notify_on_severity:
            return
        
        # Send email notification if configured
        if self.email_notifier:
            try:
                subject = (
                    f"[DriftGuard {alert.severity}] "
                    f"Alert from {alert.source}"
                )
                
                body = (
                    f"Alert Details:\n"
                    f"Severity: {alert.severity}\n"
                    f"Source: {alert.source}\n"
                    f"Time: {alert.timestamp}\n"
                    f"Message: {alert.message}\n"
                )
                
                if alert.metadata:
                    body += "\nMetadata:\n"
                    for key, value in alert.metadata.items():
                        body += f"{key}: {value}\n"
                
                self.email_notifier.send_email(subject, body)
            
            except Exception as e:
                logger.error(
                    f"Failed to send email notification for alert: {e}"
                )
