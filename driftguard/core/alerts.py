"""
Alert management module for DriftGuard.
"""
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging
from pydantic import BaseModel, EmailStr

from .interfaces import IAlertManager
from .config import AlertConfig

logger = logging.getLogger(__name__)

class Alert(BaseModel):
    """Alert model"""
    id: str
    type: str
    message: str
    severity: str
    timestamp: datetime
    metadata: Dict = {}

class AlertManager(IAlertManager):
    """Manages alerts and notifications"""
    
    def __init__(self, config: AlertConfig):
        """Initialize alert manager"""
        self.config = config
        self.alerts: List[Alert] = []
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logging"""
        if self.config.log_file:
            handler = logging.FileHandler(self.config.log_file)
            handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
            logger.addHandler(handler)
            logger.setLevel(self.config.log_level)
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        count = len(self.alerts) + 1
        return f"alert_{timestamp}_{count}"
    
    def create_alert(
        self,
        message: str,
        alert_type: str,
        severity: str = "info",
        metadata: Optional[Dict] = None
    ) -> Alert:
        """Create new alert"""
        alert = Alert(
            id=self._generate_alert_id(),
            type=alert_type,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        logger.info(f"Created alert: {alert.id} - {alert.message}")
        
        # Send notifications if configured
        if self.config.email_enabled and severity in self.config.email_severity_levels:
            self._send_email_notification(alert)
        
        return alert
    
    def _send_email_notification(self, alert: Alert) -> None:
        """Send email notification"""
        if not self.config.email_recipients:
            logger.warning("No email recipients configured")
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_sender
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"DriftGuard Alert: {alert.type} - {alert.severity}"
            
            # Create email body
            body = f"""
            Alert Details:
            --------------
            Type: {alert.type}
            Severity: {alert.severity}
            Time: {alert.timestamp}
            Message: {alert.message}
            
            Metadata:
            ---------
            """
            
            for key, value in alert.metadata.items():
                body += f"{key}: {value}\n"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.smtp_use_tls:
                    server.starttls()
                if self.config.smtp_username and self.config.smtp_password:
                    server.login(
                        self.config.smtp_username,
                        self.config.smtp_password
                    )
                server.send_message(msg)
            
            logger.info(f"Sent email notification for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
    
    def get_alerts(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Alert]:
        """Get filtered alerts"""
        filtered_alerts = self.alerts
        
        if alert_type:
            filtered_alerts = [
                a for a in filtered_alerts
                if a.type == alert_type
            ]
        
        if severity:
            filtered_alerts = [
                a for a in filtered_alerts
                if a.severity == severity
            ]
        
        if start_time:
            filtered_alerts = [
                a for a in filtered_alerts
                if a.timestamp >= start_time
            ]
        
        if end_time:
            filtered_alerts = [
                a for a in filtered_alerts
                if a.timestamp <= end_time
            ]
        
        return filtered_alerts
    
    def clear_alerts(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        older_than: Optional[datetime] = None
    ) -> int:
        """Clear alerts matching criteria"""
        initial_count = len(self.alerts)
        
        if alert_type or severity or older_than:
            self.alerts = [
                alert for alert in self.alerts
                if (alert_type and alert.type != alert_type) or
                (severity and alert.severity != severity) or
                (older_than and alert.timestamp > older_than)
            ]
        else:
            self.alerts = []
        
        cleared_count = initial_count - len(self.alerts)
        logger.info(f"Cleared {cleared_count} alerts")
        
        return cleared_count
