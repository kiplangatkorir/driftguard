"""
Alert management system for DriftGuard.
Handles notifications across multiple channels with proper error handling and rate limiting.
"""
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio
from aiohttp import ClientSession
import jinja2
from .interfaces import IAlertManager
from .config import AlertConfig, AlertLevel

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert message container"""
    type: str
    message: str
    severity: AlertLevel
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "type": self.type,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }

class AlertChannel(ABC):
    """Base class for alert channels"""
    
    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send alert through channel"""
        pass
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate channel configuration"""
        pass

class EmailChannel(AlertChannel):
    """Email notification channel"""
    
    def __init__(
        self,
        recipients: List[str],
        smtp_host: str,
        smtp_port: int,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        use_tls: bool = True
    ):
        self.recipients = recipients
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.use_tls = use_tls
        
        # Load email templates
        self.template_env = jinja2.Environment(
            loader=jinja2.PackageLoader('driftguard', 'templates')
        )
    
    def validate_config(self) -> List[str]:
        """Validate email configuration"""
        issues = []
        
        if not self.recipients:
            issues.append("No email recipients configured")
        
        if not self.smtp_host:
            issues.append("SMTP host not configured")
            
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
        except Exception as e:
            issues.append(f"Failed to connect to SMTP server: {str(e)}")
            
        return issues
    
    async def send(self, alert: Alert) -> bool:
        """Send email alert"""
        try:
            template = self.template_env.get_template('alert_email.html')
            html_content = template.render(
                alert_type=alert.type,
                message=alert.message,
                severity=alert.severity.value,
                timestamp=alert.timestamp,
                metadata=alert.metadata
            )
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"DriftGuard Alert: {alert.type}"
            msg['From'] = self.smtp_user
            msg['To'] = ", ".join(self.recipients)
            
            msg.attach(MIMEText(html_content, 'html'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
                
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False

class SlackChannel(AlertChannel):
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def validate_config(self) -> List[str]:
        """Validate Slack configuration"""
        issues = []
        
        if not self.webhook_url.startswith(('http://', 'https://')):
            issues.append("Invalid Slack webhook URL")
            
        return issues
    
    async def send(self, alert: Alert) -> bool:
        """Send Slack alert"""
        try:
            color = {
                AlertLevel.INFO: "#36a64f",
                AlertLevel.WARNING: "#ffd700",
                AlertLevel.CRITICAL: "#ff0000"
            }.get(alert.severity, "#808080")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"DriftGuard Alert: {alert.type}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value,
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.isoformat(),
                            "short": True
                        }
                    ],
                    "footer": "DriftGuard Monitoring System"
                }]
            }
            
            if alert.metadata:
                payload["attachments"][0]["fields"].extend([
                    {"title": k, "value": str(v), "short": True}
                    for k, v in alert.metadata.items()
                ])
            
            async with ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=5
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False

class AlertManager(IAlertManager):
    """Manages alert distribution across multiple channels"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.channels: List[AlertChannel] = []
        self._alert_history: List[Alert] = []
        self._last_alert_times: Dict[str, datetime] = {}
        
        self._setup_channels()
    
    def _setup_channels(self) -> None:
        """Setup configured alert channels"""
        if self.config.email_recipients:
            self.channels.append(
                EmailChannel(
                    recipients=self.config.email_recipients,
                    smtp_host="smtp.gmail.com",  # Should come from config
                    smtp_port=587,
                    smtp_user=os.getenv("SMTP_USER"),
                    smtp_password=os.getenv("SMTP_PASSWORD")
                )
            )
            
        if self.config.slack_webhook:
            self.channels.append(
                SlackChannel(webhook_url=self.config.slack_webhook)
            )
    
    def validate_channels(self) -> List[str]:
        """Validate all configured channels"""
        issues = []
        
        for channel in self.channels:
            channel_issues = channel.validate_config()
            if channel_issues:
                issues.extend([
                    f"{channel.__class__.__name__}: {issue}"
                    for issue in channel_issues
                ])
                
        return issues
    
    def should_alert(self, alert_type: str) -> bool:
        """Check if alert should be sent based on cooldown"""
        if alert_type not in self._last_alert_times:
            return True
            
        last_alert = self._last_alert_times[alert_type]
        cooldown = timedelta(seconds=self.config.cooldown_period)
        
        return datetime.now() - last_alert > cooldown
    
    async def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: Union[str, AlertLevel],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send alert through all configured channels.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity level
            metadata: Additional alert metadata
            
        Returns:
            True if alert was sent successfully through any channel
        """
        if isinstance(severity, str):
            severity = AlertLevel(severity.lower())
            
        if severity.value < self.config.min_severity.value:
            logger.debug(f"Alert severity {severity} below minimum threshold")
            return False
            
        if not self.should_alert(alert_type):
            logger.debug(f"Alert {alert_type} in cooldown period")
            return False
        
        alert = Alert(
            type=alert_type,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        # Store alert in history
        self._alert_history.append(alert)
        self._last_alert_times[alert_type] = alert.timestamp
        
        # Trim history to last 1000 alerts
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-1000:]
        
        # Send through all channels
        results = await asyncio.gather(*[
            channel.send(alert)
            for channel in self.channels
        ])
        
        return any(results)
    
    def get_alert_history(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[AlertLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get filtered alert history"""
        filtered_alerts = self._alert_history
        
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
            
        return [alert.to_dict() for alert in filtered_alerts]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        stats = {
            "total_alerts": len(self._alert_history),
            "alerts_by_type": {},
            "alerts_by_severity": {
                level.value: 0 for level in AlertLevel
            }
        }
        
        for alert in self._alert_history:
            # Count by type
            if alert.type not in stats["alerts_by_type"]:
                stats["alerts_by_type"][alert.type] = 0
            stats["alerts_by_type"][alert.type] += 1
            
            # Count by severity
            stats["alerts_by_severity"][alert.severity.value] += 1
        
        return stats
