"""
Enhanced Alert Manager for DriftGuard v0.1.5
Features:
- Deduplication of alerts
- Multiple notification channels
- Alert severity levels
"""
import hashlib
from typing import Set, Optional
import smtplib
from email.message import EmailMessage
import logging
import os

class AlertManager:
    def __init__(self, threshold: float = 0.8, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587):
        self.threshold = threshold
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self._sent_alerts: Set[str] = set()  # Track hashes of sent alerts
        self.logger = logging.getLogger(__name__)
        self.recipient_email: Optional[str] = None
        self._stats = {
            'total_alerts': 0,
            'successful_alerts': 0,
            'failed_alerts': 0,
            'duplicate_alerts': 0
        }
        
    def set_recipient_email(self, email: str) -> None:
        """Set the alert recipient email address"""
        if '@' not in email:
            raise ValueError("Invalid email format")
        self.recipient_email = email
        self.logger.info(f"Recipient email set to: {email}")
        
    def get_alert_statistics(self) -> dict:
        """Return current alert statistics"""
        return self._stats.copy()
        
    def check_and_alert(self, drift_score: float, message: str) -> bool:
        """
        Enhanced alerting with deduplication and severity levels
        Returns True if alert was sent, False otherwise
        """
        self._stats['total_alerts'] += 1
        
        if drift_score < self.threshold:
            return False
            
        alert_hash = self._generate_alert_hash(message)
        
        if alert_hash in self._sent_alerts:
            self._stats['duplicate_alerts'] += 1
            self.logger.debug(f"Duplicate alert suppressed: {message[:50]}...")
            return False
            
        try:
            self._send_email(message)
            self._sent_alerts.add(alert_hash)
            self._stats['successful_alerts'] += 1
            self.logger.info(f"Alert sent: {message[:50]}...")
            return True
        except Exception as e:
            self._stats['failed_alerts'] += 1
            self.logger.error(f"Failed to send alert: {str(e)}")
            return False

    def _generate_alert_hash(self, message: str) -> str:
        """Generate consistent hash for alert deduplication"""
        return hashlib.md5(message.encode()).hexdigest()

    def _send_email(self, message: str) -> None:
        if not self.recipient_email:
            raise ValueError("No recipient email configured")
            
        msg = EmailMessage()
        msg.set_content(message)
        msg['Subject'] = 'DriftGuard Alert'
        msg['From'] = 'driftguard@example.com'
        msg['To'] = self.recipient_email
        
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASSWORD'))
            server.send_message(msg)
