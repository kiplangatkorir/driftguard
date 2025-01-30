import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import Optional, Dict, List
import json
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AlertManager')

class AlertManager:
    def __init__(
        self,
        threshold: float = 0.5,
        alert_history_file: str = "alert_history.json",
        recipient_config_file: str = "recipient_config.json"
    ):
        """
        Initializes the AlertManager with system configuration.
        
        Args:
            threshold: The drift severity threshold that triggers an alert.
            alert_history_file: File to store alert history.
            recipient_config_file: File to store recipient configuration.
        """
        load_dotenv()
        
        self.smtp_server = os.getenv('SMTP_SERVER', "smtp.gmail.com")
        self.smtp_port = int(os.getenv('SMTP_PORT', "587"))
        self.sender_email = os.getenv('SENDER_EMAIL')
        self.sender_password = os.getenv('SENDER_PASSWORD')
        
        self.threshold = threshold
        self.alert_history_file = alert_history_file
        self.recipient_config_file = recipient_config_file
        self.alert_count = 0
        self.last_alert_time = None
        
        self._validate_system_config()
        
        self.alert_history = self._load_alert_history()
        self.recipient_config = self._load_recipient_config()

    def _validate_system_config(self) -> None:
        """Validates the system email configuration settings."""
        missing_configs = []
        
        if not self.sender_email:
            missing_configs.append("SENDER_EMAIL")
        if not self.sender_password:
            missing_configs.append("SENDER_PASSWORD")
            
        if missing_configs:
            raise ValueError(
                f"Missing system configuration(s): {', '.join(missing_configs)}. "
                "Please ensure these are set in your .env file."
            )

    def _validate_email(self, email: str) -> bool:
        """
        Validates email format.
        
        Args:
            email: Email address to validate.
            
        Returns:
            bool: True if email format is valid, False otherwise.
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def set_recipient_email(self, email: str, name: Optional[str] = None) -> bool:
        """
        Sets or updates the recipient's email configuration.
        
        Args:
            email: Recipient's email address.
            name: Recipient's name (optional).
            
        Returns:
            bool: True if configuration was updated successfully.
            
        Raises:
            ValueError: If email format is invalid.
        """
        if not self._validate_email(email):
            raise ValueError("Invalid email format")
            
        self.recipient_config = {
            "email": email,
            "name": name,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.recipient_config_file, 'w') as f:
                json.dump(self.recipient_config, f, indent=2)
            logger.info(f"Recipient configuration updated: {email}")
            return True
        except Exception as e:
            logger.error(f"Failed to save recipient configuration: {e}")
            return False

    def get_recipient_config(self) -> Dict:
        """
        Returns current recipient configuration.
        
        Returns:
            Dict containing recipient email and name if set.
        """
        return self.recipient_config

    def _load_recipient_config(self) -> Dict:
        """
        Loads the recipient configuration from file.
        
        Returns:
            Dict containing recipient configuration.
        """
        try:
            if os.path.exists(self.recipient_config_file):
                with open(self.recipient_config_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Failed to load recipient configuration: {e}")
            return {}

    def _load_alert_history(self) -> List[Dict]:
        """
        Loads the alert history from file.
        
        Returns:
            List of alert history entries.
        """
        try:
            if os.path.exists(self.alert_history_file):
                with open(self.alert_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.warning(f"Failed to load alert history: {e}")
            return []

    def _save_alert_history(self) -> None:
        """Saves the alert history to file."""
        try:
            with open(self.alert_history_file, 'w') as f:
                json.dump(self.alert_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alert history: {e}")

    def _generate_html_email(
        self,
        message: str,
        drift_score: Optional[float],
        current_time: datetime
    ) -> str:
        """
        Generates HTML-formatted email content.
        
        Args:
            message: Alert message
            drift_score: Drift score value
            current_time: Current timestamp
            
        Returns:
            str: HTML-formatted email content
        """
        recipient_name = self.recipient_config.get('name', '')
        greeting = f"Hello {recipient_name}," if recipient_name else "Hello,"
        
        return f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .alert-details {{
                    background-color: #fff3cd;
                    padding: 15px;
                    border-left: 5px solid #ffeeba;
                    margin-bottom: 20px;
                }}
                .message-box {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .stats {{
                    background-color: #e9ecef;
                    padding: 15px;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 10px;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #dee2e6;
                }}
                .severity-high {{
                    color: #dc3545;
                    font-weight: bold;
                }}
                .severity-medium {{
                    color: #ffc107;
                    font-weight: bold;
                }}
                .severity-low {{
                    color: #28a745;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2 style="margin: 0;">Model Drift Alert</h2>
                <p>{greeting}</p>
            </div>
            
            <div class="alert-details">
                <h3>Alert Details</h3>
                <table>
                    <tr>
                        <th>Time</th>
                        <td>{current_time.strftime('%Y-%m-%d %H:%M:%S')}</td>
                    </tr>
                    <tr>
                        <th>Drift Score</th>
                        <td class="{self._get_severity_class(drift_score)}">
                            {drift_score:.3f if drift_score is not None else 'N/A'}
                        </td>
                    </tr>
                    <tr>
                        <th>Threshold</th>
                        <td>{self.threshold}</td>
                    </tr>
                </table>
            </div>
            
            <div class="message-box">
                <h3>Alert Message</h3>
                <p>{message}</p>
            </div>
            
            <div class="stats">
                <h3>Alert Statistics</h3>
                <table>
                    <tr>
                        <th>Total Alerts Today</th>
                        <td>{self.alert_count + 1}</td>
                    </tr>
                    <tr>
                        <th>Last Alert</th>
                        <td>{self.last_alert_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_alert_time else 'None'}</td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """

    def _generate_plain_text_email(
        self,
        message: str,
        drift_score: Optional[float],
        current_time: datetime
    ) -> str:
        """
        Generates plain text email content as fallback.
        
        Args:
            message: Alert message
            drift_score: Drift score value
            current_time: Current timestamp
            
        Returns:
            str: Plain text email content
        """
        recipient_name = self.recipient_config.get('name', '')
        greeting = f"Hello {recipient_name}," if recipient_name else "Hello,"
        
        return f"""
{greeting}

Drift Alert Details:
-------------------
Time: {current_time}
Drift Score: {drift_score if drift_score is not None else 'N/A'}
Threshold: {self.threshold}

Message:
{message}

Alert Statistics:
----------------
Total Alerts Today: {self.alert_count + 1}
Last Alert: {self.last_alert_time if self.last_alert_time else 'None'}
"""

    def _get_severity_class(self, drift_score: Optional[float]) -> str:
        """
        Determines the severity class based on drift score.
        
        Args:
            drift_score: The drift score value
            
        Returns:
            str: CSS class name for severity styling
        """
        if drift_score is None:
            return ""
        if drift_score > self.threshold * 1.5:
            return "severity-high"
        if drift_score > self.threshold * 1.2:
            return "severity-medium"
        return "severity-low"

    def send_alert(self, message: str, drift_score: Optional[float] = None) -> bool:
        """
        Sends an HTML-formatted alert message via email with enhanced error handling and logging.
        
        Args:
            message: The alert message.
            drift_score: The drift score that triggered the alert (optional).
            
        Returns:
            bool: True if alert was sent successfully, False otherwise.
            
        Raises:
            ValueError: If no recipient email is configured.
        """
        if not self.recipient_config.get('email'):
            raise ValueError(
                "No recipient email configured. "
                "Please call set_recipient_email() first."
            )
            
        recipient_email = self.recipient_config['email']
        current_time = datetime.now()
        
        alert_details = {
            "timestamp": current_time.isoformat(),
            "message": message,
            "drift_score": drift_score,
            "recipient": recipient_email
        }
        
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f'Drift Alert: Model Drift Detected (Score: {drift_score:.3f})'

            # Plain text version
            text_content = self._generate_plain_text_email(message, drift_score, current_time)
            plain_part = MIMEText(text_content, 'plain')
            
            # HTML version
            html_content = self._generate_html_email(message, drift_score, current_time)
            html_part = MIMEText(html_content, 'html')
            
            # Add both parts to the message
            msg.attach(plain_part)
            msg.attach(html_part)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient_email, msg.as_string())

            self.alert_count += 1
            self.last_alert_time = current_time
            alert_details["status"] = "success"
            
            logger.info(f"Alert sent successfully to {recipient_email}")
            
            self.alert_history.append(alert_details)
            self._save_alert_history()
            
            return True

        except Exception as e:
            alert_details["status"] = "failed"
            alert_details["error"] = str(e)
            self.alert_history.append(alert_details)
            self._save_alert_history()
            
            logger.error(f"Failed to send alert: {e}")
            return False

    def check_and_alert(
        self,
        drift_score: float,
        message: Optional[str] = None,
        custom_threshold: Optional[float] = None
    ) -> bool:
        """
        Checks if drift severity exceeds the threshold and sends an alert.
        
        Args:
            drift_score: The drift severity score.
            message: Custom alert message (optional).
            custom_threshold: Override default threshold for this check (optional).
            
        Returns:
            bool: True if alert was sent, False otherwise.
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        
        if drift_score > threshold:
            default_message = (
                f"Drift detected! Score: {drift_score:.3f} exceeds "
                f"threshold: {threshold:.3f}"
            )
            return self.send_alert(
                message or default_message,
                drift_score=drift_score
            )
        return False

    def get_alert_statistics(self) -> Dict:
        """
        Returns statistics about sent alerts.
        
        Returns:
            Dict containing alert statistics.
        """
        return {
            "total_alerts": len(self.alert_history),
            "successful_alerts": sum(
                1 for alert in self.alert_history
                if alert["status"] == "success"
            ),
            "failed_alerts": sum(
                1 for alert in self.alert_history
                if alert["status"] == "failed"
            ),
            "last_alert_time": self.last_alert_time,
            "alert_count_today": self.alert_count
        }