"""
Enhanced alert manager for DriftGuard v0.1.5.
Supports multiple notification channels, rate limiting, and advanced alert aggregation.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import logging
import json
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union, Any
from dotenv import load_dotenv
import jinja2
import requests
from collections import defaultdict
import asyncio
from aiohttp import ClientSession
import hashlib
from dataclasses import dataclass, asdict
import yaml

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AlertManager')

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    severity: str
    message: str
    metadata: Dict[str, Any]
    timestamp: datetime
    source: str
    tags: List[str]
    
    @property
    def hash(self) -> str:
        """Generate unique hash for deduplication"""
        content = f"{self.severity}:{self.message}:{str(sorted(self.metadata.items()))}"
        return hashlib.sha256(content.encode()).hexdigest()

class AlertChannel:
    """Base class for alert channels"""
    async def send(self, alert: Alert) -> bool:
        raise NotImplementedError

class EmailChannel(AlertChannel):
    """Email notification channel"""
    def __init__(self, config: Dict[str, Any]):
        self.smtp_server = config.get('smtp_host', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('smtp_user')
        self.password = config.get('smtp_password')
        self.use_ssl = config.get('use_ssl', False)
        self.use_tls = config.get('use_tls', True)
        self.template_dir = config.get('template_dir')
        self.retry_count = config.get('retry_count', 3)
        
        if self.template_dir:
            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.template_dir)
            )
    
    async def send(self, alert: Alert, recipients: List[str]) -> bool:
        for attempt in range(self.retry_count):
            try:
                msg = self._create_message(alert, recipients)
                
                smtp_class = smtplib.SMTP_SSL if self.use_ssl else smtplib.SMTP
                with smtp_class(self.smtp_server, self.smtp_port) as server:
                    if self.use_tls:
                        server.starttls()
                    server.login(self.username, self.password)
                    server.send_message(msg)
                return True
            
            except Exception as e:
                logger.error(f"Email send attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.retry_count - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def _create_message(self, alert: Alert, recipients: List[str]) -> MIMEMultipart:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"DriftGuard Alert: {alert.severity} - {alert.message[:50]}"
        msg['From'] = self.username
        msg['To'] = ", ".join(recipients)
        
        if self.template_dir:
            template = self.jinja_env.get_template('alert_email.html')
            html = template.render(alert=asdict(alert))
        else:
            html = self._default_template(alert)
        
        msg.attach(MIMEText(html, 'html'))
        return msg
    
    def _default_template(self, alert: Alert) -> str:
        return f"""
        <html>
            <body>
                <h2>DriftGuard Alert</h2>
                <p><strong>Severity:</strong> {alert.severity}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                <p><strong>Time:</strong> {alert.timestamp.isoformat()}</p>
                <p><strong>Source:</strong> {alert.source}</p>
                <p><strong>Tags:</strong> {', '.join(alert.tags)}</p>
                <h3>Metadata:</h3>
                <pre>{json.dumps(alert.metadata, indent=2)}</pre>
            </body>
        </html>
        """

class SlackChannel(AlertChannel):
    """Slack notification channel"""
    def __init__(self, config: Dict[str, Any]):
        self.webhook_url = config['webhook_url']
        self.channel = config.get('channel')
        self.username = config.get('username', 'DriftGuard')
        self.icon_emoji = config.get('icon_emoji', ':robot_face:')
    
    async def send(self, alert: Alert) -> bool:
        async with ClientSession() as session:
            payload = {
                "channel": self.channel,
                "username": self.username,
                "icon_emoji": self.icon_emoji,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"DriftGuard Alert: {alert.severity}"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Message:* {alert.message}"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Source:* {alert.source}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Time:* {alert.timestamp.isoformat()}"
                            }
                        ]
                    }
                ]
            }
            
            if alert.metadata:
                payload["blocks"].append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Metadata:*\n```{json.dumps(alert.metadata, indent=2)}```"
                    }
                })
            
            async with session.post(self.webhook_url, json=payload) as response:
                return response.status == 200

class AlertManager:
    """Enhanced alert manager with multiple channels and advanced features"""
    
    def __init__(
        self,
        config_path: str = "config/alerts.yaml",
        history_path: str = "data/alert_history.json"
    ):
        """Initialize alert manager with configuration"""
        self.config = self._load_config(config_path)
        self.history_path = history_path
        self.channels: Dict[str, AlertChannel] = {}
        self.alert_history: Dict[str, List[Dict]] = defaultdict(list)
        self.alert_counts: Dict[str, int] = defaultdict(int)
        self.last_alert_times: Dict[str, datetime] = {}
        
        self._setup_channels()
        self._load_history()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def _setup_channels(self) -> None:
        """Set up notification channels"""
        if 'email' in self.config and self.config['email'].get('enabled'):
            self.channels['email'] = EmailChannel(self.config['email'])
        
        if 'slack' in self.config and self.config['slack'].get('enabled'):
            self.channels['slack'] = SlackChannel(self.config['slack'])
    
    def _load_history(self) -> None:
        """Load alert history"""
        if os.path.exists(self.history_path):
            with open(self.history_path) as f:
                history = json.load(f)
                self.alert_history.update(history)
    
    def _save_history(self) -> None:
        """Save alert history"""
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(dict(self.alert_history), f, indent=2)
    
    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent based on rate limits and deduplication"""
        now = datetime.now()
        alert_hash = alert.hash
        
        # Check cooldown period
        if alert_hash in self.last_alert_times:
            cooldown = timedelta(hours=self.config.get('alert_cooldown_hours', 1))
            if now - self.last_alert_times[alert_hash] < cooldown:
                return False
        
        # Check daily limit
        today = now.date()
        if self.alert_counts[str(today)] >= self.config.get('max_alerts_per_day', 100):
            logger.warning("Daily alert limit reached")
            return False
        
        # Check for similar recent alerts
        recent_window = timedelta(minutes=self.config.get('dedup_window_minutes', 30))
        for recent_alert in self.alert_history[alert_hash][-5:]:
            if now - datetime.fromisoformat(recent_alert['timestamp']) < recent_window:
                return False
        
        return True
    
    async def send_alert(
        self,
        message: str,
        severity: str = "INFO",
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "DriftGuard",
        tags: Optional[List[str]] = None
    ) -> bool:
        """Send alert through configured channels"""
        alert = Alert(
            id=f"alert_{datetime.now().timestamp()}",
            severity=severity,
            message=message,
            metadata=metadata or {},
            timestamp=datetime.now(),
            source=source,
            tags=tags or []
        )
        
        if not self._should_send_alert(alert):
            logger.info(f"Alert suppressed due to rate limiting or deduplication: {message}")
            return False
        
        success = True
        tasks = []
        
        # Send to all enabled channels
        for channel in self.channels.values():
            if isinstance(channel, EmailChannel):
                tasks.append(
                    channel.send(
                        alert,
                        self.config['email'].get('recipients', [])
                    )
                )
            else:
                tasks.append(channel.send(alert))
        
        # Wait for all channels to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success = all(
            isinstance(result, bool) and result
            for result in results
        )
        
        if success:
            # Update tracking
            self.last_alert_times[alert.hash] = alert.timestamp
            self.alert_counts[str(alert.timestamp.date())] += 1
            self.alert_history[alert.hash].append(asdict(alert))
            self._save_history()
        
        return success
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        now = datetime.now()
        today = now.date()
        
        return {
            'total_alerts': sum(len(alerts) for alerts in self.alert_history.values()),
            'alerts_today': self.alert_counts[str(today)],
            'active_channels': list(self.channels.keys()),
            'recent_alerts': [
                alert
                for alerts in self.alert_history.values()
                for alert in alerts[-10:]
                if now - datetime.fromisoformat(alert['timestamp']) < timedelta(hours=24)
            ]
        }