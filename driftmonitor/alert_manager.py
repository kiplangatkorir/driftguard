import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
load_dotenv()  

class AlertManager:
    def __init__(self, threshold=0.5, smtp_server="smtp.gmail.com", smtp_port=587, sender_email=None, sender_password=None, recipient_email=None):
        """
        Initializes the AlertManager with a threshold for drift severity.
        :param threshold: The drift severity threshold that triggers an alert (default is 0.5).
        :param smtp_server: SMTP server address (default is Gmail).
        :param smtp_port: SMTP server port (default is 587 for TLS).
        :param sender_email: The sender's email address.
        :param sender_password: The sender's email password or app password.
        :param recipient_email: The recipient's email address.
        """
        self.threshold = threshold
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = os.getenv('SENDER_EMAIL')  
        self.sender_password = os.getenv('SENDER_PASSWORD')  
        self.recipient_email = recipient_email or os.getenv('RECIPIENT_EMAIL')  

        if not self.sender_email or not self.sender_password or not self.recipient_email:
            raise ValueError("Email configuration is incomplete. Please provide valid email credentials.")

    def send_alert(self, message):
        """
        Sends an alert message via email when drift exceeds the threshold.
        :param message: The alert message.
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = 'Drift Alert: Model Drift Detected'

            body = MIMEText(message, 'plain')
            msg.attach(body)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls() 
                server.login(self.sender_email, self.sender_password)
                text = msg.as_string()
                server.sendmail(self.sender_email, self.recipient_email, text)

            print(f"ALERT: {message} (Email sent to {self.recipient_email})")
        
        except Exception as e:
            print(f"Failed to send email alert: {e}")

    def check_and_alert(self, drift_score, message="Drift detected!"):
        """
        Checks if drift severity exceeds the threshold and sends an email alert.
        :param drift_score: The drift severity score.
        :param message: The alert message to be sent (optional).
        """
        if drift_score > self.threshold:
            self.send_alert(message)
