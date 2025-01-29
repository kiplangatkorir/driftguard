import os
class AlertManager:
    def __init__(self, threshold=0.5):
        """
        Initializes the AlertManager with a threshold for drift severity.
        :param threshold: The drift severity threshold that triggers an alert (default is 0.5).
        """
        self.threshold = threshold

    def send_alert(self, message):
        """
        Sends an alert message when drift exceeds the threshold.
        :param message: The alert message.
        """
        # In a real implementation, this could send an email or SMS.
        print(f"ALERT: {message}")

    def check_and_alert(self, drift_score, message="Drift detected!"):
        """
        Checks if drift severity exceeds the threshold and sends an alert.
        :param drift_score: The drift severity score.
        :param message: The alert message to be sent (optional).
        """
        if drift_score > self.threshold:
            self.send_alert(message)
