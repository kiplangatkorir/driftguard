import os
import datetime

class VersionManager:
    def __init__(self):
        """
        Initializes the version manager to track model versions.
        """
        self.version_history = []

    def create_new_version(self, model, version_name=None):
        """
        Creates a new version entry for the model.
        :param model: The model to be versioned.
        :param version_name: A custom version name (optional).
        :return: A dictionary containing the version information.
        """
        version_info = {
            "model": model,
            "version_name": version_name or f"v{len(self.version_history)+1}",
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.version_history.append(version_info)
        
        return version_info

    def list_versions(self):
        """
        Lists all tracked model versions.
        :return: A list of dictionaries containing version information.
        """
        return self.version_history
