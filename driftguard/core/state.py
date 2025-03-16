"""
State management for DriftGuard.
Handles persistence of monitoring state and metrics.
"""
from typing import Any, Dict, List, Optional, Union
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from .interfaces import IStateManager, StorageConfig

logger = logging.getLogger(__name__)

class StateManager(IStateManager):
    """Manages monitoring state and metrics history"""
    
    def __init__(self, config: StorageConfig):
        """Initialize state manager"""
        self.config = config
        self.storage_path = Path(config.path)
        self._initialize_storage()
        
        # In-memory cache
        self._state: Dict[str, Any] = {}
        self._metrics_history: List[Dict[str, Any]] = []
        self._warnings: List[Dict[str, Any]] = []
        
        # Load existing state if available
        self._load_persisted_state()
    
    def _initialize_storage(self) -> None:
        """Initialize storage directory structure"""
        try:
            # Create main storage directory
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.storage_path / "metrics").mkdir(exist_ok=True)
            (self.storage_path / "state").mkdir(exist_ok=True)
            (self.storage_path / "warnings").mkdir(exist_ok=True)
            
            logger.info(f"Initialized storage at {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize storage: {str(e)}")
            raise
    
    def _load_persisted_state(self) -> None:
        """Load persisted state from storage"""
        try:
            # Load state
            state_file = self.storage_path / "state" / "current_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    self._state = json.load(f)
            
            # Load metrics history
            metrics_file = self.storage_path / "metrics" / "history.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self._metrics_history = json.load(f)
            
            # Load warnings
            warnings_file = self.storage_path / "warnings" / "history.json"
            if warnings_file.exists():
                with open(warnings_file, 'r') as f:
                    self._warnings = json.load(f)
            
            logger.info("Loaded persisted state")
            
        except Exception as e:
            logger.error(f"Failed to load persisted state: {str(e)}")
            self._state = {}
            self._metrics_history = []
            self._warnings = []
    
    def _persist_state(self) -> None:
        """Persist current state to storage"""
        try:
            # Save state
            with open(self.storage_path / "state" / "current_state.json", 'w') as f:
                json.dump(self._state, f, indent=2, default=str)
            
            # Save metrics history
            with open(self.storage_path / "metrics" / "history.json", 'w') as f:
                json.dump(self._metrics_history, f, indent=2, default=str)
            
            # Save warnings
            with open(self.storage_path / "warnings" / "history.json", 'w') as f:
                json.dump(self._warnings, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Failed to persist state: {str(e)}")
            raise
    
    def _cleanup_old_data(self) -> None:
        """Clean up data older than retention period"""
        if not self.config.retention_days:
            return
            
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        # Clean up metrics history
        self._metrics_history = [
            entry for entry in self._metrics_history
            if datetime.fromisoformat(entry['timestamp']) > cutoff_date
        ]
        
        # Clean up warnings
        self._warnings = [
            entry for entry in self._warnings
            if datetime.fromisoformat(entry['timestamp']) > cutoff_date
        ]
        
        # Persist cleaned state
        self._persist_state()
        logger.info(f"Cleaned up data older than {cutoff_date}")
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save current state"""
        self._state.update(state)
        self._persist_state()
    
    def load_state(self) -> Dict[str, Any]:
        """Load current state"""
        return self._state.copy()
    
    def update_metrics(
        self,
        metrics: Dict[str, float],
        timestamp: datetime
    ) -> None:
        """Update metrics history"""
        entry = {
            "timestamp": timestamp.isoformat(),
            "metrics": metrics
        }
        
        self._metrics_history.append(entry)
        self._persist_state()
        
        # Clean up old data if needed
        if len(self._metrics_history) % 100 == 0:  # Periodic cleanup
            self._cleanup_old_data()
    
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get metrics history as DataFrame"""
        if not self._metrics_history:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "timestamp": datetime.fromisoformat(entry['timestamp']),
                **entry['metrics']
            }
            for entry in self._metrics_history
        ])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Filter by time range
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]
        
        return df
    
    def add_warning(self, message: str) -> None:
        """Add warning message"""
        warning = {
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        
        self._warnings.append(warning)
        self._persist_state()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status summary"""
        now = datetime.now()
        
        # Get recent metrics
        recent_metrics = self.get_metrics_history(
            start_time=now - timedelta(hours=24)
        )
        
        # Get recent warnings
        recent_warnings = [
            w for w in self._warnings
            if datetime.fromisoformat(w['timestamp']) > now - timedelta(hours=24)
        ]
        
        return {
            "last_update": self._state.get("last_update"),
            "metrics_count": len(self._metrics_history),
            "recent_metrics_count": len(recent_metrics),
            "recent_warnings_count": len(recent_warnings),
            "storage_path": str(self.storage_path),
            "retention_days": self.config.retention_days
        }
