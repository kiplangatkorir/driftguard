"""
State management system for DriftGuard.
Handles persistence of monitoring state, metrics, and drift history.
"""
from typing import Any, Dict, List, Optional, Union
import json
import pickle
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import threading
from contextlib import contextmanager
import logging
from .interfaces import IStateManager
from .config import StorageConfig

logger = logging.getLogger(__name__)

class StateManager(IStateManager):
    """Thread-safe state management for DriftGuard"""
    
    def __init__(self, storage_config: StorageConfig):
        self.storage_config = storage_config
        self.state_file = Path(storage_config.path) / "state.json"
        self.metrics_file = Path(storage_config.path) / "metrics.parquet"
        self.drift_file = Path(storage_config.path) / "drift_history.parquet"
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = self._initialize_state()
        
    def _initialize_state(self) -> Dict[str, Any]:
        """Initialize or load existing state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                return self._create_default_state()
        return self._create_default_state()
    
    def _create_default_state(self) -> Dict[str, Any]:
        """Create default state structure"""
        return {
            'version': '1.0.0',
            'last_update': datetime.now().isoformat(),
            'monitoring': {
                'start_time': datetime.now().isoformat(),
                'total_samples_processed': 0,
                'last_drift_detected': None,
                'alert_history': [],
            },
            'model': {
                'baseline_metrics': {},
                'current_metrics': {},
                'drift_status': {
                    'has_drift': False,
                    'drifted_features': [],
                },
            },
            'system': {
                'errors': [],
                'warnings': [],
            }
        }
    
    @contextmanager
    def state_lock(self):
        """Thread-safe state access context manager"""
        try:
            self._lock.acquire()
            yield
        finally:
            self._lock.release()
    
    def save_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        """Save current state to storage"""
        with self.state_lock():
            if state is not None:
                self._state.update(state)
            self._state['last_update'] = datetime.now().isoformat()
            
            try:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.state_file, 'w') as f:
                    json.dump(self._state, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                raise
    
    def load_state(self) -> Dict[str, Any]:
        """Load current state"""
        with self.state_lock():
            return self._state.copy()
    
    def clear_state(self) -> None:
        """Clear all saved state"""
        with self.state_lock():
            self._state = self._create_default_state()
            self.save_state()
            
            # Clear metric history
            if self.metrics_file.exists():
                self.metrics_file.unlink()
            if self.drift_file.exists():
                self.drift_file.unlink()
    
    def update_metrics(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """Update performance metrics history"""
        if timestamp is None:
            timestamp = datetime.now()
            
        metrics_df = pd.DataFrame([{
            'timestamp': timestamp,
            **metrics
        }])
        
        try:
            if self.metrics_file.exists():
                existing_metrics = pd.read_parquet(self.metrics_file)
                metrics_df = pd.concat([existing_metrics, metrics_df])
            
            # Apply retention policy
            if self.storage_config.retention_days > 0:
                cutoff = datetime.now() - timedelta(days=self.storage_config.retention_days)
                metrics_df = metrics_df[metrics_df['timestamp'] > cutoff]
            
            metrics_df.to_parquet(
                self.metrics_file,
                compression='snappy' if self.storage_config.compression else None
            )
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
            raise
    
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get metrics history within specified timeframe"""
        try:
            if not self.metrics_file.exists():
                return pd.DataFrame()
                
            metrics_df = pd.read_parquet(self.metrics_file)
            
            if start_time:
                metrics_df = metrics_df[metrics_df['timestamp'] >= start_time]
            if end_time:
                metrics_df = metrics_df[metrics_df['timestamp'] <= end_time]
                
            return metrics_df
        except Exception as e:
            logger.error(f"Failed to read metrics history: {e}")
            return pd.DataFrame()
    
    def update_drift_history(self, drift_reports: List[Dict[str, Any]]) -> None:
        """Update drift detection history"""
        drift_df = pd.DataFrame(drift_reports)
        drift_df['timestamp'] = datetime.now()
        
        try:
            if self.drift_file.exists():
                existing_drift = pd.read_parquet(self.drift_file)
                drift_df = pd.concat([existing_drift, drift_df])
            
            # Apply retention policy
            if self.storage_config.retention_days > 0:
                cutoff = datetime.now() - timedelta(days=self.storage_config.retention_days)
                drift_df = drift_df[drift_df['timestamp'] > cutoff]
            
            drift_df.to_parquet(
                self.drift_file,
                compression='snappy' if self.storage_config.compression else None
            )
        except Exception as e:
            logger.error(f"Failed to update drift history: {e}")
            raise
    
    def get_drift_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get drift detection history within specified timeframe"""
        try:
            if not self.drift_file.exists():
                return pd.DataFrame()
                
            drift_df = pd.read_parquet(self.drift_file)
            
            if start_time:
                drift_df = drift_df[drift_df['timestamp'] >= start_time]
            if end_time:
                drift_df = drift_df[drift_df['timestamp'] <= end_time]
            if features:
                drift_df = drift_df[drift_df['feature_name'].isin(features)]
                
            return drift_df
        except Exception as e:
            logger.error(f"Failed to read drift history: {e}")
            return pd.DataFrame()
    
    def add_error(self, error: str, error_type: str = "system") -> None:
        """Add error to state history"""
        with self.state_lock():
            self._state['system']['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'type': error_type,
                'message': error
            })
            # Keep only last 100 errors
            self._state['system']['errors'] = self._state['system']['errors'][-100:]
            self.save_state()
    
    def add_warning(self, warning: str, warning_type: str = "system") -> None:
        """Add warning to state history"""
        with self.state_lock():
            self._state['system']['warnings'].append({
                'timestamp': datetime.now().isoformat(),
                'type': warning_type,
                'message': warning
            })
            # Keep only last 100 warnings
            self._state['system']['warnings'] = self._state['system']['warnings'][-100:]
            self.save_state()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status summary"""
        with self.state_lock():
            return {
                'last_update': self._state['last_update'],
                'total_samples': self._state['monitoring']['total_samples_processed'],
                'drift_status': self._state['model']['drift_status'],
                'recent_errors': self._state['system']['errors'][-5:],
                'recent_warnings': self._state['system']['warnings'][-5:],
            }
