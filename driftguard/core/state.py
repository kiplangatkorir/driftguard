"""
State management module for DriftGuard.
"""
from typing import Any, Dict, List, Optional
import json
import os
from datetime import datetime
import pandas as pd
from pathlib import Path

from .interfaces import IStateManager

class StateManager(IStateManager):
    """Manages persistence of monitoring state and metrics"""
    
    def __init__(
        self,
        path: str = "./storage",
        retention_days: int = 30
    ):
        """Initialize state manager"""
        self.base_path = Path(path)
        self.retention_days = retention_days
        
        # Create storage directories
        self.state_path = self.base_path / "state"
        self.metrics_path = self.base_path / "metrics"
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize storage directories"""
        self.state_path.mkdir(parents=True, exist_ok=True)
        self.metrics_path.mkdir(parents=True, exist_ok=True)
    
    def _get_current_state_file(self) -> Path:
        """Get path to current state file"""
        return self.state_path / "current_state.json"
    
    def _get_metrics_file(self, timestamp: datetime) -> Path:
        """Get path to metrics file for given timestamp"""
        date_str = timestamp.strftime("%Y-%m-%d")
        return self.metrics_path / f"metrics_{date_str}.csv"
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save current state"""
        state_file = self._get_current_state_file()
        
        # Add timestamp to state
        state['last_updated'] = datetime.now().isoformat()
        
        # Save state
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self) -> Dict[str, Any]:
        """Load saved state"""
        state_file = self._get_current_state_file()
        
        if not state_file.exists():
            return {}
        
        with open(state_file, 'r') as f:
            return json.load(f)
    
    def update_metrics(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update metrics history"""
        if timestamp is None:
            timestamp = datetime.now()
        
        metrics_file = self._get_metrics_file(timestamp)
        
        # Add timestamp to metrics
        metrics['timestamp'] = timestamp.isoformat()
        
        # Convert to DataFrame
        df = pd.DataFrame([metrics])
        
        # Append or create metrics file
        if metrics_file.exists():
            existing_df = pd.read_csv(metrics_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        # Save metrics
        df.to_csv(metrics_file, index=False)
        
        # Clean up old metrics
        self._cleanup_old_metrics()
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics files older than retention period"""
        cutoff_date = datetime.now().timestamp() - (
            self.retention_days * 24 * 60 * 60
        )
        
        for file in self.metrics_path.glob("metrics_*.csv"):
            if file.stat().st_mtime < cutoff_date:
                file.unlink()
    
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get metrics history within time range"""
        # Get all metrics files
        metrics_files = list(self.metrics_path.glob("metrics_*.csv"))
        if not metrics_files:
            return pd.DataFrame()
        
        # Read and concatenate all metrics
        dfs = []
        for file in metrics_files:
            df = pd.read_csv(file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine all metrics
        metrics_df = pd.concat(dfs, ignore_index=True)
        
        # Filter by time range
        if start_time:
            metrics_df = metrics_df[metrics_df['timestamp'] >= start_time]
        if end_time:
            metrics_df = metrics_df[metrics_df['timestamp'] <= end_time]
        
        return metrics_df.sort_values('timestamp')
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        state = self.load_state()
        
        # Get latest metrics
        latest_metrics = pd.DataFrame()
        metrics_files = list(self.metrics_path.glob("metrics_*.csv"))
        if metrics_files:
            latest_file = max(metrics_files, key=lambda p: p.stat().st_mtime)
            latest_metrics = pd.read_csv(latest_file)
            if not latest_metrics.empty:
                latest_metrics = latest_metrics.iloc[-1].to_dict()
        
        status = {
            'state': state,
            'latest_metrics': latest_metrics,
            'storage': {
                'base_path': str(self.base_path),
                'retention_days': self.retention_days,
                'metrics_files': len(metrics_files)
            }
        }
        
        return status
