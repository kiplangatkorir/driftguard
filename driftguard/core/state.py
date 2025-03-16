"""
State management module for DriftGuard.
"""
from typing import Dict, Optional
import os
import json
import shutil
from datetime import datetime, timedelta
import pandas as pd

from .interfaces import IStateManager

class StateManager(IStateManager):
    """Manages persistence of monitoring state and metrics"""
    
    def __init__(self, path: str, retention_days: int = 7):
        """Initialize state manager"""
        self.path = path
        self.retention_days = retention_days
        
        # Create storage directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(path, 'state'), exist_ok=True)
    
    def _get_state_file(self) -> str:
        """Get path to state file"""
        return os.path.join(self.path, 'state', 'current_state.json')
    
    def _get_metrics_file(self, timestamp: datetime) -> str:
        """Get path to metrics file for given timestamp"""
        date_str = timestamp.strftime('%Y-%m-%d')
        return os.path.join(
            self.path,
            'metrics',
            f'metrics_{date_str}.csv'
        )
    
    def save_state(self, state: Dict) -> None:
        """Save current state"""
        state_file = self._get_state_file()
        
        # Add timestamp to state
        state['timestamp'] = datetime.now().isoformat()
        
        # Save state to file
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self) -> Dict:
        """Load saved state"""
        state_file = self._get_state_file()
        
        if not os.path.exists(state_file):
            return {}
        
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Convert timestamp back to datetime
        if 'timestamp' in state:
            state['timestamp'] = datetime.fromisoformat(state['timestamp'])
        
        return state
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update metrics history"""
        timestamp = datetime.now()
        metrics_file = self._get_metrics_file(timestamp)
        
        # Add timestamp to metrics
        metrics['timestamp'] = timestamp
        
        # Convert metrics to DataFrame
        df = pd.DataFrame([metrics])
        
        # Append or create metrics file
        if os.path.exists(metrics_file):
            existing_df = pd.read_csv(metrics_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        # Save metrics
        df.to_csv(metrics_file, index=False)
        
        # Clean up old metrics files
        self._cleanup_old_files()
    
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get metrics history"""
        metrics_dir = os.path.join(self.path, 'metrics')
        
        # Get list of metrics files
        metrics_files = []
        for file in os.listdir(metrics_dir):
            if file.startswith('metrics_') and file.endswith('.csv'):
                file_path = os.path.join(metrics_dir, file)
                metrics_files.append(file_path)
        
        if not metrics_files:
            return pd.DataFrame()
        
        # Read and concatenate all metrics files
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
        
        # Filter by time range if specified
        if start_time:
            metrics_df = metrics_df[metrics_df['timestamp'] >= start_time]
        if end_time:
            metrics_df = metrics_df[metrics_df['timestamp'] <= end_time]
        
        return metrics_df.sort_values('timestamp')
    
    def _cleanup_old_files(self) -> None:
        """Remove files older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Clean up metrics files
        metrics_dir = os.path.join(self.path, 'metrics')
        for file in os.listdir(metrics_dir):
            if not file.startswith('metrics_'):
                continue
            
            try:
                # Extract date from filename
                date_str = file.replace('metrics_', '').replace('.csv', '')
                file_date = datetime.strptime(date_str, '%Y-%m-%d')
                
                # Remove if older than cutoff
                if file_date < cutoff_date:
                    os.remove(os.path.join(metrics_dir, file))
            except (ValueError, OSError):
                continue
    
    def clear_storage(self) -> None:
        """Clear all stored state and metrics"""
        try:
            shutil.rmtree(self.path)
            os.makedirs(self.path, exist_ok=True)
            os.makedirs(os.path.join(self.path, 'metrics'), exist_ok=True)
            os.makedirs(os.path.join(self.path, 'state'), exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to clear storage: {e}")
