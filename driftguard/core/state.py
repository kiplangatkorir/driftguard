"""
State management module for DriftGuard.
Handles persistence of monitoring state, metrics, and drift history.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import logging
from .interfaces import IStateManager

logger = logging.getLogger(__name__)

class StateManager(IStateManager):
    """Manages persistence of monitoring state"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize state manager"""
        self.config = config
        self.storage_path = Path(config['path'])
        self.metrics_file = self.storage_path / 'metrics.csv'
        self.state_file = self.storage_path / 'state.json'
        
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize empty state if not exists
        if not self.state_file.exists():
            self.save_state({
                'last_update': datetime.now().isoformat(),
                'monitoring': {
                    'total_samples_processed': 0,
                    'last_drift_detected': None
                }
            })
        
        # Initialize empty metrics file if not exists
        if not self.metrics_file.exists():
            pd.DataFrame(columns=['timestamp']).to_csv(
                self.metrics_file,
                index=False
            )
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save current state"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug("Saved state")
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")
            raise
    
    def load_state(self) -> Dict[str, Any]:
        """Load saved state"""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            return state
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            raise
    
    def update_metrics(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update metrics history"""
        try:
            # Load existing metrics
            df = pd.read_csv(self.metrics_file)
            
            # Create new row
            new_row = {
                'timestamp': timestamp or datetime.now()
            }
            new_row.update(metrics)
            
            # Append new row
            df = pd.concat([
                df,
                pd.DataFrame([new_row])
            ], ignore_index=True)
            
            # Apply retention policy
            if self.config['retention_days']:
                cutoff = pd.Timestamp.now() - pd.Timedelta(
                    days=self.config['retention_days']
                )
                df = df[pd.to_datetime(df['timestamp']) > cutoff]
            
            # Save updated metrics
            df.to_csv(self.metrics_file, index=False)
            logger.debug("Updated metrics")
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {str(e)}")
            raise
    
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get metrics history"""
        try:
            # Load metrics
            df = pd.read_csv(self.metrics_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Apply time filters
            if start_time:
                df = df[df['timestamp'] >= start_time]
            if end_time:
                df = df[df['timestamp'] <= end_time]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get metrics history: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            state = self.load_state()
            metrics_df = self.get_metrics_history(
                start_time=datetime.now() - pd.Timedelta(days=7)
            )
            
            status = {
                'last_update': state['last_update'],
                'monitoring': state['monitoring'],
                'metrics': {
                    'last_7_days': {
                        col: {
                            'mean': metrics_df[col].mean(),
                            'std': metrics_df[col].std(),
                            'min': metrics_df[col].min(),
                            'max': metrics_df[col].max()
                        }
                        for col in metrics_df.columns
                        if col != 'timestamp'
                    }
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {str(e)}")
            raise
