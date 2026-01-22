"""
REST API server for DriftGuard.
Provides endpoints for model monitoring and drift detection.
"""
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path
import asyncio
from ..core.guardian import DriftGuard
from ..core.config import ConfigManager
from prometheus_client import Counter, Histogram, generate_latest

# Initialize FastAPI app
app = FastAPI(
    title="DriftGuard API",
    description="API for model monitoring and drift detection",
    version="0.1.6"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize metrics
DRIFT_DETECTIONS = Counter(
    'driftguard_drift_detections_total',
    'Total number of drift detections'
)
MONITORING_REQUESTS = Counter(
    'driftguard_monitoring_requests_total',
    'Total number of monitoring requests'
)
PROCESSING_TIME = Histogram(
    'driftguard_processing_seconds',
    'Time spent processing monitoring requests'
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store active monitoring instances
active_monitors: Dict[str, DriftGuard] = {}

class MonitoringRequest(BaseModel):
    """Request model for monitoring data"""
    project_id: str
    features: Dict[str, List[float]]
    actual_labels: Optional[List[Union[int, float]]] = None
    metadata: Optional[Dict[str, Any]] = None

class MonitoringResponse(BaseModel):
    """Response model for monitoring results"""
    status: str
    timestamp: datetime
    project_id: str
    drift_detected: bool
    drift_reports: List[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, float]]
    warnings: List[str]

@app.post("/monitor", response_model=MonitoringResponse)
async def monitor_data(
    request: MonitoringRequest,
    background_tasks: BackgroundTasks
):
    """
    Monitor a batch of data for drift and performance.
    
    Args:
        request: Monitoring request containing features and optional labels
        background_tasks: FastAPI background tasks
        
    Returns:
        Monitoring results
    """
    MONITORING_REQUESTS.inc()
    
    try:
        # Get or create monitor
        monitor = active_monitors.get(request.project_id)
        if not monitor:
            raise HTTPException(
                status_code=404,
                detail=f"Project {request.project_id} not found"
            )
        
        # Convert features to DataFrame
        features = pd.DataFrame(request.features)
        
        # Convert labels if provided
        actual_labels = None
        if request.actual_labels:
            actual_labels = np.array(request.actual_labels)
        
        # Monitor data
        with PROCESSING_TIME.time():
            results = await monitor.monitor_batch(
                features,
                actual_labels,
                request.metadata
            )
            
        if results["drift_detected"]:
            DRIFT_DETECTIONS.inc()
            
        # Add background task to update metrics
        background_tasks.add_task(
            update_monitoring_metrics,
            request.project_id,
            results
        )
        
        return {
            **results,
            "project_id": request.project_id
        }
        
    except Exception as e:
        logger.error(f"Monitoring failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/projects/{project_id}/initialize")
async def initialize_project(
    project_id: str,
    reference_data: UploadFile = File(...),
    config_file: Optional[UploadFile] = None
):
    """
    Initialize a new monitoring project.
    
    Args:
        project_id: Unique project identifier
        reference_data: CSV file containing reference data
        config_file: Optional YAML configuration file
        
    Returns:
        Project initialization status
    """
    try:
        # Read reference data
        reference_df = pd.read_csv(reference_data.file)
        
        # Load configuration
        config_path = None
        if config_file:
            config_content = await config_file.read()
            config_path = f"/tmp/config_{project_id}.yaml"
            with open(config_path, "wb") as f:
                f.write(config_content)
        
        # Create monitor
        monitor = DriftGuard(
            model=None,  # Model will be provided later
            reference_data=reference_df,
            config_path=config_path,
            project_name=project_id
        )
        
        active_monitors[project_id] = monitor
        
        return {
            "status": "success",
            "message": f"Project {project_id} initialized successfully"
        }
        
    except Exception as e:
        logger.error(f"Project initialization failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/projects/{project_id}/summary")
async def get_project_summary(project_id: str):
    """Get monitoring summary for a project"""
    try:
        monitor = active_monitors.get(project_id)
        if not monitor:
            raise HTTPException(
                status_code=404,
                detail=f"Project {project_id} not found"
            )
            
        return monitor.get_monitoring_summary()
        
    except Exception as e:
        logger.error(f"Failed to get project summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/projects/{project_id}/update-reference")
async def update_reference_data(
    project_id: str,
    reference_data: UploadFile = File(...)
):
    """Update reference data for a project"""
    try:
        monitor = active_monitors.get(project_id)
        if not monitor:
            raise HTTPException(
                status_code=404,
                detail=f"Project {project_id} not found"
            )
            
        # Read new reference data
        reference_df = pd.read_csv(reference_data.file)
        
        # Update reference data
        monitor.update_reference_data(reference_df)
        
        return {
            "status": "success",
            "message": "Reference data updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to update reference data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    return generate_latest()

async def update_monitoring_metrics(
    project_id: str,
    results: Dict[str, Any]
) -> None:
    """Update monitoring metrics in background"""
    try:
        # Update project-specific metrics here
        pass
    except Exception as e:
        logger.error(f"Failed to update metrics: {str(e)}")

def start_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """Start the API server"""
    import uvicorn
    uvicorn.run(
        "driftguard.api.server:app",
        host=host,
        port=port,
        reload=reload
    )
