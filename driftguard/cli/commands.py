"""
Command-line interface for DriftGuard.
Provides easy access to monitoring functionality through CLI.
"""
import click
import pandas as pd
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any
import asyncio
from ..core.guardian import DriftGuard
from ..core.config import ConfigManager
from ..api.server import start_api_server

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """DriftGuard - ML Model Monitoring and Drift Detection"""
    pass

@cli.command()
@click.argument('reference_data', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--model-type', '-t', type=click.Choice(['classification', 'regression']),
              default='classification', help='Type of model to monitor')
@click.option('--project-name', '-n', help='Name for this monitoring instance')
@click.option('--output', '-o', type=click.Path(),
              help='Path to save monitoring results')
def initialize(
    reference_data: str,
    config: Optional[str],
    model_type: str,
    project_name: Optional[str],
    output: Optional[str]
):
    """Initialize DriftGuard with reference data"""
    try:
        # Load reference data
        reference_df = pd.read_csv(reference_data)
        logger.info(f"Loaded reference data: {reference_df.shape}")
        
        # Initialize DriftGuard
        monitor = DriftGuard(
            model=None,  # Model will be provided later
            reference_data=reference_df,
            config_path=config,
            model_type=model_type,
            project_name=project_name
        )
        
        # Save initial state if output specified
        if output:
            summary = monitor.get_monitoring_summary()
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Saved initialization summary to {output}")
            
        click.echo("DriftGuard initialized successfully!")
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--reference', '-r', type=click.Path(exists=True),
              help='Path to reference data')
@click.option('--labels', '-l', type=click.Path(exists=True),
              help='Path to actual labels')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--output', '-o', type=click.Path(),
              help='Path to save monitoring results')
@click.option('--model-type', '-t', type=click.Choice(['classification', 'regression']),
              default='classification', help='Type of model to monitor')
def monitor(
    data: str,
    reference: Optional[str],
    labels: Optional[str],
    config: Optional[str],
    output: Optional[str],
    model_type: str
):
    """Monitor a batch of data for drift"""
    try:
        # Load data
        features_df = pd.read_csv(data)
        logger.info(f"Loaded features data: {features_df.shape}")
        
        # Load reference data if provided
        reference_df = None
        if reference:
            reference_df = pd.read_csv(reference)
            logger.info(f"Loaded reference data: {reference_df.shape}")
        
        # Load labels if provided
        labels_data = None
        if labels:
            labels_data = pd.read_csv(labels)
            if len(labels_data.shape) > 1 and labels_data.shape[1] > 1:
                click.echo("Warning: Multiple columns in labels file, using first column")
                labels_data = labels_data.iloc[:, 0]
            logger.info(f"Loaded labels data: {len(labels_data)}")
        
        # Initialize monitor
        monitor = DriftGuard(
            model=None,
            reference_data=reference_df or features_df,
            config_path=config,
            model_type=model_type
        )
        
        # Run monitoring
        results = asyncio.run(
            monitor.monitor_batch(
                features_df,
                labels_data
            )
        )
        
        # Save results if output specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Saved monitoring results to {output}")
        
        # Display summary
        click.echo("\nMonitoring Results:")
        click.echo(f"Status: {results['status']}")
        click.echo(f"Timestamp: {results['timestamp']}")
        click.echo(f"Sample Size: {results['sample_size']}")
        click.echo(f"Drift Detected: {results['drift_detected']}")
        
        if results['drift_reports']:
            click.echo("\nDrift Reports:")
            for report in results['drift_reports']:
                click.echo(
                    f"- {report['feature_name']}: "
                    f"score={report['drift_score']:.3f}"
                )
        
        if results['performance_metrics']:
            click.echo("\nPerformance Metrics:")
            for metric, value in results['performance_metrics'].items():
                click.echo(f"- {metric}: {value:.3f}")
        
    except Exception as e:
        logger.error(f"Monitoring failed: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--host', '-h', default='0.0.0.0',
              help='Host to bind the server to')
@click.option('--port', '-p', default=8000,
              help='Port to bind the server to')
@click.option('--reload', is_flag=True,
              help='Enable auto-reload on code changes')
def serve(host: str, port: int, reload: bool):
    """Start the DriftGuard API server"""
    try:
        click.echo(f"Starting DriftGuard API server on {host}:{port}")
        start_api_server(host, port, reload)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('project_dir', type=click.Path(exists=True))
@click.option('--reference', '-r', type=click.Path(exists=True),
              help='Path to reference data')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--interval', '-i', default=60,
              help='Monitoring interval in seconds')
def watch(
    project_dir: str,
    reference: Optional[str],
    config: Optional[str],
    interval: int
):
    """Watch a directory for new data files and monitor them"""
    try:
        project_path = Path(project_dir)
        
        # Load reference data if provided
        reference_df = None
        if reference:
            reference_df = pd.read_csv(reference)
            logger.info(f"Loaded reference data: {reference_df.shape}")
        
        # Initialize monitor
        monitor = DriftGuard(
            model=None,
            reference_data=reference_df,
            config_path=config
        )
        
        click.echo(f"Watching {project_dir} for new data files...")
        
        def process_file(file_path: Path) -> None:
            try:
                # Load and monitor data
                features_df = pd.read_csv(file_path)
                results = asyncio.run(monitor.monitor_batch(features_df))
                
                # Save results
                output_path = file_path.parent / f"{file_path.stem}_results.json"
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                    
                logger.info(f"Processed {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
        
        # Watch directory
        import time
        processed_files = set()
        
        while True:
            # Check for new CSV files
            for file_path in project_path.glob("*.csv"):
                if file_path not in processed_files:
                    process_file(file_path)
                    processed_files.add(file_path)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        click.echo("\nStopping watch...")
    except Exception as e:
        logger.error(f"Watch failed: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli()
