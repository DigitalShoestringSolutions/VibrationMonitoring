from celery import Celery
import time
import logging
from typing import Dict, Any
from .influx import influx_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
celery = Celery('tasks', broker='redis://redis:6379/0')

# Configure Celery
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/London',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3000,  # 50 minutes
)

@celery.task(bind=True, max_retries=3)
def process_vibration_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process vibration data by querying InfluxDB and performing analysis.
    
    Args:
        data: Dictionary containing:
            - start_time: ISO format start timestamp
            - end_time: ISO format end timestamp
    
    Returns:
        Dictionary containing analysis results
    """
    try:
        logger.info(f"Starting vibration analysis for time range: {data}")
        
        # Query InfluxDB for data in the specified time range
        vibration_data = influx_service.query_vibration_data(
            start_time=data['start_time'],
            end_time=data['end_time']
        )
        
        if not vibration_data:
            logger.warning("No vibration data found for the specified time range")
            return {
                'status': 'completed',
                'message': 'No data found for the specified time range',
                'analysis_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z')
            }
        
        # Import analysis module only in worker context
        from .analysis import analyze_vibration_data as analyze_data
        
        # Perform analysis on the retrieved data
        analysis_result = analyze_data(vibration_data)
        
        result = {
            **analysis_result,
            'status': 'completed',
            'analysis_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'data_start_time': data['start_time'],
            'data_end_time': data['end_time'],
            'task_id': self.request.id,
        }
        
        logger.info(f"Analysis completed: {result}")

        # Save the result to InfluxDB
        influx_service.save_analysis_result(result)

        return result
        
    except Exception as exc:
        logger.error(f"Error in vibration analysis: {str(exc)}")
        self.retry(exc=exc, countdown=60)  # Retry after 1 minute


@celery.task(bind=True, max_retries=3)
def train_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fine tune a model

    Args:
        data: start_time, end_time
    """
    try:
        logger.info(f"Starting model training with data: {data}")
        
        # Query training data from InfluxDB
        training_data = influx_service.query_vibration_data(
            start_time=data['start_time'],
            end_time=data['end_time']
        )

        if not training_data:
            logger.warning("No training data found for the specified time range")
            return {
                'status': 'completed', 
                'message': 'No data found for the specified time range',
                'training_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z')
            }

        # Import training module only in worker context
        from .train import finetune_model

        # Perform model training
        training_result = finetune_model(training_data, data['start_time'], data['end_time'])
        
        logger.info(f"Model training completed: {training_result}")

        return training_result
    
    except Exception as exc:
        logger.error(f"Error in model training: {str(exc)}")
        self.retry(exc=exc, countdown=60)
        