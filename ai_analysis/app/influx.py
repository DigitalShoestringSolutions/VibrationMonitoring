from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client import Point
import os
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InfluxDBService:
    def __init__(self):
        self.url = os.getenv('INFLUXDB_URL', 'http://timeseries-db.docker.local:8086')
        self.token = os.getenv('INFLUXDB_TOKEN')
        self.org = os.getenv('INFLUXDB_ORG')
        self.bucket = os.getenv('INFLUXDB_BUCKET')
        self.measurement = os.getenv('INFLUXDB_MEASUREMENT_ANALYSIS')
        self._client = None
        self._query_api = None
        self._write_api = None

    @property
    def client(self):
        if self._client is None:
            self._client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org
            )
        return self._client

    @property
    def query_api(self):
        if self._query_api is None:
            self._query_api = self.client.query_api()
        return self._query_api

    @property
    def write_api(self):
        if self._write_api is None:
            self._write_api = self.client.write_api(write_options=SYNCHRONOUS)
        return self._write_api

    def query_vibration_data(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """
        Query vibration data from InfluxDB within a time range.
        
        Args:
            start_time: ISO format start timestamp
            end_time: ISO format end timestamp
            
        Returns:
            List of vibration data points
        """
        try:
            query = f'''
                from(bucket: "{self.bucket}")
                    |> range(start: {start_time}, stop: {end_time})
                    |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")
            '''
            
            logger.info(f"Querying InfluxDB: {query}")
            result = self.query_api.query(query=query, org=self.org)
            
            # Process the result into a list of dictionaries
            data_points = []
            for table in result:
                for record in table.records:
                    data_points.append({
                        'timestamp': record.get_time().isoformat(),
                        'acceleration': record.values.get('_value')
                    })
            
            logger.info(f"Retrieved {len(data_points)} data points")
            return data_points
            
        except Exception as e:
            logger.error(f"Error querying InfluxDB: {str(e)}")
            raise

    def save_analysis_result(self, result: Dict[str, Any]) -> None:
        """
        Save the analysis result to InfluxDB.
        
        Args:
            result: Dictionary containing analysis results
        """
        try:
            # Create a new measurement for analysis results
            analysis_measurement = "analysis_results"
            
            # Create a new point for the analysis result
            point = (
                Point(analysis_measurement)
                .tag("analysis_model", result['analysis_model'])
                .tag("status", result['status'])
                .tag("timestamp", result['analysis_timestamp'])
                .field("data_points_analyzed", result['data_points_analyzed'])
                .field("max_reconstruction_loss", result['max_reconstruction_loss'])
                .field("min_reconstruction_loss", result['min_reconstruction_loss'])
                .field("mean_reconstruction_loss", result['mean_reconstruction_loss'])
                .field("num_batches_processed", result['num_batches_processed'])
                .field("data_start_time", result['data_start_time'])
                .field("data_end_time", result['data_end_time'])
            )
            
            # Write the point to InfluxDB
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            
            logger.info(f"Analysis result saved to InfluxDB: {result}")

        except Exception as e:
            logger.error(f"Error saving analysis result to InfluxDB: {str(e)}")
            raise

    def close(self):
        """Close all InfluxDB connections"""
        if self._write_api:
            self._write_api.close()
        if self._client:
            self._client.close()
        self._client = None
        self._query_api = None
        self._write_api = None

# Create a singleton instance
influx_service = InfluxDBService()
