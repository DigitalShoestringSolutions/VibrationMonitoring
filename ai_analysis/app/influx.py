from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
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
        self.measurement = os.getenv('INFLUXDB_MEASUREMENT')
        
        self.client = InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org
        )
        self.query_api = self.client.query_api()
        
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
        finally:
            self.client.close()

# Create a singleton instance
influx_service = InfluxDBService()
