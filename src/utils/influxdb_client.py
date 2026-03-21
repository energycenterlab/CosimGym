from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS, WriteOptions
from datetime import datetime, timedelta
import time
import subprocess
import requests
from pathlib import Path
import logging
import os


# TODO: all of this should be called result manager including dashbopard launch


class InfluxClient:
    def __init__(self, batch_size=100, logger=None, auto_start=True):
        
        self.logger = logger or logging.getLogger(__name__)
        self.auto_start = auto_start
        self.batch_size = batch_size
        self.url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        self.token = os.getenv('INFLUXDB_TOKEN', 'mytoken123456')
        self.org = os.getenv('INFLUXDB_ORG', 'myorg')
        
        # Docker configuration
        self.docker_compose_path = Path(__file__).parent.parent / 'influxdb' / 'docker-compose.yaml'
        self.health_check_timeout = 30  # seconds
        
        # Ensure InfluxDB is running before initializing client
        if self.auto_start and not self.is_running():
            self.logger.warning("InfluxDB not running, attempting to start...")
            self.start()
        
       
        
        # Initialize client
        self.client = None
        self.write_api = None
        self.write_batch_api = None
        self._initialize_client()


    
    def _initialize_client(self):
        """Initialize InfluxDB client connection."""
        try:
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            write_options = WriteOptions(
                batch_size=self.batch_size,           # Write in batches of 1000 points
                flush_interval=10_000,     # Flush every 10 seconds
                jitter_interval=2_000,     # Random delay up to 2 seconds
                retry_interval=5_000,      # Retry after 5 seconds on failure
                max_retries=3,             # Maximum retry attempts
                max_retry_delay=30_000,    # Maximum delay between retries
                exponential_base=2         # Exponential backoff multiplier
            )
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.write_batch_api = self.client.write_api(write_options=write_options)
            
            self.logger.info(f"InfluxDB client connected to {self.url}")
        except Exception as e:
            self.logger.error(f"Failed to initialize InfluxDB client: {e}")
            raise
    
    def is_running(self):
        """
        Check if InfluxDB is running and accessible.
        
        Returns:
            bool: True if InfluxDB is healthy
        """
        # Try Docker check first (fastest)
        if self._check_docker_status():
            # Verify with HTTP health check
            return self._check_http_health()
        return False
    
    def _check_docker_status(self):
        """
        Check if InfluxDB Docker container is running.
        
        Returns:
            bool: True if container is running
        """
        try:
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=influxdb', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            running_containers = result.stdout.strip().split('\n')
            is_running = any('influxdb' in container for container in running_containers if container)
            
            if is_running:
                self.logger.debug("InfluxDB Docker container is running")
            
            return is_running
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.debug(f"Docker check failed: {e}")
            return False
    
    def _check_http_health(self):
        """
        Check InfluxDB health via HTTP endpoint.
        
        Returns:
            bool: True if healthy
        """
        try:
            response = requests.get(
                f"{self.url}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                health_data = response.json()
                is_healthy = health_data.get('status') == 'pass'
                
                if is_healthy:
                    self.logger.debug(f"InfluxDB is healthy at {self.url}")
                else:
                    self.logger.warning(f"InfluxDB running but not healthy: {health_data}")
                
                return is_healthy
            
            return False
            
        except (requests.RequestException, requests.Timeout) as e:
            self.logger.debug(f"HTTP health check failed: {e}")
            return False
    
    def start(self):
        """
        Start InfluxDB using docker-compose.
        
        Returns:
            bool: True if started successfully
        """

        # TODO: add the start of dashboard process
        if not self.docker_compose_path.exists():
            self.logger.error(f"Docker compose file not found: {self.docker_compose_path}")
            return False
        
        try:
            self.logger.info(f"Starting InfluxDB from {self.docker_compose_path}")
            
            result = subprocess.run(
                ['docker-compose', '-f', str(self.docker_compose_path), 'up', '-d'],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.docker_compose_path.parent
            )
            
            if result.returncode != 0:
                self.logger.error(f"Failed to start InfluxDB: {result.stderr}")
                return False
            
            self.logger.info("Docker compose started. Waiting for InfluxDB to be healthy...")
            return self._wait_for_health()
            
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout starting InfluxDB")
            return False
        except FileNotFoundError:
            self.logger.error("docker-compose not found. Install Docker and docker-compose")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error starting InfluxDB: {e}")
            return False
    
    def stop(self):
        """
        Stop InfluxDB using docker-compose.
        """
        if not self.docker_compose_path.exists():
            self.logger.warning("Docker compose file not found")
            return
        
        try:
            self.logger.info("Stopping InfluxDB...")
            
            result = subprocess.run(
                ['docker-compose', '-f', str(self.docker_compose_path), 'down'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.docker_compose_path.parent
            )
            
            if result.returncode == 0:
                self.logger.info("InfluxDB stopped successfully")
            else:
                self.logger.warning(f"Error stopping InfluxDB: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Failed to stop InfluxDB: {e}")
    
    def _wait_for_health(self):
        """
        Wait for InfluxDB to become healthy.
        
        Returns:
            bool: True if healthy within timeout
        """
        start_time = time.time()
        
        self.logger.info(f"Waiting up to {self.health_check_timeout}s for InfluxDB...")
        
        while time.time() - start_time < self.health_check_timeout:
            if self._check_http_health():
                elapsed = time.time() - start_time
                self.logger.info(f"InfluxDB healthy after {elapsed:.1f}s")
                return True
            
            time.sleep(1)
        
        self.logger.error(f"InfluxDB not healthy within {self.health_check_timeout}s")
        return False
    
    def get_health_status(self):
        """
        Get detailed health status.
        
        Returns:
            dict: Health check results
        """
        status = {
            'docker_running': False,
            'http_accessible': False,
            'connection_valid': False,
            'overall_healthy': False
        }
        
        status['docker_running'] = self._check_docker_status()
        status['http_accessible'] = self._check_http_health()
        
        # Test actual connection
        if status['http_accessible']:
            try:
                # Try to get health endpoint with auth
                health = self.client.health() if self.client else None
                status['connection_valid'] = health is not None
            except Exception as e:
                self.logger.debug(f"Connection validation failed: {e}")
                status['connection_valid'] = False
        
        status['overall_healthy'] = all([
            status['docker_running'],
            status['http_accessible'],
            status['connection_valid']
        ])
        
        return status
    
    def write_metadata(self, bucket, measurement, tags, fields):
        """Write metadata point to InfluxDB."""
        if not self.write_api:
            self.logger.error("Write API not initialized")
            return False
        
        try:
            point = Point(measurement)
            for tag_key, tag_value in tags.items():
                point.tag(tag_key, tag_value)
            for field_key, field_value in fields.items():
                point.field(field_key, field_value)
            point.time(datetime.utcnow())
            
            self.write_api.write(bucket=bucket, org=self.org, record=point)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write metadata: {e}")
            return False
    
    def write_time_series(self, bucket, measurement, tags, time_series_data):
        """Write time series data to InfluxDB."""
        if not self.write_api:
            self.logger.error("Write API not initialized")
            return False
        
        try:
            for timestamp, fields in time_series_data:
                point = Point(measurement)
                for tag_key, tag_value in tags.items():
                    point.tag(tag_key, tag_value)
                for field_key, field_value in fields.items():
                    point.field(field_key, field_value)
                point.time(timestamp)
                self.write_api.write(bucket=bucket, org=self.org, record=point)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write time series: {e}")
            return False
    
    def write_time_series_batch(self, bucket, measurement, time_series_data):
        """Write time series data in batches to InfluxDB."""
        if not self.write_batch_api:
            self.logger.error("Batch Write API not initialized")
            return False
        
        try:
            points = []
            for datapoint in time_series_data:
                point = Point(datapoint['measurement'])
                for tag_key, tag_value in datapoint['tags'].items():
                    point.tag(tag_key, tag_value)
                for field_key, field_value in datapoint['fields'].items():
                    point.field(field_key, field_value)
                point.time(datapoint['time'])
                points.append(point)
     
            self.write_batch_api.write(bucket=bucket, org=self.org, record=points)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write batch time series: {e}")
            return False
        
    def close(self):
        """Flush pending writes and close InfluxDB client connection."""
        # Close write APIs first so their internal buffers are flushed before
        # the underlying HTTP client is torn down.  The async write_batch_api
        # runs a background daemon thread; calling .close() joins that thread
        # and waits for all buffered points to be delivered.
        if self.write_batch_api:
            try:
                self.write_batch_api.close()
            except Exception as e:
                self.logger.warning(f"Error closing batch write API: {e}")
        if self.write_api:
            try:
                self.write_api.close()
            except Exception as e:
                self.logger.warning(f"Error closing write API: {e}")
        if self.client:
            self.client.close()
            self.logger.info("InfluxDB client connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()