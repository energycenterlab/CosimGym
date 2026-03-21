"""
redis_client.py

Client wrapper for Redis interactions, managing connection and JSON configuration data.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

import redis
import json
import logging
from typing import Any, Dict, Optional

class RedisClient:
    """
    A simple wrapper around redis-py to handle connecting, setting, 
    and getting JSON configurations for the Cosim_gym framework.
    """
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, logger: Optional[logging.Logger] = None):
        self.host = host
        self.port = port
        self.db = db
        self.logger = logger or logging.getLogger(__name__)
        self.client = None
        self.connect()

    def connect(self):
        """Establish connection to the Redis server."""
        try:
            self.client = redis.Redis(host=self.host, port=self.port, db=self.db, decode_responses=True)
            # Ping to verify connection
            self.client.ping()
            self.logger.info(f"Successfully connected to Redis at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            self.logger.error(f"Failed to connect to Redis at {self.host}:{self.port} - {e}")
            self.client = None
            raise

    def set_json(self, key: str, data: Dict[str, Any], expire_seconds: Optional[int] = None) -> bool:
        """
        Store a dictionary as JSON in Redis using RedisJSON module.
        
        Args:
            key: The Redis key.
            data: The dictionary to store.
            expire_seconds: Optional expiration time in seconds.
        """
        if not self.client:
            self.logger.error("Redis client is not connected.")
            return False
        
        try:
            # Use RedisJSON to store the data
            self.client.json().set(key, '.', data)
            
            # Set expiration if specified
            if expire_seconds:
                self.client.expire(key, expire_seconds)
            
            self.logger.debug(f"Successfully stored JSON data at key: {key}")
            return True
        except (TypeError, redis.RedisError) as e:
            self.logger.error(f"Failed to store JSON data at key {key}: {e}")
            return False

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve JSON data from Redis using RedisJSON module.
        
        Args:
            key: The Redis key.
            
        Returns:
            The deserialized dictionary, or None if the key doesn't exist or an error occurs.
        """
        if not self.client:
            self.logger.error("Redis client is not connected.")
            return None
        
        try:
            result = self.client.json().get(key, '.')
            if result is None:
                self.logger.warning(f"Key not found in Redis: {key}")
                return None
            return result
        except redis.RedisError as e:
            self.logger.error(f"Failed to retrieve JSON data at key {key}: {e}")
            return None
        
    def get_json_path(self, key: str, path: str = '.'):
        """
        Get JSON data at specific path using JSONPath.
        
        Args:
            key: Redis key
            path: JSONPath expression (e.g., '.federations.federation_1', '.influxdb')
                Examples:
                - '.' - get entire JSON
                - '.influxdb' - get influxdb config
                - '.federations.federation_1' - get specific federation
                - '.start_time' - get start_time value
        
        Returns:
            Parsed JSON data at specified path or None
            
        Note:
            Requires RedisJSON module installed on Redis server
        """
        if not self.client:
            self.logger.error("Redis client is not connected.")
            return None
            
        try:
            result = self.client.json().get(key, path)
            if result is not None:
                self.logger.debug(f"Retrieved JSON from {key} at path {path}")
                return result
            self.logger.warning(f"No data found at path {path} in key {key}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to get JSON path {path} from {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        if not self.client:
            return False
        try:
            self.client.delete(key)
            return True
        except redis.RedisError as e:
            self.logger.error(f"Failed to delete key {key}: {e}")
            return False
