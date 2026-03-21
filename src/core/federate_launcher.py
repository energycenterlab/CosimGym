"""
federate_launcher.py

Entry point for launching HELICS federates as standalone processes with dynamic configuration.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

import argparse
import sys
import os
from pathlib import Path
import pprint

pp =  pprint.PrettyPrinter(indent=4)
# Fix imports to work from both project root and when called as subprocess
current_dir = Path(__file__).parent
src_dir = current_dir.parent

# Add src directory to Python path if not already there
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.config_reader import read_yaml, reconstruct_federate_config_from_dict
from utils.logging_config import setup_process_logger
from utils.redis_client import RedisClient


def main():
    """
    Main entry point when running as separate process.
    
    This function parses command line arguments, loads configuration from Redis,
    and dynamically instantiates and runs the appropriate federate class.
    
    Command line arguments:
        --name: Required. Name of the federate
        --type: Required. Type/class of the federate
        --simid: Required. Simulation ID
        --redis-url: Required. Redis connection URL
        --redis-key: Required. Redis key for configuration
        --log-file: Required. Path to log file
        --log-level: Required. Logging level
        
    Raises:
        SystemExit: If configuration loading fails or federate execution fails
    """
    parser = argparse.ArgumentParser(description='Run a HELICS federate')
    parser.add_argument('--name', required=True, help='name of the federate')
    parser.add_argument('--scenario_name', required=True, help='name of the scenario')
    parser.add_argument('--federation_name', required=True, help='name of the federation')
    parser.add_argument('--type', required=True, help='Federate type')
    parser.add_argument('--simid', required=True, help='Simulation ID')
    parser.add_argument('--redis-url', required=True, help='Redis connection URL')
    parser.add_argument('--redis-key', required=True, help='Redis key for configuration')
    parser.add_argument('--log-file', required=True, help='log_file path for logging')
    parser.add_argument('--log-level', required=True, help='log level for logging')

    args = parser.parse_args()
    
    logger = setup_process_logger(
        process_name=args.name,
        process_type =f"federate_{args.type}",
        log_file_path = args.log_file,
        log_level = args.log_level
    )
    logger.info(f"Starting federate launcher for {args.name} of type {args.type}")

    # Load federate type mappings
    fed_types = read_yaml('src/core/mappings.yaml')['federate_types']
    # Get federate type configuration
    if args.type not in fed_types:
        logger.error(f"Process {os.getpid()}: Unknown federate type: {args.type}")
        sys.exit(1)


    # Load configuration from Redis
    try:
        # Parse Redis URL to extract host and port
        # Format: redis://host:port/db
        redis_url_parts = args.redis_url.replace('redis://', '').split('/')
        host_port = redis_url_parts[0].split(':')
        redis_host = host_port[0]
        redis_port = int(host_port[1]) if len(host_port) > 1 else 6379
        redis_db = int(redis_url_parts[1]) if len(redis_url_parts) > 1 else 0 
        redis_client = RedisClient(host=redis_host, port=redis_port, db=redis_db, logger=logger)
        config_dict = redis_client.get_json_path(args.redis_key, path=f'$.federations.{args.federation_name}.federate_configs.{args.name}')[0]
        rl_task = redis_client.get_json_path(args.redis_key, path='$.reinforcement_learning_config')[0]
        logger.debug(f"Raw configuration dict retrieved from Redis for federate {args.name}: {pp.pformat(config_dict)}")
        logger.debug(f"Raw RL task config retrieved from Redis: {pp.pformat(rl_task)}")

        if config_dict is None:
            raise ValueError(f"Configuration not found in Redis at key: {args.redis_key}")
        else:
            logger.info(f"Successfully retrieved configuration from Redis key: {args.redis_key}")
        
         # Reconstruct FederateConfig from dict (handles nested dataclasses)
        if args.type == 'rl':
            config = reconstruct_federate_config_from_dict(config_dict, rl_task=rl_task)
        else:
            mode = 'train' if rl_task and rl_task.get('training', {}) and rl_task['training'].get('mode', 'offline')!='offline' else 'test'
            training = rl_task.get('training', {}) if rl_task else {}
            rl_4_fed = {'reset_period':   training.get('reset_period'),
                        'reset_type':     training.get('reset_mode'),
                        'episode_length': training.get('episode_length'),
                        'n_episodes':     training.get('n_episodes'),
                        'rolling_window':   training.get('rolling_window'),
                        'mode': mode}
                
            
            config = reconstruct_federate_config_from_dict(config_dict, rl_config=rl_4_fed)

    
    
    except Exception as e:
        logger.error(f"Failed with exception: {e}")
        sys.exit(1)
    
   
    
    
    logger.info(f"=== FEDERATE {args.name} STARTUP ===")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Type: {args.type}")
    logger.info(f"Redis key: {args.redis_key}")

    
    
    # Parse module and class names from mapping
    module_name = fed_types[args.type].split(':')[0]
    class_name = fed_types[args.type].split(':')[1]
    
    # Dynamic import and instantiation
    try:
        module = __import__(module_name, fromlist=[class_name])
        federate_class = getattr(module, class_name)
    
        federate = federate_class(args.name, config, logger, args.simid, federation_name=args.federation_name)
        
        logger.info("Succesfully created federate!")
        
        # Execute federate lifecycle
        federate.initialize()
        federate.run()
        federate.finalize()
        
    except Exception as e:
        logger.error(f"Process {os.getpid()}: Error running federate {args.name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # When launched by FederationManager, run with arguments
    if len(sys.argv) > 1:
     
        main()
    else:
        raise RuntimeError("Federate launcher must be run with arguments")