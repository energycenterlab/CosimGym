#!/usr/bin/env python3
"""
fmu_catalog_register.py

CLI tool that auto-registers an FMU into the CosimGym model catalog.

It reads the FMU's modelDescription.xml via fmpy to extract variable
metadata (inputs, outputs, parameters, units, defaults), optionally uploads
the .fmu binary to MinIO, appends the generated entry to catalog.yaml, and
pushes the updated catalog to Redis.

Usage
-----
# Register with MinIO upload (default):
python src/models/model_catalog/fmu_catalog_register.py \\
    --fmu /path/to/Model.fmu \\
    --name my_model_name

# Register using a local path (no upload):
python src/models/model_catalog/fmu_catalog_register.py \\
    --fmu /path/to/Model.fmu \\
    --name my_model_name \\
    --local

# Full options:
python src/models/model_catalog/fmu_catalog_register.py \\
    --fmu /path/to/Model.fmu \\
    --name my_model_name \\
    --domain building_energy \\
    --tags fmu thermal \\
    --minio-endpoint http://localhost:9000 \\
    --minio-access-key cosimgym \\
    --minio-secret-key cosimgym123 \\
    --minio-bucket fmus \\
    --no-redis   # skip Redis push (update catalog.yaml only)

Author: Pietro Rando Mazzarino
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CATALOG_YAML = Path(__file__).parent / "catalog.yaml"

# FMI causality → catalog section
_CAUSALITY_SECTION = {
    'input':               'inputs',
    'output':              'outputs',
    'parameter':           'parameters',
    'calculatedParameter': 'parameters',
}

# FMI type string → catalog type string
_FMI_TO_CATALOG_TYPE = {
    'Real':        'float',
    'Integer':     'int',
    'Boolean':     'bool',
    'String':      'string',
    'Enumeration': 'int',
}


# ---------------------------------------------------------------------------
# FMU parsing
# ---------------------------------------------------------------------------

def parse_fmu(fmu_path: str) -> dict:
    """
    Read modelDescription.xml via fmpy and build a catalog entry dict.

    Returns a dict ready to be inserted under the model name key in catalog.yaml.
    The fmu_source block is NOT filled here — it is added by the caller after
    the upload step so the path/URL is known.
    """
    try:
        from fmpy import read_model_description
    except ImportError:
        logger.error("fmpy not installed. Activate the cosim_gym conda env first.")
        sys.exit(1)

    md = read_model_description(fmu_path, validate=False)

    inputs = {}
    outputs = {}
    parameters = {}

    for v in md.modelVariables:
        section_key = _CAUSALITY_SECTION.get(v.causality)
        if section_key is None:
            continue  # skip local / independent / unknown causality

        fmi_type = v.type if v.type else 'Real'
        catalog_type = _FMI_TO_CATALOG_TYPE.get(fmi_type, 'float')
        unit = v.unit if hasattr(v, 'unit') and v.unit else ''
        start = v.start if hasattr(v, 'start') and v.start is not None else 0.0
        description = v.description if hasattr(v, 'description') and v.description else ''

        try:
            default_value = float(start) if catalog_type == 'float' else start
        except (TypeError, ValueError):
            default_value = start

        spec = {
            'type':          catalog_type,
            'default_value': default_value,
            'description':   description,
            'unit':          unit,
            'required':      False,
            'tags':          ['fmu'],
        }

        if section_key == 'inputs':
            inputs[v.name] = spec
        elif section_key == 'outputs':
            outputs[v.name] = spec
        elif section_key == 'parameters':
            parameters[v.name] = spec

    fmi_version = md.fmiVersion
    model_name_fmu = md.modelName if md.modelName else Path(fmu_path).stem
    fmu_description = md.description if md.description else f"FMU model: {model_name_fmu}"
    fmu_author = md.author if md.author else ''
    fmu_version = md.version if md.version else '1.0.0'

    entry = {
        'class_name':   'BaseFMUModel',
        'module_path':  'models.base_FMU_model',
        'version':      fmu_version,
        'description':  fmu_description,
        'author':       fmu_author,
        'domain':       '',
        'category':     'physical_model',
        'tags':         ['fmu', f'fmi{fmi_version}'],
        'dependencies': ['fmpy'],
        'time_step':    60,
        'max_time_step': 3600,
        'min_time_step': 1,
        'user_defined': {
            'fmu_source': {}  # filled by caller
        },
        'parameters': parameters,
        'inputs':      inputs,
        'outputs':     outputs,
    }

    logger.info(
        f"Parsed FMU '{model_name_fmu}' (FMI {fmi_version}): "
        f"{len(inputs)} inputs, {len(outputs)} outputs, {len(parameters)} parameters"
    )
    return entry


# ---------------------------------------------------------------------------
# MinIO upload
# ---------------------------------------------------------------------------

def upload_to_minio(
    fmu_path: str,
    model_name: str,
    version: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
) -> dict:
    """
    Upload the FMU binary to MinIO and return a populated fmu_source dict.
    """
    try:
        from minio import Minio
        from minio.error import S3Error
    except ImportError:
        logger.error("minio package not installed. Run: pip install minio>=7.0.0")
        sys.exit(1)

    secure = endpoint.startswith('https://')
    clean_endpoint = endpoint.replace('https://', '').replace('http://', '')

    client = Minio(clean_endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info(f"Created bucket '{bucket}'")

    filename = Path(fmu_path).name
    object_key = f"{model_name}/{version}/{filename}"

    client.fput_object(bucket, object_key, fmu_path)
    logger.info(f"Uploaded FMU to MinIO: {bucket}/{object_key}")

    return {
        'type':       'minio',
        'endpoint':   endpoint,
        'bucket':     bucket,
        'object_key': object_key,
        'access_key': access_key,
        'secret_key': secret_key,
    }


# ---------------------------------------------------------------------------
# catalog.yaml read / write
# ---------------------------------------------------------------------------

def load_catalog(path: Path) -> dict:
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    if 'models' not in data:
        data['models'] = {}
    return data


def save_catalog(path: Path, catalog: dict) -> None:
    with open(path, 'w') as f:
        yaml.dump(catalog, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    logger.info(f"catalog.yaml updated at {path}")


# ---------------------------------------------------------------------------
# Redis push
# ---------------------------------------------------------------------------

def push_to_redis(catalog: dict) -> None:
    """Re-run the catalog_loader logic to push the updated catalog to Redis."""
    try:
        import redis as redis_lib
    except ImportError:
        logger.error("redis package not installed.")
        sys.exit(1)

    host = os.getenv('REDIS_HOST', 'localhost')
    port = int(os.getenv('REDIS_PORT', '6379'))

    CATEGORY_MAP = {
        'physical_model': 'physical_models',
        'rl_agent':       'rl_agents',
    }
    DEFAULT_CATEGORY = 'other'

    for attempt in range(1, 6):
        try:
            client = redis_lib.Redis(host=host, port=port, db=0, decode_responses=True)
            client.ping()
            break
        except redis_lib.ConnectionError:
            logger.warning(f"Redis not reachable (attempt {attempt}/5), retrying in 2s…")
            time.sleep(2)
    else:
        logger.error("Could not connect to Redis. Catalog.yaml was updated but Redis was NOT.")
        return

    models = catalog.get('models', {})
    index: dict = {}

    try:
        existing_index = client.json().get('catalog:index', '.') or {}
        index = existing_index
    except Exception:
        pass

    for model_name, model_data in models.items():
        category_raw = model_data.get('category', DEFAULT_CATEGORY)
        category_key = CATEGORY_MAP.get(category_raw, DEFAULT_CATEGORY)
        redis_key = f"catalog:{category_key}:{model_name}"
        payload = {'name': model_name, **model_data}
        client.json().set(redis_key, '.', payload)

        index.setdefault(category_key, [])
        if model_name not in index[category_key]:
            index[category_key].append(model_name)

    client.json().set('catalog:index', '.', index)
    logger.info(f"Redis catalog updated at {host}:{port}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Auto-register an FMU into the CosimGym model catalog.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--fmu',            required=True,  help='Path to the .fmu file')
    p.add_argument('--name',           required=True,  help='Catalog key for this model (e.g. my_building_fmu)')
    p.add_argument('--domain',         default='',     help='Domain tag (e.g. building_energy)')
    p.add_argument('--tags',           nargs='*',      default=[], help='Extra tags')
    p.add_argument('--local',          action='store_true',
                   help='Skip MinIO upload; store the absolute local path instead')
    p.add_argument('--minio-endpoint', default='http://localhost:9000')
    p.add_argument('--minio-access-key', default=os.getenv('MINIO_ACCESS_KEY', 'cosimgym'))
    p.add_argument('--minio-secret-key', default=os.getenv('MINIO_SECRET_KEY', 'cosimgym123'))
    p.add_argument('--minio-bucket',   default='fmus')
    p.add_argument('--no-redis',       action='store_true', help='Update catalog.yaml but skip Redis push')
    p.add_argument('--overwrite',      action='store_true', help='Overwrite existing catalog entry')
    return p


def main() -> None:
    args = build_parser().parse_args()

    fmu_path = os.path.abspath(args.fmu)
    if not os.path.isfile(fmu_path):
        logger.error(f"FMU file not found: {fmu_path}")
        sys.exit(1)

    catalog = load_catalog(CATALOG_YAML)

    if args.name in catalog['models'] and not args.overwrite:
        logger.error(
            f"Model '{args.name}' already exists in catalog.yaml. "
            "Use --overwrite to replace it."
        )
        sys.exit(1)

    # 1. Parse modelDescription.xml
    entry = parse_fmu(fmu_path)

    # 2. Override domain / tags from CLI
    if args.domain:
        entry['domain'] = args.domain
    if args.tags:
        entry['tags'] = list(set(entry['tags']) | set(args.tags))

    # 3. Resolve FMU source
    if args.local:
        entry['user_defined']['fmu_source'] = {
            'type': 'local',
            'path': fmu_path,
        }
        logger.info(f"Using local path: {fmu_path}")
    else:
        fmu_source = upload_to_minio(
            fmu_path=fmu_path,
            model_name=args.name,
            version=entry['version'],
            endpoint=args.minio_endpoint,
            access_key=args.minio_access_key,
            secret_key=args.minio_secret_key,
            bucket=args.minio_bucket,
        )
        entry['user_defined']['fmu_source'] = fmu_source

    # 4. Write to catalog.yaml
    catalog['models'][args.name] = entry
    save_catalog(CATALOG_YAML, catalog)

    # 5. Push to Redis
    if not args.no_redis:
        push_to_redis(catalog)

    logger.info(
        f"\n✓ FMU '{args.name}' registered successfully.\n"
        f"  Inputs:     {list(entry['inputs'].keys())}\n"
        f"  Outputs:    {list(entry['outputs'].keys())}\n"
        f"  Parameters: {list(entry['parameters'].keys())}\n"
        f"  Source:     {entry['user_defined']['fmu_source']['type']}"
    )


if __name__ == '__main__':
    main()
