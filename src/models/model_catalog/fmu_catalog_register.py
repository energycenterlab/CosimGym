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
import re
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

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
    # FMI 1.0 / 2.0
    'Real':        'float',
    'Integer':     'int',
    'Boolean':     'bool',
    'String':      'string',
    'Enumeration': 'int',
    # FMI 3.0
    'Float32':     'float',
    'Float64':     'float',
    'Int8':        'int',
    'UInt8':       'int',
    'Int16':       'int',
    'UInt16':      'int',
    'Int32':       'int',
    'UInt32':      'int',
    'Int64':       'int',
    'UInt64':      'int',
    'Binary':      'string',
}


# ---------------------------------------------------------------------------
# Simulation horizon detection
#
# Some co-simulation FMUs cannot run for an unlimited span: an EnergyPlus export
# stops at the end of the RunPeriod it was built with. The catalog records that
# limit as ``max_sim_time`` (seconds) plus ``sim_start_date`` (the calendar date
# model-local time 0 corresponds to), and CosimGym restarts the model there
# instead of stepping it past the limit. Both are detected here so a user never
# has to work them out by hand.
# ---------------------------------------------------------------------------

def parse_idf_run_period(idf_path: str) -> Optional[dict]:
    """Read the RunPeriod out of an EnergyPlus input file.

    Returns ``{'max_sim_time': seconds, 'sim_start_date': 'MM-DD'}`` or None when
    the file has no usable RunPeriod. The end day is inclusive, as in EnergyPlus.
    """
    try:
        text = Path(idf_path).read_text(errors='replace')
    except OSError as exc:
        logger.warning(f"Could not read IDF {idf_path}: {exc}")
        return None

    match = re.search(r'^\s*RunPeriod\s*,(.*?);', text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if not match:
        logger.warning(f"No RunPeriod object found in {idf_path}")
        return None

    # Strip the '!- Field name' comments, then split the object into its fields.
    body = re.sub(r'!.*', '', match.group(1))
    fields = [f.strip() for f in body.split(',')]

    def field(idx):
        return fields[idx] if idx < len(fields) and fields[idx] else None

    try:
        begin_month = int(field(1)); begin_day = int(field(2))
        end_month   = int(field(4)); end_day   = int(field(5))
    except (TypeError, ValueError):
        logger.warning(f"RunPeriod in {idf_path} has no usable begin/end dates")
        return None

    # Years are usually blank, meaning a generic (non-leap) weather year.
    begin_year = int(field(3)) if field(3) else 2001
    end_year   = int(field(6)) if field(6) else begin_year
    if end_year < begin_year or (end_year == begin_year and
                                 (end_month, end_day) < (begin_month, begin_day)):
        end_year = begin_year + 1  # run period wraps across new year

    try:
        begin = date(begin_year, begin_month, begin_day)
        end = date(end_year, end_month, end_day)
    except ValueError as exc:
        logger.warning(f"RunPeriod in {idf_path} has an invalid date: {exc}")
        return None

    span_days = (end - begin).days + 1  # EnergyPlus simulates the end day too
    return {
        'max_sim_time': float(span_days * 86400),
        'sim_start_date': f"{begin_month:02d}-{begin_day:02d}",
    }


def detect_horizon_from_fmu(md) -> Optional[dict]:
    """Derive the horizon from the FMU's own DefaultExperiment, when it has one."""
    exp = getattr(md, 'defaultExperiment', None)
    if exp is None or getattr(exp, 'stopTime', None) is None:
        return None
    start = float(getattr(exp, 'startTime', None) or 0.0)
    span = float(exp.stopTime) - start
    if span <= 0:
        return None
    return {'max_sim_time': span, 'sim_start_date': None}


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

    # Can the slave save and restore its state? If it can, a rolling reset can
    # rewind it instantly instead of restarting and replaying from the beginning.
    cosim = getattr(md, 'coSimulation', None)
    supports_rollback = bool(getattr(cosim, 'canGetAndSetFMUstate', False)) if cosim else False

    horizon = detect_horizon_from_fmu(md)

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
        # None => no intrinsic limit: the model runs for the whole scenario.
        'max_sim_time':   horizon['max_sim_time'] if horizon else None,
        'sim_start_date': horizon['sim_start_date'] if horizon else None,
        'user_defined': {
            'fmu_source': {},  # filled by caller
            'fmu_reset': {
                'supports_rollback': supports_rollback,
            },
        },
        'parameters': parameters,
        'inputs':      inputs,
        'outputs':     outputs,
    }

    logger.info(
        f"Parsed FMU '{model_name_fmu}' (FMI {fmi_version}): "
        f"{len(inputs)} inputs, {len(outputs)} outputs, {len(parameters)} parameters"
    )
    if horizon:
        logger.info(f"  DefaultExperiment gives a {horizon['max_sim_time']} s simulation horizon")
    logger.info(f"  State save/restore (rollback) supported: {supports_rollback}")
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
    p.add_argument('--idf',            default=None,
                   help='EnergyPlus input file this FMU was exported from. Its RunPeriod gives '
                        'the simulation horizon and start date, so the model is restarted at the '
                        'right point instead of failing when it runs past its run period.')
    p.add_argument('--max-sim-time',   type=float, default=None,
                   help='Longest span of simulated time (seconds) this FMU can run before it must '
                        'be restarted. Overrides anything detected from the FMU or the IDF.')
    p.add_argument('--sim-start-date', default=None,
                   help="Calendar date model-local time 0 means, as MM-DD or YYYY-MM-DD. "
                        "This is where the model restarts from when it reaches its horizon.")
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

    # 3. Resolve the simulation horizon: explicit flags win, then the IDF's
    #    RunPeriod, then whatever the FMU declared in its DefaultExperiment.
    if args.idf:
        from_idf = parse_idf_run_period(args.idf)
        if from_idf:
            entry['max_sim_time'] = from_idf['max_sim_time']
            entry['sim_start_date'] = from_idf['sim_start_date']
            logger.info(
                f"RunPeriod from {args.idf}: starts {from_idf['sim_start_date']}, "
                f"runs {from_idf['max_sim_time']} s "
                f"({from_idf['max_sim_time'] / 86400:.0f} days)"
            )
    if args.max_sim_time is not None:
        entry['max_sim_time'] = args.max_sim_time
    if args.sim_start_date is not None:
        entry['sim_start_date'] = args.sim_start_date

    if not entry.get('max_sim_time'):
        logger.warning(
            "No simulation horizon detected: this model is registered as unbounded and "
            "will be stepped for the whole scenario. If the FMU cannot run that long "
            "(an EnergyPlus export stops at the end of its RunPeriod), re-register it "
            "with --idf or --max-sim-time, or the run will fail when the limit is reached."
        )

    # 4. Resolve FMU source
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

    # 5. Write to catalog.yaml
    catalog['models'][args.name] = entry
    save_catalog(CATALOG_YAML, catalog)

    # 6. Push to Redis
    if not args.no_redis:
        push_to_redis(catalog)

    logger.info(
        f"\n✓ FMU '{args.name}' registered successfully.\n"
        f"  Inputs:     {list(entry['inputs'].keys())}\n"
        f"  Outputs:    {list(entry['outputs'].keys())}\n"
        f"  Parameters: {list(entry['parameters'].keys())}\n"
        f"  Source:     {entry['user_defined']['fmu_source']['type']}\n"
        f"  Horizon:    {entry['max_sim_time'] or 'unbounded'}"
        f"{' s from ' + entry['sim_start_date'] if entry.get('sim_start_date') else ''}"
    )


if __name__ == '__main__':
    main()
