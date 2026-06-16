"""
base_FMU_model.py

Wrapper that integrates any FMU (FMI 2.0 and FMI 3.0 co-simulation, with partial
FMI 1.0 support) into the CosimGym model framework.

At initialization the model resolves the FMU binary from one of three sources
declared in the catalog entry's ``user_defined.fmu_source`` block:

  type: "local"  → path on the local filesystem
  type: "minio"  → download from MinIO / S3-compatible store into local cache
  type: "http"   → download via HTTP into local cache

Downloaded FMUs are cached in ``~/.cosimgym/fmu_cache/<model_name>/<version>/``
and re-used on subsequent runs (no re-download unless the file is missing).

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
"""

import contextlib
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import requests

from .base_model import BaseModel
from fmpy import read_model_description, extract, dump
from fmpy.fmi1 import FMU1Slave
from fmpy.fmi2 import FMU2Slave
from fmpy.fmi3 import FMU3Slave


# Map FMI variable type strings to fmpy get/set method names for dispatch
_FMI_TYPE_GETSET = {
    # FMI 1.0 / 2.0
    'Real':        ('getReal',    'setReal'),
    'Integer':     ('getInteger', 'setInteger'),
    'Boolean':     ('getBoolean', 'setBoolean'),
    'String':      ('getString',  'setString'),
    'Enumeration': ('getInteger', 'setInteger'),
    # FMI 3.0 (Boolean/String/Enumeration share the FMI 1.0/2.0 entries above)
    'Float32':     ('getFloat32', 'setFloat32'),
    'Float64':     ('getFloat64', 'setFloat64'),
    'Int8':        ('getInt8',    'setInt8'),
    'UInt8':       ('getUInt8',   'setUInt8'),
    'Int16':       ('getInt16',   'setInt16'),
    'UInt16':      ('getUInt16',  'setUInt16'),
    'Int32':       ('getInt32',   'setInt32'),
    'UInt32':      ('getUInt32',  'setUInt32'),
    'Int64':       ('getInt64',   'setInt64'),
    'UInt64':      ('getUInt64',  'setUInt64'),
    'Binary':      ('getBinary',  'setBinary'),
}

# Map FMI type strings to catalog schema type strings
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


class BaseFMUModel(BaseModel):

    def __init__(self, name, metadata, config, logger):
        self.fmu = None
        self.model_description = None
        self.unzipdir = None
        self.fmiVersion = None

        # value-reference maps: var_name → (vref, fmi_type_str)
        self.vars = {}
        self.in_vars = {}
        self.ou_vars = {}
        self.params_vars = {}

        # Directory the FMU runtime runs in. EnergyPlus exports drop an
        # ``Output_EPExport_<instanceName>`` folder into the process CWD; we
        # point that at the scenario log dir instead of the workspace root.
        self._fmu_workdir = None

        super().__init__(name, metadata, config, logger)

    # ------------------------------------------------------------------
    # BaseModel abstract interface
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self.logger.debug(f"Initializing FMU model {self.name}")

        fmu_path = self._resolve_fmu_path()

        self._unpack_fmu(fmu_path)
        # EnergyPlus FMUs create their Output_EPExport_<instance> folder in the
        # CWD active during instantiate/doStep, so run the lifecycle from the
        # log dir to keep the workspace root clean.
        with self._in_fmu_workdir():
            self._instantiate_fmu()
            self._setup_experiment()
            self._enter_initialization_mode()
            self._push_initial_state_to_fmu()
            self._exit_initialization_mode()

        self.logger.info(f"FMU model {self.name} initialized (FMI {self.fmiVersion})")

    def step(self) -> None:
        self.logger.debug(f"Stepping FMU model {self.name} at ts={self.state.ts}")
        self._inputs_to_fmu()
        current_time = max(0, (self.state.ts - 1)) * self.real_period
        with self._in_fmu_workdir():
            self.fmu.doStep(
                currentCommunicationPoint=current_time,
                communicationStepSize=self.real_period,
            )
        self._outputs_from_fmu()

    def finalize(self) -> None:
        self.logger.info(f"Finalizing FMU model {self.name}")
        if self.fmu is not None:
            try:
                with self._in_fmu_workdir():
                    self.fmu.terminate()
                    self.fmu.freeInstance()
            except Exception as exc:
                self.logger.warning(f"FMU terminate/free raised: {exc}")
        if self.unzipdir and os.path.isdir(self.unzipdir):
            try:
                shutil.rmtree(self.unzipdir)
            except PermissionError as exc:
                self.logger.error(f"Could not remove unzip dir: {exc}")

    def reset(self) -> None:
        super().reset()
        self.logger.debug(f"Resetting FMU model {self.name} — re-running initialization")
        self.initialize()

    # ------------------------------------------------------------------
    # FMU working directory (keeps Output_EPExport_* out of workspace root)
    # ------------------------------------------------------------------

    def _resolve_fmu_workdir(self) -> Path:
        """Directory the FMU runtime should execute in.

        Derived from the federate logger's file handler so the EnergyPlus
        ``Output_EPExport_<instance>`` folder lands next to the scenario logs.
        Falls back to ``./logs`` when no file handler is found.
        """
        base = None

        # Primary: federate log path exported by federate_launcher. Layout is
        # .../<scenario>/<timestamp>/federates/<name>.log -> run dir is parent.parent.
        log_file = os.environ.get('COSIM_FEDERATE_LOG_FILE')
        if log_file:
            base = Path(log_file).resolve().parent.parent

        # Secondary: walk the logger chain for a FileHandler.
        if base is None:
            lg = self.logger
            while lg is not None and base is None:
                for handler in getattr(lg, 'handlers', []):
                    if isinstance(handler, logging.FileHandler):
                        base = Path(handler.baseFilename).resolve().parent.parent
                        break
                lg = getattr(lg, 'parent', None)

        # Fallback: ./logs
        if base is None:
            base = Path('logs').resolve()

        workdir = base / 'fmu_output'
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir

    @contextlib.contextmanager
    def _in_fmu_workdir(self):
        if self._fmu_workdir is None:
            self._fmu_workdir = self._resolve_fmu_workdir()
        prev_cwd = os.getcwd()
        os.chdir(self._fmu_workdir)
        try:
            yield
        finally:
            os.chdir(prev_cwd)

    # ------------------------------------------------------------------
    # FMU source resolution (local / MinIO / HTTP)
    # ------------------------------------------------------------------

    def _resolve_fmu_path(self) -> str:
        fmu_source = self.metadata.user_defined.get('fmu_source', {})
        src_type = fmu_source.get('type', 'local')

        if src_type == 'local':
            path = fmu_source.get('path', '')
            if not path:
                raise ValueError(
                    f"Model '{self.metadata.name}': fmu_source.type is 'local' "
                    "but fmu_source.path is empty. Set it in the catalog entry."
                )
            if not os.path.exists(path):
                raise FileNotFoundError(f"FMU not found at local path: {path}")
            return path

        elif src_type == 'minio':
            return self._download_from_minio(fmu_source)

        elif src_type == 'http':
            return self._download_from_http(fmu_source)

        else:
            raise ValueError(
                f"Unknown fmu_source.type '{src_type}'. "
                "Valid values: 'local', 'minio', 'http'."
            )

    def _cache_dir(self) -> Path:
        d = Path.home() / '.cosimgym' / 'fmu_cache' / self.metadata.name / self.metadata.version
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _download_from_minio(self, fmu_source: dict) -> str:
        try:
            from minio import Minio
        except ImportError:
            raise ImportError(
                "minio package not installed. Run: pip install minio>=7.0.0"
            )

        raw_endpoint = fmu_source.get('endpoint', 'http://localhost:9000')
        secure = raw_endpoint.startswith('https://')
        endpoint = raw_endpoint.replace('https://', '').replace('http://', '')
        bucket = fmu_source.get('bucket', 'fmus')
        object_key = fmu_source['object_key']
        access_key = fmu_source.get('access_key') or os.getenv('MINIO_ACCESS_KEY', 'cosimgym')
        secret_key = fmu_source.get('secret_key') or os.getenv('MINIO_SECRET_KEY', 'cosimgym123')

        local_path = self._cache_dir() / os.path.basename(object_key)
        if local_path.exists():
            self.logger.info(f"FMU cache hit: {local_path}")
            return str(local_path)

        self.logger.info(f"Downloading FMU from MinIO {endpoint}/{bucket}/{object_key}")
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        client.fget_object(bucket, object_key, str(local_path))
        self.logger.info(f"FMU downloaded to {local_path}")
        return str(local_path)

    def _download_from_http(self, fmu_source: dict) -> str:
        url = fmu_source['url']
        local_path = self._cache_dir() / url.split('/')[-1]
        if local_path.exists():
            self.logger.info(f"FMU cache hit: {local_path}")
            return str(local_path)

        self.logger.info(f"Downloading FMU from {url}")
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        self.logger.info(f"FMU downloaded to {local_path}")
        return str(local_path)

    # ------------------------------------------------------------------
    # FMU lifecycle helpers
    # ------------------------------------------------------------------

    def _unpack_fmu(self, fmu_path: str) -> None:
        self.model_description = read_model_description(fmu_path, validate=True)
        self._get_vars_from_fmu()
        self.unzipdir = extract(fmu_path)
        self.fmiVersion = self.model_description.fmiVersion
        self.logger.debug(f"FMU unpacked: {dump(fmu_path)}")

    def _get_vars_from_fmu(self) -> None:
        for v in self.model_description.modelVariables:
            vtype = v.type if v.type else 'Real'
            self.vars[v.name] = (v.valueReference, vtype, v.causality, v.variability)
            if v.causality == 'parameter':
                self.params_vars[v.name] = (v.valueReference, vtype)
            elif v.causality == 'input':
                self.in_vars[v.name] = (v.valueReference, vtype)
            elif v.causality == 'output':
                self.ou_vars[v.name] = (v.valueReference, vtype)

    def _instantiate_fmu(self) -> None:
        guid = self.model_description.guid
        model_id = self.model_description.coSimulation.modelIdentifier

        if self.fmiVersion == '1.0':
            self.fmu = FMU1Slave(
                guid=guid,
                unzipDirectory=self.unzipdir,
                modelIdentifier=model_id,
                instanceName=self.name,
            )
            self.fmu.instantiate(loggingOn=False)

        elif self.fmiVersion == '2.0':
            self.fmu = FMU2Slave(
                guid=guid,
                unzipDirectory=self.unzipdir,
                modelIdentifier=model_id,
                instanceName=self.name,
            )
            self.fmu.instantiate()

        elif self.fmiVersion == '3.0':
            self.fmu = FMU3Slave(
                guid=guid,
                unzipDirectory=self.unzipdir,
                modelIdentifier=model_id,
                instanceName=self.name,
            )
            self.fmu.instantiate()

        else:
            raise RuntimeError(f"Unsupported FMI version: {self.fmiVersion}")

    def _stop_time_seconds(self):
        """Total simulation horizon in seconds.

        Some co-sim FMUs (notably EnergyPlus exports) require a *defined* stop
        time: with stopTime=None fmpy sets stopTimeDefined=False and EnergyPlus
        clamps the stop time to 0, so the second doStep fails with fmi2Error.
        Derive it from the scenario start/end, falling back to time_stop*period.
        """
        try:
            start = datetime.fromisoformat(self.config.start_time)
            end = datetime.fromisoformat(self.config.end_time)
            return (end - start).total_seconds()
        except (TypeError, ValueError):
            pass
        if self.config.time_stop is not None and self.real_period is not None:
            return float(self.config.time_stop) * float(self.real_period)
        return None

    def _setup_experiment(self) -> None:
        if self.fmiVersion == '2.0':
            self.fmu.setupExperiment(startTime=0.0, stopTime=self._stop_time_seconds())
        # FMI 1.0 has no setupExperiment; initialization happens in _exit_initialization_mode
        # FMI 3.0 folds setupExperiment into enterInitializationMode(startTime, stopTime)

    def _enter_initialization_mode(self) -> None:
        if self.fmiVersion == '2.0':
            self.fmu.enterInitializationMode()
        elif self.fmiVersion == '3.0':
            self.fmu.enterInitializationMode(startTime=0.0, stopTime=self._stop_time_seconds())

    def _push_initial_state_to_fmu(self) -> None:
        for param_name, (vref, vtype) in self.params_vars.items():
            value = self.state.parameters.get(param_name)
            if value is not None:
                self._set_var(vref, vtype, value)

        for inp_name, (vref, vtype) in self.in_vars.items():
            value = self.state.inputs.get(inp_name)
            if value is not None:
                self._set_var(vref, vtype, value)

    def _exit_initialization_mode(self) -> None:
        if self.fmiVersion == '2.0':
            status = self.fmu.exitInitializationMode()
            if status != 0:
                raise RuntimeError(f"FMU exitInitializationMode returned status {status}")
        elif self.fmiVersion == '1.0':
            self.fmu.initialize(tStart=0.0, stopTime=None)
        elif self.fmiVersion == '3.0':
            self.fmu.exitInitializationMode()  # raises FMICallException on non-OK status

    # ------------------------------------------------------------------
    # Per-step I/O transfer
    # ------------------------------------------------------------------

    def _inputs_to_fmu(self) -> None:
        for var_name, (vref, vtype) in self.in_vars.items():
            value = self.state.inputs.get(var_name)
            if value is not None:
                self._set_var(vref, vtype, value)

    def _outputs_from_fmu(self) -> None:
        for var_name, (vref, vtype) in self.ou_vars.items():
            if var_name in self.state.outputs:
                self.state.outputs[var_name] = self._get_var(vref, vtype)

    # ------------------------------------------------------------------
    # Generic get / set via FMI type dispatch
    # ------------------------------------------------------------------

    def _set_var(self, vref: int, vtype: str, value) -> None:
        getter, setter = _FMI_TYPE_GETSET.get(vtype, ('getReal', 'setReal'))
        try:
            getattr(self.fmu, setter)([vref], [value])
        except Exception as exc:
            self.logger.error(f"setVar vref={vref} type={vtype} value={value}: {exc}")

    def _get_var(self, vref: int, vtype: str):
        getter, setter = _FMI_TYPE_GETSET.get(vtype, ('getReal', 'setReal'))
        try:
            return getattr(self.fmu, getter)([vref])[0]
        except Exception as exc:
            self.logger.error(f"getVar vref={vref} type={vtype}: {exc}")
            return None
