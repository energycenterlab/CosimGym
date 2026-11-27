"""
eplus_idf.py

Pure-Python (no EnergyPlus needed) utilities for EnergyPlus ``.idf`` text files:
parse into objects, read the facts a co-simulation model needs (version, Timestep,
RunPeriod, zones, autosize, ExternalInterface I/O), and edit a *copy* (RunPeriod,
SimulationControl, strip / convert ``ExternalInterface*``, extra ``Output:Meter``).

Generic by construction: it knows EnergyPlus object *classes* and their documented
field order (IDF layout of versions >= 9.0), never a particular building. Untouched
objects are written back byte for byte (comments included); only edited objects are
regenerated. ``.imf`` (EPMacro) and ``.epJSON`` inputs are not supported.
"""
import fnmatch
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

_TOKEN = re.compile(r'![^\n]*|;')
_COMMENT = re.compile(r'![^\n]*')
_WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Field positions after the class name, IDF layout of EnergyPlus >= 9.0.
_RP_BEGIN_M, _RP_BEGIN_D, _RP_BEGIN_Y, _RP_END_M, _RP_END_D, _RP_END_Y, _RP_DOW = 1, 2, 3, 4, 5, 6, 7
_SC_SIZING_PERIODS, _SC_WEATHER_PERIODS = 3, 4   # SimulationControl fields


class IdfError(ValueError):
    """The IDF cannot be handled by these utilities (message says why)."""


@dataclass
class IdfObject:
    cls: str
    fields: List[str]
    raw: Optional[str] = None      # original text (incl. leading comments); None once regenerated

    def text(self) -> str:
        if self.raw is not None:
            return self.raw
        body = ',\n'.join(f'    {f}' for f in self.fields)
        return f'\n{self.cls},\n{body};\n' if self.fields else f'\n{self.cls};\n'

    def set_field(self, idx: int, value) -> None:
        while len(self.fields) <= idx:
            self.fields.append('')
        self.fields[idx] = '' if value is None else str(value)
        self.raw = None


@dataclass
class RunPeriod:
    """First RunPeriod of the file, dates as written (year ``None`` when blank)."""
    obj: IdfObject
    begin: date
    end: date            # inclusive, as in EnergyPlus
    begin_year_given: bool
    start_dow: Optional[str]   # weekday name, or None when blank / UseWeatherFile

    @property
    def days(self) -> int:
        return (self.end - self.begin).days + 1


@dataclass
class ExternalInterfaceIO:
    """I/O an FMU-export IDF already declares (names are what the FMU exposed)."""
    inputs: List[dict] = field(default_factory=list)    # {alias, kind: schedule|actuator|global, ...}
    outputs: List[dict] = field(default_factory=list)   # {alias, key, variable}


class Idf:
    def __init__(self, objects: List[IdfObject], tail: str = ''):
        self.objects = objects
        self.tail = tail

    # ---- parse / write -------------------------------------------------
    @classmethod
    def parse(cls, source) -> 'Idf':
        text = Path(source).read_text(errors='replace') if not isinstance(source, str) or os.path.exists(source) else source
        if re.search(r'^\s*##\s*(def|include|if|set|fileprefix)', text, re.MULTILINE | re.IGNORECASE):
            raise IdfError('EPMacro directives (##def / ##include ...) found: expand the file with EPMacro first')
        objs, start = [], 0
        for m in _TOKEN.finditer(text):
            if m.group(0) == ';':
                raw = text[start:m.end()]
                body = _COMMENT.sub('', raw).rstrip().rstrip(';')
                parts = [p.strip() for p in body.split(',')]
                objs.append(IdfObject(parts[0], parts[1:], raw))
                start = m.end()
        return cls(objs, text[start:])

    def text(self) -> str:
        return ''.join(o.text() for o in self.objects) + self.tail

    def write(self, path) -> None:
        """Atomic write (temp file + rename): concurrent readers never see a partial IDF."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
        with os.fdopen(fd, 'w') as fh:
            fh.write(self.text())
        os.replace(tmp, path)

    # ---- queries -------------------------------------------------------
    def find(self, cls: str) -> List[IdfObject]:
        c = cls.lower()
        return [o for o in self.objects if o.cls.lower() == c]

    def find_prefix(self, prefix: str) -> List[IdfObject]:
        p = prefix.lower()
        return [o for o in self.objects if o.cls.lower().startswith(p)]

    def version(self) -> Optional[Tuple[int, int, int]]:
        v = self.find('Version')
        if not v or not v[0].fields or not v[0].fields[0]:
            return None
        return parse_version(v[0].fields[0])

    def timestep_per_hour(self) -> int:
        """``Timestep`` object, EnergyPlus default (6) when absent."""
        t = self.find('Timestep')
        return int(float(t[0].fields[0])) if t and t[0].fields and t[0].fields[0] else 6

    def zone_step_s(self) -> float:
        return 3600.0 / self.timestep_per_hour()

    def zones(self) -> List[str]:
        return [o.fields[0] for o in self.find('Zone') if o.fields]

    def has_autosize(self) -> bool:
        return any(f.lower() == 'autosize' for o in self.objects for f in o.fields)

    def run_period(self) -> Optional[RunPeriod]:
        rps = self.find('RunPeriod')
        if not rps:
            return None
        o = rps[0]
        f = lambda i: o.fields[i] if i < len(o.fields) and o.fields[i] else None
        try:
            bm, bd, em, ed = int(f(_RP_BEGIN_M)), int(f(_RP_BEGIN_D)), int(f(_RP_END_M)), int(f(_RP_END_D))
        except (TypeError, ValueError):
            raise IdfError('RunPeriod has no usable begin/end month and day')
        by = int(f(_RP_BEGIN_Y)) if f(_RP_BEGIN_Y) else 2001   # blank year = generic non-leap year
        ey = int(f(_RP_END_Y)) if f(_RP_END_Y) else by
        if (ey, em, ed) < (by, bm, bd):
            ey = by + 1                                        # wraps across new year
        dow = f(_RP_DOW)
        dow = dow.capitalize() if dow and dow.capitalize() in _WEEKDAYS else None
        return RunPeriod(o, date(by, bm, bd), date(ey, em, ed), bool(f(_RP_BEGIN_Y)), dow)

    def external_interface_io(self) -> ExternalInterfaceIO:
        """I/O declared by ``ExternalInterface*`` objects (FMU export / BCVTB), by FMU variable name."""
        io = ExternalInterfaceIO()
        for o in self.find_prefix('ExternalInterface:'):
            c, f = o.cls.lower(), o.fields
            g = lambda i: f[i] if i < len(f) else ''
            if c.endswith(':schedule'):          # (Name, TypeLimits, [FMU var,] Initial)
                fmu = c.startswith('externalinterface:functionalmockupunitexport')
                io.inputs.append({'alias': g(2) if fmu else g(0), 'kind': 'schedule', 'name': g(0),
                                  'default': _num(g(3) if fmu else g(2))})
            elif c.endswith(':actuator'):        # (Name, component name, type, control, [FMU var,] Initial)
                fmu = c.startswith('externalinterface:functionalmockupunitexport')
                io.inputs.append({'alias': g(4) if fmu else g(0), 'kind': 'actuator', 'component_type': g(2),
                                  'control_type': g(3), 'key': g(1), 'default': _num(g(5) if fmu else g(4))})
            elif c.endswith(':variable') and 'from' not in c:   # To:Variable / Variable = EMS global
                fmu = c.startswith('externalinterface:functionalmockupunitexport')
                io.inputs.append({'alias': g(1) if fmu else g(0), 'kind': 'global', 'name': g(0),
                                  'default': _num(g(2) if fmu else g(1))})
            elif c.endswith(':from:variable'):   # (Output:Variable key, variable name, FMU var)
                io.outputs.append({'alias': g(2), 'key': g(0), 'variable': g(1)})
        return io

    # ---- edits (on a copy; callers write the result elsewhere) -----------
    def remove(self, pred) -> int:
        n = len(self.objects)
        self.objects = [o for o in self.objects if not pred(o)]
        return n - len(self.objects)

    def add(self, cls: str, fields: List[str]) -> None:
        self.objects.append(IdfObject(cls, [str(x) for x in fields]))

    def convert_external_interface(self) -> ExternalInterfaceIO:
        """Make an FMU-export / BCVTB IDF runnable stand-alone through the Python API.

        ``...:Schedule`` becomes ``Schedule:Constant`` (same name, same limits, the initial
        value), ``...:Variable`` an EMS ``GlobalVariable``; ``...:Actuator`` and
        ``From:Variable`` need nothing (the component is already an actuator / variable);
        then every ``ExternalInterface*`` object is dropped. Returns the I/O they declared.
        """
        io = self.external_interface_io()
        for o in self.find_prefix('ExternalInterface:'):
            c = o.cls.lower()
            if c.endswith(':schedule'):
                fmu = c.startswith('externalinterface:functionalmockupunitexport')
                init = o.fields[3] if fmu and len(o.fields) > 3 else (o.fields[2] if not fmu and len(o.fields) > 2 else '0')
                self.add('Schedule:Constant', [o.fields[0], o.fields[1] if len(o.fields) > 1 else '', init or '0'])
            elif c.endswith(':variable') and 'from' not in c:
                self.add('EnergyManagementSystem:GlobalVariable', [o.fields[0]])
        self.remove(lambda o: o.cls.lower().startswith('externalinterface'))
        return io

    def set_run_period(self, begin: date, end: date, start_dow: Optional[str] = None) -> None:
        """Replace every RunPeriod by one covering ``begin``..``end`` (inclusive), keeping all
        other fields of the first one. Years are written only if the original had one."""
        rp = self.run_period()
        if rp is None:
            raise IdfError('no RunPeriod to rewrite')
        o = rp.obj
        o.set_field(_RP_BEGIN_M, begin.month); o.set_field(_RP_BEGIN_D, begin.day)
        o.set_field(_RP_END_M, end.month); o.set_field(_RP_END_D, end.day)
        if rp.begin_year_given:
            o.set_field(_RP_BEGIN_Y, begin.year); o.set_field(_RP_END_Y, end.year)
        if start_dow:
            o.set_field(_RP_DOW, start_dow)
        first = o
        self.objects = [x for x in self.objects if x is first or x.cls.lower() != 'runperiod']

    def force_weather_run(self, skip_sizing: bool = False) -> bool:
        """``SimulationControl``: make sure the weather-file run period is simulated; optionally
        skip the sizing-period runs (only if nothing is autosized). Returns True if sizing was skipped."""
        sc = self.find('SimulationControl')
        if not sc:
            return False
        sc[0].set_field(_SC_WEATHER_PERIODS, 'Yes')
        if skip_sizing and not self.has_autosize():
            for i in (0, 1, 2, _SC_SIZING_PERIODS):
                sc[0].set_field(i, 'No')
            return True
        return False

    def ensure_meter(self, name: str) -> None:
        if not any(o.cls.lower() == 'output:meter' and o.fields and o.fields[0].lower() == name.lower()
                   for o in self.objects):
            self.add('Output:Meter', [name, 'Timestep'])


def parse_version(s: str) -> Tuple[int, int, int]:
    """'23.1', '23.1.0', 'V23-1-0', '23.1.0-87ed9199d4' -> (23, 1, 0)."""
    nums = [int(x) for x in re.findall(r'\d+', re.split(r'-[0-9a-f]{8,}', s.strip().lstrip('Vv'))[0])[:3]]
    return tuple(nums + [0] * (3 - len(nums)))


def weekday_after(dow: Optional[str], days: int) -> Optional[str]:
    return None if dow is None else _WEEKDAYS[(_WEEKDAYS.index(dow) + days) % 7]


def content_hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(Path(p).read_bytes() if isinstance(p, Path) else str(p).encode())
    return h.hexdigest()[:16]


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


_SCHEDULES = ('Schedule:Constant', 'Schedule:Compact', 'Schedule:Year', 'Schedule:File')


def _schedule_actuator(idf: 'Idf', name: str) -> dict:
    """``{schedule: NAME}`` -> the actuator EnergyPlus exposes for that IDF schedule (no IDF change needed)."""
    names = []
    for cls in _SCHEDULES:
        for o in idf.find(cls):
            if o.fields and o.fields[0].lower() == str(name).lower():
                return {'component_type': o.cls, 'control_type': 'Schedule Value', 'key': o.fields[0]}
            names.append(o.fields[0] if o.fields else '?')
    raise IdfError(f"eplus_io: schedule {name!r} not found among the IDF's Schedule:Constant/Compact/Year/File "
                   f"objects (found: {names[:15]}{'...' if len(names) > 15 else ''})")


def listing_keys(listing: str, kind: str, a: str, b: Optional[str] = None) -> List[str]:
    """Keys an ``list_available_api_data_csv`` listing offers for an actuator (component_type, control_type) or a variable (name)."""
    out = []
    for line in listing.splitlines():
        p = [x.strip() for x in line.split(',')]
        if kind == 'actuator' and p[0] == 'Actuator' and len(p) >= 4 and p[1].lower() == a.lower() and p[2].lower() == b.lower():
            out.append(p[3])
        if kind == 'variable' and p[0] == 'OutputVariable' and len(p) >= 3 and p[1].lower() == a.lower():
            out.append(p[2])
    return out


def has_pattern(io: dict) -> bool:
    """True when an actuator / variable ``key`` in ``eplus_io`` holds ``*`` or ``?`` (needs the API listing to expand)."""
    return any(re.search(r'[*?]', str((d.get('actuator') or d.get('variable') or {}).get('key', '')))
               for sect in ('inputs', 'outputs') for d in (io.get(sect) or {}).values())


def expand_io(io: dict, listing: str, zones) -> dict:
    """Expand ``*`` / ``?`` in the ``key`` of actuator / variable entries (fnmatch over the keys the listing offers,
    else over the IDF zones). Each hit becomes ``<name>_<key>`` (non-alphanumerics -> ``_``); other entries pass through."""
    res = dict(io, inputs={}, outputs={})
    for sect in ('inputs', 'outputs'):
        for name, d in (io.get(sect) or {}).items():
            kind = 'actuator' if 'actuator' in d else 'variable' if 'variable' in d else None
            leaf = d.get(kind) if kind else None
            if not leaf or not re.search(r'[*?]', str(leaf.get('key', ''))):
                res[sect][name] = d
                continue
            cands = (listing_keys(listing, 'actuator', leaf['component_type'], leaf['control_type']) if kind == 'actuator'
                     else listing_keys(listing, 'variable', leaf['name'])) or list(zones)
            hits = [k for k in cands if fnmatch.fnmatchcase(k.lower(), leaf['key'].lower())]
            if not hits:
                raise IdfError(f"eplus_io {sect[:-1]} {name!r}: key pattern {leaf['key']!r} matches nothing "
                               f"(candidates: {cands[:8]})")
            for k in hits:
                new = f"{name}_{re.sub(r'[^A-Za-z0-9]+', '_', k).strip('_')}"
                if new in res[sect] or new in io[sect]:
                    raise IdfError(f"eplus_io {sect[:-1]} {name!r}: key {k!r} expands to {new!r}, which is already "
                                   f"declared or produced by another key: use a more specific pattern or another name")
                res[sect][new] = {**d, kind: {**leaf, 'key': k}}
    return res


def normalize_io(spec: dict, ext: Optional[ExternalInterfaceIO] = None, idf: Optional['Idf'] = None):
    """Declarative ``eplus_io`` (+ optionally what an FMU-export IDF declared) -> the worker's I/O dicts.

    Returns ``(inputs, outputs)``: ``{name: {kind, ...}}``. Kinds: input ``actuator`` / ``global``;
    output ``variable`` / ``meter`` / ``actuator`` / ``global``. Explicit entries win over ``ext``.
    ``{schedule: NAME}`` (input) needs ``idf`` and becomes the schedule's actuator.
    """
    ins, outs = {}, {}
    if ext:
        for i in ext.inputs:
            if i['kind'] == 'global':
                ins[i['alias']] = {'kind': 'global', 'name': i['name'], 'default': i['default']}
            elif i['kind'] == 'schedule':
                ins[i['alias']] = {'kind': 'actuator', 'component_type': 'Schedule:Constant',
                                   'control_type': 'Schedule Value', 'key': i['name'], 'default': i['default']}
            else:
                ins[i['alias']] = {k: i[k] for k in ('component_type', 'control_type', 'key', 'default')} | {'kind': 'actuator'}
        for o in ext.outputs:
            outs[o['alias']] = {'kind': 'variable', 'name': o['variable'], 'key': o['key'], 'agg': 'last'}
    for name, d in (spec.get('inputs') or {}).items():
        d = dict(d)
        if 'schedule' in d:
            if idf is None:
                raise IdfError(f"eplus_io input {name!r}: 'schedule' needs the IDF")
            ins[name] = {'kind': 'actuator', **_schedule_actuator(idf, d['schedule']), 'default': d.get('default')}
        elif 'actuator' in d:
            ins[name] = {'kind': 'actuator', **d['actuator'], 'default': d.get('default')}
        elif 'global' in d:
            ins[name] = {'kind': 'global', 'name': d['global'], 'default': d.get('default')}
        else:
            raise IdfError(f"eplus_io input {name!r}: need 'actuator', 'schedule' or 'global'")
    for name, d in (spec.get('outputs') or {}).items():
        agg = d.get('agg', 'last')
        if agg not in ('last', 'mean', 'sum', 'min', 'max'):
            raise IdfError(f"eplus_io output {name!r}: unknown agg {agg!r}")
        if 'variable' in d:
            outs[name] = {'kind': 'variable', **d['variable'], 'agg': agg}
        elif 'meter' in d:
            outs[name] = {'kind': 'meter', 'name': d['meter'], 'agg': agg}
        elif 'actuator' in d:
            outs[name] = {'kind': 'actuator', **d['actuator'], 'agg': agg}
        elif 'global' in d:
            outs[name] = {'kind': 'global', 'name': d['global'], 'agg': agg}
        else:
            raise IdfError(f"eplus_io output {name!r}: need variable / meter / actuator / global")
    return ins, outs
