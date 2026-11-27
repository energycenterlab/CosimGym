"""
eplus_registry.py

Which EnergyPlus installs exist on *this* machine, which one an IDF should run on, and
how to bring an older IDF up to that install's version (EnergyPlus's own Transition
binaries; downgrading is impossible, so the IDF version picks the lowest install that is
not older than it).

Discovery (first hit per version wins, nothing is hardcoded to a user home):
  1. ``COSIMGYM_ENERGYPLUS_DIRS``: ``os.pathsep``-separated paths, each an install dir
     or a folder of installs;
  2. ``~/.cosimgym/energyplus/<anything>/``;
  3. the stock locations ``/usr/local/EnergyPlus-*``, ``/opt/EnergyPlus-*``,
     ``/Applications/EnergyPlus-*``.
An install is a directory holding an ``energyplus`` executable and ``pyenergyplus/``.

Upgraded IDFs are cached in ``$COSIMGYM_EPLUS_CACHE`` (default ``~/.cosimgym/eplus_cache``)
keyed by IDF content hash + target version; the original is never touched.
Every machine (remote federate hosts included) runs its own discovery.
"""
import fcntl
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from .eplus_idf import Idf, IdfError, content_hash, parse_version

logger = logging.getLogger(__name__)
Version = Tuple[int, int, int]

_TRANSITION = re.compile(r'^Transition-V(\d+-\d+-\d+)-to-V(\d+-\d+-\d+)$')


class EnergyPlusNotFound(RuntimeError):
    pass


@dataclass(frozen=True)
class Install:
    path: Path
    version: Version

    @property
    def label(self) -> str:
        return '.'.join(map(str, self.version))

    @property
    def exe(self) -> Path:
        return self.path / 'energyplus'

    @property
    def updater_dir(self) -> Path:
        return self.path / 'PreProcess' / 'IDFVersionUpdater'


def _is_install(p: Path) -> bool:
    return (p / 'energyplus').exists() and (p / 'pyenergyplus').is_dir()


@lru_cache(maxsize=None)
def _install_version(path: str) -> Optional[Version]:
    try:
        out = subprocess.run([str(Path(path) / 'energyplus'), '--version'], capture_output=True,
                             text=True, timeout=30).stdout
        return parse_version(out.split('Version')[1].strip())
    except Exception as exc:   # not runnable here (wrong OS / glibc): not an install for us
        logger.warning(f'EnergyPlus at {path} is not runnable: {exc}')
        return None


def _candidates() -> List[Path]:
    roots = [Path(p).expanduser() for p in os.environ.get('COSIMGYM_ENERGYPLUS_DIRS', '').split(os.pathsep) if p]
    roots.append(Path('~/.cosimgym/energyplus').expanduser())
    out = []
    for r in roots:
        if _is_install(r):
            out.append(r)
        elif r.is_dir():
            out += sorted(c for c in r.iterdir() if c.is_dir() and _is_install(c))
    for pat in ('/usr/local/EnergyPlus-*', '/opt/EnergyPlus-*', '/Applications/EnergyPlus-*'):
        out += sorted(Path('/').glob(pat.lstrip('/')))
    return out


def find_installs() -> List[Install]:
    """Runnable installs on this machine, oldest first, one per major.minor."""
    found = {}
    for p in _candidates():
        v = _install_version(str(p))
        if v and v[:2] not in found:
            found[v[:2]] = Install(p, v)
    return sorted(found.values(), key=lambda i: i.version)


def resolve(idf_version: Optional[Version], version: Optional[str] = None,
            install_dir: Optional[str] = None) -> Install:
    """Pick the install for an IDF. ``install_dir`` / ``version`` force one (user override)."""
    if install_dir:
        p = Path(install_dir).expanduser()
        v = _install_version(str(p)) if _is_install(p) else None
        if not v:
            raise EnergyPlusNotFound(f'eplus_dir {p} is not a runnable EnergyPlus install')
        return Install(p, v)
    installs = find_installs()
    if version:
        want = parse_version(str(version))[:2]
        hit = [i for i in installs if i.version[:2] == want]
        if not hit:
            raise EnergyPlusNotFound(f'EnergyPlus {version} requested but not installed; found: {_have(installs)}')
        return hit[0]
    if idf_version is None:
        raise IdfError('IDF has no Version object: set eplus.version (or eplus.eplus_dir) explicitly')
    exact = [i for i in installs if i.version[:2] == idf_version[:2]]
    if exact:
        return exact[0]
    newer = [i for i in installs if i.version[:2] > idf_version[:2]]
    if newer:
        return newer[0]      # lowest newer (list is sorted): upgrade the IDF to it
    raise EnergyPlusNotFound(
        f'IDF is version {idf_version[0]}.{idf_version[1]} and no EnergyPlus >= that is runnable here '
        f'(found: {_have(installs)}). Install a newer EnergyPlus under ~/.cosimgym/energyplus/ or '
        f'list it in COSIMGYM_ENERGYPLUS_DIRS. A newer IDF cannot be downgraded.')


def _have(installs) -> str:
    return ', '.join(i.label for i in installs) or 'none'


def cache_root() -> Path:
    return Path(os.environ.get('COSIMGYM_EPLUS_CACHE', '~/.cosimgym/eplus_cache')).expanduser()


def upgrade_idf(idf_path: Path, install: Install) -> Tuple[Path, List[str]]:
    """Return ``(idf path runnable on install, log lines)``; the input itself when no upgrade is needed.

    Upgrade = chain of the install's ``Transition-Vx-to-Vy`` binaries, run from their own
    directory (they look up the IDDs there). Cached; safe against concurrent callers.
    """
    idf_path = Path(idf_path)
    src = Idf.parse(idf_path).version()
    if src is None or src[:2] == install.version[:2]:
        return idf_path, []
    if src[:2] > install.version[:2]:
        raise IdfError(f'{idf_path.name} is version {src[0]}.{src[1]}, newer than EnergyPlus {install.label}')

    steps = {}
    for f in install.updater_dir.iterdir():
        m = _TRANSITION.match(f.name)
        if m:
            steps[parse_version(m.group(1))[:2]] = (f, parse_version(m.group(2))[:2])
    chain, cur = [], src[:2]
    while cur != install.version[:2]:
        if cur not in steps:
            raise IdfError(f'EnergyPlus {install.label} has no transition from {cur[0]}.{cur[1]}; '
                           f'IDF {idf_path.name} is too old for it')
        chain.append(steps[cur][0]); cur = steps[cur][1]

    out_dir = cache_root() / content_hash(idf_path) / install.label
    out = out_dir / 'upgraded.idf'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if out.exists():
            return out, (out_dir / 'upgrade.log').read_text().splitlines()
        log = [f'upgrade {idf_path.name}: {src[0]}.{src[1]} -> {install.label} via {len(chain)} transition(s)']
        with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
            work = Path(tmp) / 'step.idf'
            shutil.copyfile(idf_path, work)
            for exe in chain:
                r = subprocess.run([str(exe), str(work)], cwd=install.updater_dir, capture_output=True,
                                   text=True, timeout=600)
                new = work.with_suffix('.idfnew')
                if not new.exists():
                    raise IdfError(f'{exe.name} failed on {idf_path.name}:\n{r.stdout[-500:]}{r.stderr[-300:]}')
                new.replace(work)
                tail = [l.strip() for l in r.stdout.splitlines() if 'Conversion' in l or 'Warning' in l]
                log.append(f'{exe.name}: ' + (tail[-1] if tail else 'done'))
                for junk in Path(tmp).glob('step.*'):
                    if junk != work:
                        junk.unlink()
            reached = Idf.parse(work).version()
            if reached is None or reached[:2] != install.version[:2]:
                raise IdfError(f'upgrade of {idf_path.name} ended at {reached}, expected {install.label}')
            work.replace(out)
        (out_dir / 'upgrade.log').write_text('\n'.join(log))
    return out, log
