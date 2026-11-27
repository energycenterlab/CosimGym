"""
eplus_discover.py

What can an EnergyPlus IDF be co-simulated through? Lists everything the EnergyPlus Python API exposes for it
(actuators = possible inputs, output variables, ...) and, with ``--io``, shows what the ``*`` / ``?`` key patterns of
an ``eplus_io`` block expand to. The model needs no registration: you write ``eplus_io`` in the scenario
(see docs/user_guide/eplus_models.md); this tool only helps you choose the names. Run from the repo root, ``cosim_gym`` env.
Slow for big buildings (sizing + warmup), once per IDF: the listing is cached.

  python src/models/model_catalog/eplus_discover.py --idf B.idf --epw W.epw [--out listing.csv] [--io io.yaml]
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # src/
import yaml  # noqa: E402

from models.base_eplus_model import discover_listing  # noqa: E402
from models.eplus_idf import Idf, expand_io  # noqa: E402
from models.eplus_registry import resolve, upgrade_idf  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--idf', required=True)
    ap.add_argument('--epw', required=True)
    ap.add_argument('--version', default=None, help='force an EnergyPlus major.minor')
    ap.add_argument('--out', default=None, help='listing file (default <idf>.eplus_api.csv)')
    ap.add_argument('--io', default=None, help='YAML with an eplus_io block: print what its key patterns expand to')
    a = ap.parse_args()

    idf_path, epw = Path(a.idf).expanduser().resolve(), Path(a.epw).expanduser().resolve()
    inst = resolve(Idf.parse(idf_path).version(), a.version)
    path, _ = upgrade_idf(idf_path, inst)
    idf = Idf.parse(path)
    ext = idf.convert_external_interface()
    with tempfile.TemporaryDirectory(prefix='eplus_discover_') as tmp:
        listing = discover_listing(idf, inst, epw, Path(tmp))
    out = Path(a.out) if a.out else idf_path.with_suffix('.eplus_api.csv')
    out.write_text(listing)
    n = {t: sum(1 for l in listing.splitlines() if l.startswith(t + ',')) for t in ('Actuator', 'OutputVariable', 'Meter')}
    print(f'listing -> {out}  {n}\nzones: {idf.zones()[:10]}{"..." if len(idf.zones()) > 10 else ""}')
    print(f'ExternalInterface I/O (use eplus_io.from_external_interface): {[i["alias"] for i in ext.inputs]} -> '
          f'{[o["alias"] for o in ext.outputs]}')
    if a.io:
        io = expand_io(yaml.safe_load(Path(a.io).read_text()), listing, idf.zones())
        print('eplus_io expands to:\n' + yaml.safe_dump(io, sort_keys=False))


if __name__ == '__main__':
    main()
