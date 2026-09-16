"""
make_daily_shading_fmu.py — build a BUI0 variant whose shading is recomputed daily.

EnergyPlus recomputes shading on a *periodic* schedule anchored to the beginning of
the run period (BUI0: `ShadowCalculation, PolygonClipping, Periodic, 20`). Moving
the begin date therefore moves the days on which shading is recomputed, which is a
difference between a shifted restart and a continuous run that has nothing to do
with the building's thermal state and never decays.

This writes a copy of the FMU with the update frequency set to 1 day, so the
validation can be re-run against it and the two error figures compared: what is
left is the state error, what disappears was the shading phase.

    python scripts/fmu_warmstart_validation/make_daily_shading_fmu.py <out.fmu>
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src/models/model_catalog/physical_models/resources/BUI0.fmu'

SHADOW_RE = re.compile(r'(^[ \t]*ShadowCalculation[ \t]*,)(.*?);', re.MULTILINE | re.DOTALL | re.IGNORECASE)


def daily_shading(idf_text: str) -> str:
    match = SHADOW_RE.search(idf_text)
    if not match:
        raise SystemExit('no ShadowCalculation object in the IDF')
    body = match.group(2)
    # third field (0-based 2) is the update frequency in days
    fields = body.split(',')
    if len(fields) < 3:
        raise SystemExit('unexpected ShadowCalculation layout')
    fields[2] = re.sub(r'(\s*)\d+', r'\g<1>1', fields[2], count=1)
    return idf_text[:match.start()] + match.group(1) + ','.join(fields) + ';' + idf_text[match.end():]


def main():
    out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'BUI0_dailyshading.fmu'
    shutil.copy(SRC, out)
    with zipfile.ZipFile(SRC) as z:
        names = z.namelist()
        payload = {n: z.read(n) for n in names}
    idf_name = next(n for n in names if n.endswith('.idf'))
    payload[idf_name] = daily_shading(payload[idf_name].decode()).encode()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, payload[n])
    print(f"written: {out}")


if __name__ == '__main__':
    main()
