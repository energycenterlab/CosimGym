"""
eplus_worker.py

Runs ONE EnergyPlus simulation through ``pyenergyplus`` inside its own process and talks to
the CosimGym model over an inherited socket (JSON lines). Started as a script by
``base_eplus_model.py``; imports nothing from CosimGym (and no helics), so any EnergyPlus
install, whatever its version, can be put on ``sys.path`` and used.

EnergyPlus calls us (callbacks), CosimGym calls the model once per tick; the callbacks are
the rendezvous. Per tick (``group`` zone timesteps):

    begin zone timestep, first of the group:  recv {"t":"step","in":{...}}  -> set actuators
    end zone timestep, last of the group:     send {"t":"out","out":{...}}  (read variables)

parent -> worker:  {"t":"step","in":{name: value}} | {"t":"stop"}
worker -> parent:  {"t":"ready","dow":..,"year":..}            handles resolved, simulation running
                   {"t":"discovery","path":..}                  (discover mode) API data written
                   {"t":"out","out":{...},"clk":[doy,hour,zts]}
                   {"t":"error","msg":..} | {"t":"done","rc":int,"err":[...]}

Facts this relies on (measured, see docs/changes_reports/eplus_model_plan.md section 12):
  * ``current_sim_time`` is NOT a tick boundary under HVAC sub-stepping: never used for time;
  * a failure inside a callback is swallowed by EnergyPlus (the run would silently go on), so
    every callback reports the error and kills the process;
  * ``Runtime.stop_simulation`` does not exist before 24.1: without it the worker exits itself.
"""
import json
import os
import socket
import sys
import threading
import time
import traceback


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    sock = socket.socket(fileno=cfg['fd'])
    buf = [b'']

    def send(obj):
        sock.sendall((json.dumps(obj) + '\n').encode())

    def recv():
        while b'\n' not in buf[0]:
            chunk = sock.recv(65536)
            if not chunk:          # parent closed the pipe / died
                os._exit(3)
            buf[0] += chunk
        line, buf[0] = buf[0].split(b'\n', 1)
        return json.loads(line)

    def watch_parent():
        while True:
            time.sleep(1.0)
            if os.getppid() != cfg['parent_pid']:
                os._exit(4)
    threading.Thread(target=watch_parent, daemon=True).start()

    sys.path.insert(0, cfg['install'])
    from pyenergyplus.api import EnergyPlusAPI
    api = EnergyPlusAPI()
    ex, rt = api.exchange, api.runtime
    st = api.state_manager.new_state()

    ins, outs = cfg['inputs'], cfg['outputs']
    group, skip = cfg['group'], cfg['skip_ticks']
    for o in outs.values():
        if o['kind'] == 'variable':
            ex.request_variable(st, o['name'], o['key'])

    h = {}                         # name -> handle
    n = [0]                        # zone timesteps seen in the run period
    acc = {}                       # per-output accumulators within a tick
    dead = [False]
    stop_sim = getattr(rt, 'stop_simulation', None)

    def fail(msg):
        try:
            send({'t': 'error', 'msg': msg})
        finally:
            os._exit(2)

    def active(s):
        return ex.api_data_fully_ready(s) and not ex.warmup_flag(s) and ex.kind_of_sim(s) == 3

    def hints(s, spec):
        try:
            data = ex.list_available_api_data_csv(s)
            data = data.decode() if isinstance(data, bytes) else data
        except Exception:
            return []
        needle = [str(spec.get(k, '')).lower() for k in ('key', 'name', 'control_type') if spec.get(k)]
        return [l for l in data.splitlines() if any(x and x in l.lower() for x in needle)][:12]

    def handle(s, spec):
        k = spec['kind']
        if k == 'actuator':
            return ex.get_actuator_handle(s, spec['component_type'], spec['control_type'], spec['key'])
        if k == 'global':
            return ex.get_global_handle(s, spec['name'])
        if k == 'variable':
            return ex.get_variable_handle(s, spec['name'], spec['key'])
        if k == 'meter':
            return ex.get_meter_handle(s, spec['name'])
        raise ValueError(f'unknown I/O kind {k!r}')

    def read(s, spec, hd):
        k = spec['kind']
        return {'variable': ex.get_variable_value, 'meter': ex.get_meter_value,
                'actuator': ex.get_actuator_value, 'global': ex.get_global_value}[k](s, hd)

    def write(s, spec, hd, v):
        (ex.set_actuator_value if spec['kind'] == 'actuator' else ex.set_global_value)(s, hd, float(v))

    def resolve(s):
        if cfg.get('discover'):
            path = os.path.join(cfg['outdir'], 'api_data.csv')
            data = ex.list_available_api_data_csv(s)
            open(path, 'w').write(data.decode() if isinstance(data, bytes) else data)
            send({'t': 'discovery', 'path': path})
            os._exit(0)
        bad = {}
        for name, spec in {**ins, **outs}.items():
            h[name] = handle(s, spec)
            if h[name] == -1:
                bad[name] = hints(s, spec)
        if bad:
            fail('unresolved handles (not found in this IDF / version): ' + json.dumps(bad, indent=1))
        send({'t': 'ready', 'dow': ex.day_of_week(s), 'year': ex.year(s)})

    def begin(s):
        if dead[0] or not active(s):
            return
        if not h:
            resolve(s)
        if n[0] == 0:              # defaults hold during the burn-in (and until the first tick)
            for name, spec in ins.items():
                if spec.get('default') is not None:
                    write(s, spec, h[name], spec['default'])
        j = n[0] - skip
        n[0] += 1
        if j >= 0 and j % group == 0:
            msg = recv()
            if msg['t'] == 'stop':
                dead[0] = True
                if stop_sim:
                    stop_sim(s)
                else:
                    send({'t': 'done', 'rc': 0, 'err': []})
                    os._exit(0)
                return
            for name, v in msg['in'].items():
                if name in ins:
                    write(s, ins[name], h[name], v)

    def end(s):
        if dead[0] or not h or not active(s):
            return
        j = n[0] - 1 - skip
        if j < 0:
            return
        for name, spec in outs.items():
            v = read(s, spec, h[name])
            a = acc.setdefault(name, [0.0, 0, v, v, v])      # sum, count, last, min, max
            a[0] += v; a[1] += 1; a[2] = v; a[3] = min(a[3], v); a[4] = max(a[4], v)
        if (j + 1) % group == 0:
            res = {}
            for name, spec in outs.items():
                sm, c, last, lo, hi = acc[name]
                res[name] = {'last': last, 'mean': sm / c, 'sum': sm, 'min': lo, 'max': hi}[spec.get('agg', 'last')]
            acc.clear()
            send({'t': 'out', 'out': res, 'clk': [ex.day_of_year(s), ex.hour(s), ex.zone_time_step_number(s)]})

    def guard(fn):
        def wrapped(s):
            try:
                fn(s)
            except BaseException:
                fail(traceback.format_exc())
        return wrapped

    rt.callback_begin_zone_timestep_after_init_heat_balance(st, guard(begin))
    rt.callback_end_zone_timestep_after_zone_reporting(st, guard(end))
    args = ['-d', cfg['outdir']] + (['-w', cfg['epw']] if cfg.get('epw') else []) + [cfg['idf']]
    rc = rt.run_energyplus(st, args)
    err = []
    if rc != 0:
        try:
            err = open(os.path.join(cfg['outdir'], 'eplusout.err')).read().splitlines()[-25:]
        except OSError:
            pass
    send({'t': 'done', 'rc': rc, 'err': err})


if __name__ == '__main__':
    main(sys.argv[1])
