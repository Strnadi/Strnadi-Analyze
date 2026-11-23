"""Tiny CPU job scheduler for the shared server (hard budget: cores 0..BUDGET-1).

Queue file lines:  name | ncores | shell command   (use {T} for thread count = ncores, {W} workers)
Lines starting with # are ignored.  A job runs once; state in <qdir>/state/<name>.{run,ok,fail}.
Busy cores = affinity of every running process whose cmdline mentions 'strnadi' (also jobs started by hand).
Usage: sched.py <qdir> [budget=45]
"""

from config import WS
import os, sys, time, subprocess, re

QD = sys.argv[1]
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 45
ST = f'{QD}/state'
LOG = f'{WS}/logs'
os.makedirs(ST, exist_ok=True)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = f'source {WS}/env.sh; cd {REPO}; '


def parse_cpus(s):
    out = set()
    for part in s.split(','):
        if '-' in part:
            a, b = map(int, part.split('-'))
            out |= set(range(a, b + 1))
        elif part.strip():
            out.add(int(part))
    return out


def busy_cores():
    busy = set()
    me = os.getpid()
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            cmd = (
                open(f'/proc/{pid}/cmdline', 'rb')
                .read()
                .replace(b'\0', b' ')
                .decode(errors='ignore')
            )
            cwd = os.readlink(f'/proc/{pid}/cwd')
            if 'sched.py' in cmd or not ('strnadi' in cmd or cwd.startswith(REPO)):
                continue
            if os.stat(f'/proc/{pid}').st_uid != os.getuid():
                continue
            st = open(f'/proc/{pid}/status').read()
            cpus = parse_cpus(re.search(r'Cpus_allowed_list:\s*(\S+)', st).group(1))
            if len(cpus) < 79:
                busy |= cpus  # ignore unpinned helpers
        except Exception:
            pass
    return busy


def jobs():
    out = []
    for line in open(f'{QD}/jobs.txt'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, n, cmd = [x.strip() for x in line.split('|', 2)]
        out.append((name, int(n), cmd))
    return out


running = {}
while True:
    for name, p in list(running.items()):
        rc = p.poll()
        if rc is not None:
            os.rename(f'{ST}/{name}.run', f'{ST}/{name}.{"ok" if rc == 0 else "fail"}')
            del running[name]
            print(time.strftime('%H:%M'), 'finished', name, rc, flush=True)
    free = sorted(set(range(BUDGET)) - busy_cores())
    for name, n, cmd in jobs():
        if any(os.path.exists(f'{ST}/{name}.{s}') for s in ('run', 'ok', 'fail')):
            continue
        if len(free) < n:
            break  # strict FIFO: wait until the head job fits
        cores = free[:n]
        free = free[n:]
        cl = ','.join(map(str, cores))
        full = ENV + f'export NT=2; ' + cmd.replace('{T}', str(max(1, n - 1))).replace('{W}', '2')
        open(f'{ST}/{name}.run', 'w').write(cl)
        running[name] = subprocess.Popen(
            ['taskset', '-c', cl, 'zsh', '-c', full],
            stdout=open(f'{LOG}/{name}.txt', 'w'),
            stderr=subprocess.STDOUT,
        )
        print(time.strftime('%H:%M'), 'started', name, 'cores', cl, flush=True)
        time.sleep(20)
    time.sleep(30)
