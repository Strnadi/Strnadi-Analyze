"""List every experiment with its status, origin and first line of description.

    python -m tools.list_experiments [status]        e.g. `proposed`, `done`

The information is read from the markdown header of each script; nothing is executed.
"""

import glob
import os
import re
import sys

EXPERIMENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments'
)


def read_header(path):
    """Return (status, origin, description) from an experiment script's markdown header."""
    with open(path) as file:
        text = file.read()

    header_lines = []
    for line in text.split('\n')[1:]:
        if not line.startswith('#'):
            break
        header_lines.append(line[2:] if line.startswith('# ') else '')
    header = '\n'.join(header_lines)

    status = re.search(r'\*\*Status:\*\* (\w+) \(from ([^)]*)\)', header)
    paragraphs = [p.replace('\n', ' ') for p in header.split('\n\n')]
    description = paragraphs[1] if len(paragraphs) > 1 else ''
    if status is None:
        return '?', '?', description
    return status.group(1), status.group(2), description


def main():
    wanted_status = sys.argv[1] if len(sys.argv) > 1 else None
    for path in sorted(glob.glob(f'{EXPERIMENT_DIR}/*.py')):
        name = os.path.basename(path)[:-3]
        if name == '__init__':
            continue
        status, origin, description = read_header(path)
        if wanted_status and status != wanted_status:
            continue
        print(f'{name:32s} {status:9s} {origin:30s} {description[:90]}')


if __name__ == '__main__':
    main()
