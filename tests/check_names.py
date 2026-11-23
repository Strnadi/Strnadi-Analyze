"""Static check: every name used in every Python file of the repo is defined, imported or a builtin, and every
`from <repo package> import name` resolves. A minimal stand-in for pyflakes (not installed on the training
server); it catches the typical damage of moving code around (a leftover call to something no longer imported).

    python -m tests.check_names
"""

import ast
import builtins
import glob
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = (
    'layers',
    'model',
    'augment',
    'losses',
    'training',
    'experiments',
    'processing',
    'evaluation',
    'serving',
    'tests',
    'tools',
)


def defined_names(tree):
    names = set(dir(builtins)) | {'__file__', '__name__', '__doc__'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split('.')[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if not isinstance(node, ast.ClassDef):
                a = node.args
                names |= (
                    {x.arg for x in a.args + a.kwonlyargs + a.posonlyargs}
                    | ({a.vararg.arg} if a.vararg else set())
                    | ({a.kwarg.arg} if a.kwarg else set())
                )
        elif isinstance(node, ast.Lambda):
            a = node.args
            names |= (
                {x.arg for x in a.args + a.kwonlyargs + a.posonlyargs}
                | ({a.vararg.arg} if a.vararg else set())
                | ({a.kwarg.arg} if a.kwarg else set())
            )
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names |= set(node.names)
    return names


def main():
    sys.path.insert(0, ROOT)
    bad = 0
    files = [f for p in PACKAGES for f in glob.glob(f'{ROOT}/{p}/*.py')] + glob.glob(f'{ROOT}/*.py')
    for path in sorted(files):
        rel = os.path.relpath(path, ROOT)
        tree = ast.parse(open(path).read(), path)
        known = defined_names(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in known
            ):
                print(f'{rel}:{node.lineno}: undefined name {node.id}')
                bad += 1
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and node.module.split('.')[0] in PACKAGES + ('config',)
            ):
                try:
                    src = open(importlib.util.find_spec(node.module).origin).read()
                except (ImportError, AttributeError, ValueError):
                    print(f'{rel}:{node.lineno}: cannot find module {node.module}')
                    bad += 1
                    continue
                target = defined_names(ast.parse(src))
                for a in node.names:
                    is_submodule = os.path.exists(
                        f'{ROOT}/{node.module.replace(".", "/")}/{a.name}.py'
                    )
                    if a.name != '*' and a.name not in target and not is_submodule:
                        print(f'{rel}:{node.lineno}: {node.module} has no name {a.name}')
                        bad += 1
    print(f'{len(files)} files checked, {bad} problems')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
