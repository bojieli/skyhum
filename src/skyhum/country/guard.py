"""Audit hook that keeps a search process blind to the truth.

install(tree, deny, allowed_read, allowed_write) registers a sys.addaudithook callback (it cannot be removed) that
raises PermissionError when the process opens, lists or hands to a subprocess any path that is
  * under a deny prefix (the truth directory), always, or
  * inside the protected tree (the work directory) but outside the allowed prefixes.
Paths outside the tree (Python, libraries, /proc, caches) are not restricted. Every distinct tree path the process
touched is recorded in TOUCHED; the search writes it into its output so the scorer can re-check it.
"""
import os
import sys

TOUCHED = set()
_STATE = {}


def _norm(p):
    if isinstance(p, int):
        return None
    if isinstance(p, bytes):
        p = p.decode("utf-8", "replace")
    if not isinstance(p, str):
        try:
            p = os.fspath(p)
        except TypeError:
            return None
        if isinstance(p, bytes):
            p = p.decode("utf-8", "replace")
    return os.path.realpath(os.path.abspath(p))


def check(p, write=False):
    if not _STATE:
        return
    q = _norm(p)
    if q is None:
        return
    for d in _STATE["deny"]:
        if q == d or q.startswith(d + os.sep):
            raise PermissionError(f"guard: access to {q} is forbidden")
    tree = _STATE["tree"]
    if q == tree or q.startswith(tree + os.sep):
        allow = _STATE["w"] if write else _STATE["r"] + _STATE["w"]
        if not any(q == a or q.startswith(a + os.sep) for a in allow):
            raise PermissionError(f"guard: {q} is outside the allowed inputs")
        TOUCHED.add(q)


def _hook(event, args):
    if event == "open":
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        write = any(c in mode for c in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        check(args[0], write)
    elif event in ("os.listdir", "os.scandir", "glob.glob", "os.chdir", "shutil.copyfile", "os.rename", "os.remove"):
        if args and args[0] is not None:
            check(args[0], event in ("os.rename", "os.remove"))
    elif event == "subprocess.Popen":
        argv = args[1] if len(args) > 1 else None
        for a in (argv if isinstance(argv, (list, tuple)) else [argv]):
            if isinstance(a, bytes):
                a = a.decode("utf-8", "replace")
            if isinstance(a, str) and "/" in a and not a.startswith("-"):
                check(a)
            elif isinstance(a, str) and a.startswith("-") and "/" in a:
                q = a.split("=", 1)[-1] if "=" in a else a[2:]
                if q.startswith("/"):
                    check(q)


def install(tree, deny, allowed_read, allowed_write):
    """Once per process. tree: the protected directory; deny: prefixes never readable."""
    if _STATE:
        return
    _STATE["tree"] = os.path.realpath(os.path.expanduser(tree))
    _STATE["deny"] = [os.path.realpath(os.path.expanduser(d)) for d in deny]
    _STATE["r"] = [os.path.realpath(os.path.expanduser(a)) for a in allowed_read]
    _STATE["w"] = [os.path.realpath(os.path.expanduser(a)) for a in allowed_write]
    for a in _STATE["r"] + _STATE["w"]:
        for d in _STATE["deny"]:
            if a == d or a.startswith(d + os.sep) or d.startswith(a + os.sep):
                raise PermissionError(f"guard: allowed path {a} overlaps the deny list ({d})")
    sys.addaudithook(_hook)
