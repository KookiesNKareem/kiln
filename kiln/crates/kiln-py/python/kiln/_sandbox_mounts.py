"""Linux `unshare` sandbox setup, run as root of a fresh user + mount namespace: builds a new root on a tmpfs that
contains only read-only binds of the system library dirs and the Python runtime, the scratch dir (read-write), a few
devices and /proc, hides `deny_read` dirs under empty read-only tmpfs, chroots into it and execs the runner. Nothing
else of the host file system (in particular $HOME) is reachable.

argv: scratch, JSON list of read-only bind sources, JSON list of dirs to hide, "--", command...
"""

import json
import os
import subprocess
import sys

DEVICES = ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")


def _mount(*args: str) -> None:
    subprocess.run(["mount", *args], check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE)


def _bind(src: str, root: str, ro: bool) -> None:
    dst = root + src
    if os.path.islink(src) and os.path.dirname(src) == "/":  # merged /usr: /lib -> usr/lib
        os.symlink(os.readlink(src), dst)
        return
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "a").close()
    _mount("--rbind", src, dst)
    if ro:
        _mount("-o", "remount,bind,ro", dst)


def main() -> None:
    scratch, binds, hidden, sep, *cmd = sys.argv[1:]
    if sep != "--" or not cmd:
        sys.exit("usage: _sandbox_mounts.py SCRATCH BINDS_JSON HIDDEN_JSON -- COMMAND...")
    _mount("--make-rprivate", "/")
    root = os.path.join(scratch, ".root")
    os.mkdir(root, 0o755)
    _mount("-t", "tmpfs", "-o", "size=1m,mode=755", "tmpfs", root)
    for src in sorted(json.loads(binds), key=len):
        _bind(src, root, ro=True)
    for dev in DEVICES:
        if os.path.exists(dev):
            _bind(dev, root, ro=False)
    os.makedirs(root + "/proc", exist_ok=True)
    _mount("-t", "proc", "proc", root + "/proc")
    os.makedirs(root + scratch, exist_ok=True)
    _mount("--bind", scratch, root + scratch)
    for d in json.loads(hidden):
        if os.path.isdir(root + d):
            _mount("-t", "tmpfs", "-o", "ro,size=4k,mode=000", "tmpfs", root + d)
    _mount("-o", "remount,ro", root)
    os.chroot(root)
    os.chdir(scratch)
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    main()
