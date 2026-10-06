"""Linux `unshare` sandbox setup, run as root of a fresh user + mount namespace: builds a new root on a tmpfs that
contains only read-only binds of the system library dirs and the Python runtime, the scratch dir (read-write), a few
devices and /proc, hides `deny_read` dirs under empty read-only tmpfs, pivots into it and detaches the old root, then
drops every capability (bounding set included, `SECBIT_NOROOT` locked, `no_new_privs`) and execs the runner. Nothing
else of the host file system (in particular $HOME) is reachable, and the program cannot chroot, mount or unmount its
way back to it.

argv: scratch, JSON list of read-only bind sources, JSON list of dirs to hide, "--", command...
"""

import ctypes
import json
import os
import subprocess
import sys

DEVICES = ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")
OLD_ROOT = "/.oldroot"
MNT_DETACH = 2
PR_CAPBSET_DROP, PR_SET_SECUREBITS, PR_SET_NO_NEW_PRIVS, PR_CAP_AMBIENT = 24, 28, 38, 47
PR_CAP_AMBIENT_CLEAR_ALL = 4
# SECBIT_NOROOT and SECBIT_NO_SETUID_FIXUP set, SECBIT_KEEP_CAPS clear, all three locked.
SECUREBITS = 0x01 | 0x02 | 0x04 | 0x08 | 0x20
LINUX_CAPABILITY_VERSION_3 = 0x20080522
SYS_PIVOT_ROOT = {"x86_64": 155, "aarch64": 41, "riscv64": 41, "ppc64le": 203, "s390x": 217}

_libc = ctypes.CDLL(None, use_errno=True)


def _check(ret: int, what: str) -> None:
    if ret != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"{what}: {os.strerror(e)}")


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


def _pivot(root: str) -> None:
    """Makes `root` the namespace's root and unmounts the old one, so no path or directory fd leads back to it."""
    nr = SYS_PIVOT_ROOT.get(os.uname().machine)
    if nr is None:
        raise OSError(f"pivot_root: unknown syscall number on {os.uname().machine}")
    os.chdir(root)
    _check(_libc.syscall(nr, b".", OLD_ROOT[1:].encode()), "pivot_root")
    os.chdir("/")
    _check(_libc.umount2(OLD_ROOT.encode(), MNT_DETACH), "umount2")


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]


def _drop_privileges() -> None:
    """Drops all capabilities for this process and everything it execs: uid 0 of the namespace keeps none."""
    last = int(open("/proc/sys/kernel/cap_last_cap").read())
    for cap in range(last + 1):
        _check(_libc.prctl(PR_CAPBSET_DROP, cap, 0, 0, 0), f"PR_CAPBSET_DROP {cap}")
    _check(_libc.prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0), "PR_CAP_AMBIENT_CLEAR_ALL")
    _check(_libc.prctl(PR_SET_SECUREBITS, SECUREBITS, 0, 0, 0), "PR_SET_SECUREBITS")
    _check(_libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "PR_SET_NO_NEW_PRIVS")
    hdr, data = _CapHeader(LINUX_CAPABILITY_VERSION_3, 0), (_CapData * 2)()
    _check(_libc.capset(ctypes.byref(hdr), data), "capset")


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
    os.mkdir(root + OLD_ROOT, 0)
    _mount("-o", "remount,bind,ro", root)
    _pivot(root)
    _drop_privileges()
    os.chdir(scratch)
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    main()
