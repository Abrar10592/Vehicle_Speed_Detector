"""Keep a camera running at full speed on Windows laptops.

Windows treats a program without a window as background work: on processors with power-saving cores (e.g. Core 5
120U) it moves such a program onto those cores, which halved the frame rate of each camera (about 5 instead of
10 frames per second). It also lets the computer fall asleep when nobody uses it. Both are switched off for as long
as the camera runs. A person choosing Sleep or closing the lid still puts the computer to sleep.
"""

from __future__ import annotations

import os


def keep_full_speed() -> None:
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    process_power_throttling, execution_speed = 4, 0x1
    # ControlMask: this program decides about execution speed; StateMask 0: do not slow it down
    state = PowerThrottlingState(1, execution_speed, 0)
    kernel32.SetProcessInformation(
        kernel32.GetCurrentProcess(), process_power_throttling, ctypes.byref(state), ctypes.sizeof(state)
    )
    es_continuous, es_system_required = 0x80000000, 0x00000001
    kernel32.SetThreadExecutionState(es_continuous | es_system_required)  # no sleeping when idle
