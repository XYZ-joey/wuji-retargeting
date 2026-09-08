# upright.py — 손이 손끝을 아래(-z)로 두고 로드되는 MJCF 를 뷰어에서 손끝이 위로 오게 root body 를 뒤집는다
"""Flip a wrist-rooted hand model so fingertips point up in the viewer.

The Wuji hand MJCFs place the wrist at the origin with fingers along -z, so the
free camera (which cannot roll) always shows fingertips pointing down. Rotating
the first world-attached body 180 deg about x makes fingers point +z. Joint
kinematics and actuator order are untouched; only the base pose changes.
"""
from __future__ import annotations

import mujoco
import numpy as np

FLIP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # (w, x, y, z): 180 deg about x


def flip_hand_upright(model: mujoco.MjModel) -> int:
    """Rotate the root hand body in place. Returns the body id that was flipped."""
    root = next(b for b in range(1, model.nbody) if model.body_parentid[b] == 0)
    q = model.body_quat[root].copy()
    # compose: new = FLIP * q  (quat multiply, w-first)
    w1, x1, y1, z1 = FLIP_QUAT
    w2, x2, y2, z2 = q
    model.body_quat[root] = [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
    return root
