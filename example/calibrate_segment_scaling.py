#!/usr/bin/env python3
# calibrate_segment_scaling.py — 현재 SDK user 의 calibration URDF 와 Wuji Hand 2 URDF 의 wrist→PIP/DIP/TIP 길이 비율로 segment_scaling 을 만든다
"""Derive `retarget.segment_scaling` from the current SDK user's hand model.

The retargeter matches the human wrist->PIP/DIP/TIP vectors to the robot's, so a
hand longer than Wuji Hand 2 puts targets out of reach and a shorter one over-closes.
This script forward-kinematics both URDFs at the neutral pose and writes
robot_length / human_length per finger segment into the config. It is a starting
point: confirm in tuning_tool (cyan target vs white robot skeleton) and nudge.

Usage:
    python3 calibrate_segment_scaling.py --hand left                      # print only
    python3 calibrate_segment_scaling.py --hand both --write             # 양손을 한 번에
    python3 calibrate_segment_scaling.py --hand left --write             # rewrite the yaml block
    python3 calibrate_segment_scaling.py --hand left --urdf <human.urdf> # explicit human model
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# 사람 URDF(Wuji SDK hand model) 와 Hand 2 URDF 의 PIP/DIP/TIP frame 이름. 엄지는 MCP/IP/TIP 이 같은 역할.
HUMAN_FRAMES = {
    "thumb":  ["thumb_mcp", "thumb_ip", "thumb_tip"],
    "index":  ["index_finger_pip", "index_finger_dip", "index_finger_tip"],
    "middle": ["middle_finger_pip", "middle_finger_dip", "middle_finger_tip"],
    "ring":   ["ring_finger_pip", "ring_finger_dip", "ring_finger_tip"],
    "pinky":  ["pinky_pip", "pinky_dip", "pinky_tip"],
}
ROBOT_FRAMES = {f: [f"{{p}}{n}" for n in HUMAN_FRAMES[f]] for f in FINGERS}
HUMAN_WRIST, ROBOT_WRIST = "wrist", "{p}wrist"


def scaling_from_lengths(human: dict, robot: dict, lo: float = 0.7, hi: float = 1.3) -> dict:
    """robot/human 길이 비율을 마디별로. 범위 밖은 clamp (모델이 이상하면 tuning tool 에서 잡는다)."""
    out = {}
    for f, hl in human.items():
        rl = robot[f]
        out[f] = [round(float(min(max(r / h, lo), hi)), 2) for h, r in zip(hl, rl)]
    return out


def replace_segment_scaling(text: str, scaling: dict, note: str) -> str:
    """yaml 본문에서 `segment_scaling:` 블록의 손가락 5줄만 바꾼다. 주석과 나머지는 그대로."""
    pat = re.compile(r"^(?P<indent>[ \t]*)segment_scaling:[^\n]*\n(?P<body>(?:(?P=indent)[ \t]+\w+:[^\n]*\n)+)", re.M)
    m = pat.search(text)
    if not m:
        raise ValueError("segment_scaling block not found")
    indent = m.group("indent")
    inner = re.match(r"[ \t]+", m.group("body")).group(0)
    lines = [f"{indent}segment_scaling:   # {note}\n"] if note else [f"{indent}segment_scaling:\n"]
    for f in FINGERS:
        v = scaling[f]
        lines.append(f"{inner}{f + ':':8s}[{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]\n")
    return text[:m.start()] + "".join(lines) + text[m.end():]


def _fk_lengths(urdf: str, wrist: str, frames: dict) -> dict:
    import pinocchio as pin
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)

    def pos(name):
        fid = model.getFrameId(name)
        if fid >= model.nframes:
            raise KeyError(f"frame {name!r} not in {urdf}")
        return data.oMf[fid].translation

    w = pos(wrist)
    return {f: [float(np.linalg.norm(pos(n) - w) * 1000) for n in names] for f, names in frames.items()}


def hands_to_process(hand: str) -> list[str]:
    """left | right | both → 처리할 손 목록. both 는 왼손 먼저."""
    return ["left", "right"] if hand == "both" else [hand]


def current_user_urdf(hand: str) -> tuple[str, str]:
    from wuji_sdk import SdkManager
    u = dict(SdkManager.instance().current_user())
    if u.get("is_default"):
        raise SystemExit("current SDK user is Default: no calibrated hand model. `wuji user switch <name>` first.")
    p = Path.home() / ".wuji" / "sdk" / "users" / u["user_id"] / "models" / f"{hand}_hand.urdf"
    if not p.exists():
        raise SystemExit(f"{u['display_name']} has no {hand} hand model at {p}. Run `wuji calib hand-model` first.")
    return str(p), u["display_name"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", choices=["left", "right", "both"], required=True)
    ap.add_argument("--config", default=None, help="default: config/adaptive_analytical_wuji_glove_wuji_hand_2_<hand>.yaml")
    ap.add_argument("--urdf", default=None, help="human hand URDF (default: current SDK user's calibrated model)")
    ap.add_argument("--write", action="store_true", help="rewrite the segment_scaling block in the config")
    args = ap.parse_args(argv)

    if args.urdf and args.hand == "both":
        raise SystemExit("--urdf 는 한 손에만 쓸 수 있습니다. --hand left 또는 right 로 지정하십시오.")

    for hand in hands_to_process(args.hand):
        cfg_path = Path(args.config) if args.config else Path(__file__).parent / "config" / f"adaptive_analytical_wuji_glove_wuji_hand_2_{hand}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        robot_urdf = (cfg_path.parent / cfg["optimizer"]["urdf_path"]).resolve()
        prefix = cfg["optimizer"].get("link_naming", {}).get("prefix", f"{hand[0]}_")

        human_urdf, who = (args.urdf, Path(args.urdf).name) if args.urdf else current_user_urdf(hand)
        human = _fk_lengths(human_urdf, HUMAN_WRIST, HUMAN_FRAMES)
        robot = _fk_lengths(str(robot_urdf), ROBOT_WRIST.format(p=prefix),
                            {f: [n.format(p=prefix) for n in names] for f, names in ROBOT_FRAMES.items()})
        scaling = scaling_from_lengths(human, robot)

        print(f"\n=== {hand} ===\nhuman: {human_urdf}\nrobot: {robot_urdf}")
        print(f"{'finger':7s} {'wrist->PIP':>14s} {'wrist->DIP':>14s} {'wrist->TIP':>14s}   segment_scaling (robot/human)")
        for f in FINGERS:
            h, r = human[f], robot[f]
            print(f"{f:7s} " + " ".join(f"{hh:6.0f}/{rr:<6.0f}" for hh, rr in zip(h, r)) + f"   {scaling[f]}")
        if args.write:
            new = replace_segment_scaling(cfg_path.read_text(), scaling, note=f"auto from {who} {hand}_hand.urdf (calibrate_segment_scaling.py); confirm in tuning_tool")
            cfg_path.write_text(new)
            print(f"wrote {cfg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
