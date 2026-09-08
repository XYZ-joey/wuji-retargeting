# test_segment_scaling.py — 사람 손 URDF 와 로봇 URDF 의 wrist→PIP/DIP/TIP 길이 비율로 segment_scaling 을 만들고 yaml 블록을 바꿔 쓰는지
import textwrap
import pytest

from calibrate_segment_scaling import scaling_from_lengths, replace_segment_scaling


def test_ratio_is_robot_over_human_per_segment():
    human = {"index": [100.0, 120.0, 140.0]}
    robot = {"index": [90.0, 120.0, 154.0]}
    out = scaling_from_lengths(human, robot)
    assert out["index"] == pytest.approx([0.9, 1.0, 1.1])


def test_ratio_is_rounded_and_clamped():
    human = {"thumb": [50.0, 100.0, 100.0]}
    robot = {"thumb": [100.0, 33.3333, 100.0]}          # 2.0 과 0.33 은 범위 밖
    out = scaling_from_lengths(human, robot, lo=0.7, hi=1.3)
    assert out["thumb"] == [1.3, 0.7, 1.0]


def test_replace_keeps_surrounding_yaml_and_comments():
    src = textwrap.dedent('''\
        retarget:
          w_pos: 1.0
          segment_scaling:
            thumb:  [1.0, 1.0, 1.0]
            index:  [1.0, 1.0, 1.0]
            middle: [1.0, 1.0, 1.0]
            ring:   [1.0, 1.0, 1.0]
            pinky:  [1.0, 1.0, 1.0]
          pinch_thresholds:   # keep me
            index:  { d1: 2.0, d2: 4.0 }
        ''')
    new = {"thumb": [1.0, 0.95, 0.94], "index": [0.93, 0.94, 0.99], "middle": [0.86, 0.94, 0.92],
           "ring": [0.83, 0.93, 0.89], "pinky": [0.99, 0.98, 0.98]}
    out = replace_segment_scaling(src, new, note="from xyz_joey left_hand.urdf")
    assert "    ring:   [0.83, 0.93, 0.89]" in out
    assert "pinch_thresholds:   # keep me" in out and "w_pos: 1.0" in out
    assert "from xyz_joey left_hand.urdf" in out
    assert out.count("segment_scaling:") == 1


def test_replace_refuses_when_block_missing():
    with pytest.raises(ValueError):
        replace_segment_scaling("retarget:\n  w_pos: 1.0\n", {"thumb": [1, 1, 1]}, note="")
