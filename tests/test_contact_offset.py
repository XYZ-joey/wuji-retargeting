# test_contact_offset.py — pinch 접촉 오프셋: 사람 손끝(뼈 끝) 목표를 엄지 쪽으로 당겨 실제 접촉 간격을 메우는지
import numpy as np
import pytest

from wuji_retargeting.opt.adaptive_analytical import apply_contact_offset


def _tips(thumb, index, others=(20.0, 20.0, 20.0)):
    """(5,3) cm. 엄지, 검지, 나머지 세 손가락은 멀리."""
    t = np.zeros((5, 3))
    t[0] = thumb
    t[1] = index
    for k, x in enumerate(others, start=2):
        t[k] = [x, 0.0, 0.0]
    return t


def test_full_pinch_closes_exactly_offset():
    tips = _tips([0, 0, 0], [3.0, 0, 0])            # 엄지-검지 3 cm
    alphas = np.array([0.7, 0.7, 0, 0, 0])          # 검지만 최대 pinch
    out = apply_contact_offset(tips, alphas, alpha_max=0.7, offset_cm=1.0)
    assert np.linalg.norm(out[1] - out[0]) == pytest.approx(2.0)   # 1 cm 만큼 좁힘
    assert out[0] == pytest.approx([0.5, 0, 0])                     # 엄지 0.5 cm
    assert out[1] == pytest.approx([2.5, 0, 0])                     # 검지 0.5 cm
    assert np.array_equal(out[2:], tips[2:])                         # 나머지 손가락 불변


def test_partial_alpha_scales_shift():
    tips = _tips([0, 0, 0], [3.0, 0, 0])
    alphas = np.array([0.35, 0.35, 0, 0, 0])        # alpha_max 의 절반
    out = apply_contact_offset(tips, alphas, alpha_max=0.7, offset_cm=1.0)
    assert np.linalg.norm(out[1] - out[0]) == pytest.approx(2.5)


def test_never_crosses_over():
    tips = _tips([0, 0, 0], [0.4, 0, 0])            # 이미 0.4 cm 로 가까움
    alphas = np.array([1.0, 1.0, 0, 0, 0])
    out = apply_contact_offset(tips, alphas, alpha_max=1.0, offset_cm=1.0)
    assert np.linalg.norm(out[1] - out[0]) == pytest.approx(0.0)   # 겹치지 않고 딱 붙음


def test_thumb_follows_only_closest_finger():
    tips = _tips([0, 0, 0], [3.0, 0, 0], others=(0, 4.0, 20.0))   # 중지 4 cm, 검지 3 cm
    tips[2] = [0, 4.0, 0]
    alphas = np.array([0.7, 0.7, 0.3, 0, 0])
    out = apply_contact_offset(tips, alphas, alpha_max=0.7, offset_cm=1.0)
    assert out[0] == pytest.approx([0.5, 0, 0])      # 엄지는 alpha 가 큰 검지 쪽으로만
    assert out[2][1] == pytest.approx(4.0 - 0.5 * (0.3 / 0.7))   # 중지는 자기 몫만 당겨짐


def test_disabled_when_offset_zero():
    tips = _tips([0, 0, 0], [3.0, 0, 0])
    alphas = np.array([0.7, 0.7, 0, 0, 0])
    out = apply_contact_offset(tips, alphas, alpha_max=0.7, offset_cm=0.0)
    assert np.array_equal(out, tips)
