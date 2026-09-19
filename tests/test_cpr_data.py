"""Guards for the CPR-Coach loader.

The pieces most likely to go wrong silently are the ones tested here: the
four synchronised channels of a take must collapse into one row rather than
four, the ``_BIG`` suffix must not fork a folder into two recording groups,
and the COCO-17 normalisation must divide by a body scale rather than a
picture scale. Each failure would corrupt the split rather than raise.
"""
import numpy as np
import pytest

from cpr_data import (NCRIT, _parse_frame_dir, fill_missing_frames,
                      normalize_pose, resample)


def test_frame_dir_parses_into_group_take_and_channel():
    group, key, channel = _parse_frame_dir("CPR_Double_Dataset_S0/DC00_S0/r00/ch2")
    assert group == "CPR_Double_Dataset_S0/DC00_S0"
    assert key == "CPR_Double_Dataset_S0/DC00_S0/r00"
    assert channel == "ch2"


def test_big_suffix_does_not_split_one_folder_into_two_groups():
    """The keypoint file spells some folders _BIG; the label lists do not.

    Left unnormalised, the same recording session would appear as two groups,
    and group-disjointness between train and test would silently stop holding.
    """
    plain = _parse_frame_dir("CPR_Double_Dataset_S0/DC00_S0/r00/ch0")
    big = _parse_frame_dir("CPR_Double_Dataset_S0_BIG/DC00_S0/r00/ch0")
    assert plain[0] == big[0]
    assert plain[1] == big[1]


def test_malformed_frame_dir_raises():
    with pytest.raises(ValueError):
        _parse_frame_dir("CPR_Dataset_S0/C00_S0/r00")


def test_resample_hits_the_requested_length_and_preserves_endpoints():
    seq = np.arange(40, dtype=np.float32).reshape(10, 2, 2)
    out = resample(seq, T=16)
    assert out.shape == (16, 2, 2)
    assert out[0] == pytest.approx(seq[0])
    assert out[-1] == pytest.approx(seq[-1])


def test_resample_is_identity_at_matching_length():
    seq = np.random.default_rng(0).normal(size=(16, 17, 2)).astype(np.float32)
    assert resample(seq, T=16) == pytest.approx(seq)


def test_whole_missing_frames_are_interpolated_not_left_at_zero():
    seq = np.ones((5, 17, 2), dtype=np.float32)
    seq[2] = 0.0                                   # a failed detection
    filled, valid = fill_missing_frames(seq)
    assert not valid[2] and valid.sum() == 4
    assert filled[2] == pytest.approx(np.ones((17, 2)))


def test_partial_joints_are_left_untouched():
    """A single zero joint may be a real coordinate; the release cannot say."""
    seq = np.ones((4, 17, 2), dtype=np.float32)
    seq[1, 5] = 0.0
    filled, valid = fill_missing_frames(seq)
    assert valid.all()
    assert filled[1, 5] == pytest.approx(np.zeros(2))


def test_an_all_zero_sequence_reports_no_valid_frame():
    seq = np.zeros((4, 17, 2), dtype=np.float32)
    _, valid = fill_missing_frames(seq)
    assert not valid.any()


def test_normalisation_centres_the_pelvis_and_is_scale_invariant():
    rng = np.random.default_rng(7)
    pose = rng.normal(size=(6, 17, 2)).astype(np.float32)
    out = normalize_pose(pose.copy())
    pelvis = (out[:, 11, :] + out[:, 12, :]) / 2
    assert pelvis == pytest.approx(np.zeros_like(pelvis), abs=1e-5)
    # A camera twice as close must not look like a different movement.
    assert normalize_pose(pose * 2.0) == pytest.approx(out, abs=1e-4)


def test_normalisation_survives_a_degenerate_pose():
    """Zero body extent would divide by zero; the floor keeps it finite."""
    assert np.isfinite(normalize_pose(np.zeros((3, 17, 2), dtype=np.float32))).all()


def test_criterion_count_matches_the_released_action_list():
    assert NCRIT == 13
