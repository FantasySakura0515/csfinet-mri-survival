import pytest

from csfinet_repro.reporting import absolute_error_bin, residual_bin, select_cases


@pytest.mark.parametrize("value,expected", [(0, "[0,180]"), (180, "[0,180]"), (180.1, "(180,360]"),
                                             (360, "(180,360]"), (720, "(540,720]"), (721, "(720,inf)")])
def test_absolute_error_bins_have_no_boundary_gaps(value, expected):
    assert absolute_error_bin(value) == expected


@pytest.mark.parametrize("value,expected", [(-721, "(-inf,-720)"), (-720, "[-720,-540)"),
                                             (-180, "[-180,0)"), (-.1, "[-180,0)"), (0, "{0}"),
                                             (.1, "(0,180]"), (180, "(0,180]"), (721, "(720,inf)")])
def test_residual_bins_cover_boundaries_and_isolate_zero(value, expected):
    assert residual_bin(value) == expected


def test_case_selection_retains_illustrative_case_and_is_deterministic():
    rows = [dict(patient_id=f"p{i}", absolute_error=i) for i in range(10)]
    rows[4]["patient_id"] = "BraTS20_Training_199"
    selected = select_cases(list(reversed(rows)))
    assert selected[0]["patient_id"] == "BraTS20_Training_199"
    assert [row["patient_id"] for row in selected] == [row["patient_id"] for row in select_cases(list(reversed(rows)))]
    assert len({row["patient_id"] for row in selected}) == 3
