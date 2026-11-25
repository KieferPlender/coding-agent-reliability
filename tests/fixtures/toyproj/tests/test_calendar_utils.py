import pytest
from toyproj.calendar_utils import clamp_day, days_in_month, is_leap


def test_leap_years():
    assert is_leap(2024)
    assert is_leap(2000)
    assert not is_leap(1900)
    assert not is_leap(2023)


def test_days_in_month():
    assert days_in_month(2023, 1) == 31
    assert days_in_month(2023, 2) == 28
    assert days_in_month(2024, 2) == 29
    assert days_in_month(2023, 12) == 31


def test_month_out_of_range():
    with pytest.raises(ValueError):
        days_in_month(2023, 0)
    with pytest.raises(ValueError):
        days_in_month(2023, 13)


def test_clamp_day():
    assert clamp_day(2023, 2, 31) == 28
    assert clamp_day(2023, 1, 0) == 1
    assert clamp_day(2023, 1, 15) == 15
    assert clamp_day(2023, 1, 31) == 31
    assert clamp_day(2023, 1, 1) == 1
