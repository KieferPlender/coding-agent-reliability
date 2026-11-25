"""A tiny module with a real test suite, used to test the task generator."""

DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_month(year: int, month: int) -> int:
    if month < 1 or month > 12:
        raise ValueError("month out of range")
    if month == 2 and is_leap(year):
        return 29
    return DAYS_IN_MONTH[month - 1]


def clamp_day(year: int, month: int, day: int) -> int:
    """Clamp `day` into the valid range for the month."""
    last = days_in_month(year, month)
    if day > last:
        return last
    if day < 1:
        return 1
    return day


def unused_helper(values):
    # Not covered by any test: mutations here must be rejected as "survived".
    return sum(v * 2 for v in values)
