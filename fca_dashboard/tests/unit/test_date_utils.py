"""Tests for date and time utility functions."""
import datetime

import pytest
from freezegun import freeze_time

from fca_dashboard.utils.date_utils import format_date, parse_date, time_since


class TestFormatDate:
    """Tests for the format_date function."""

    def test_format_date_default(self):
        """Test formatting a date with default format."""
        date = datetime.datetime(2023, 5, 15, 14, 30, 0)
        assert format_date(date) == "May 15, 2023"

    def test_format_date_custom_format(self):
        """Test formatting a date with a custom format."""
        date = datetime.datetime(2023, 5, 15, 14, 30, 0)
        assert format_date(date, "%Y-%m-%d") == "2023-05-15"

    def test_format_date_with_time(self):
        """Test formatting a date with time."""
        date = datetime.datetime(2023, 5, 15, 14, 30, 0)
        assert format_date(date, "%b %d, %Y %H:%M") == "May 15, 2023 14:30"

    def test_format_date_none(self):
        """Test formatting None date."""
        assert format_date(None) == ""

    def test_format_date_with_default_value(self):
        """Test formatting None date with a default value."""
        assert format_date(None, default="N/A") == "N/A"


class TestTimeSince:
    """Tests for the time_since function."""

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_seconds(self):
        """Test time since for seconds."""
        date = datetime.datetime(2023, 5, 15, 14, 29, 30)
        assert time_since(date) == "30 seconds ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_minute(self):
        """Test time since for a minute."""
        date = datetime.datetime(2023, 5, 15, 14, 29, 0)
        assert time_since(date) == "1 minute ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_minutes(self):
        """Test time since for minutes."""
        date = datetime.datetime(2023, 5, 15, 14, 25, 0)
        assert time_since(date) == "5 minutes ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_hour(self):
        """Test time since for an hour."""
        date = datetime.datetime(2023, 5, 15, 13, 30, 0)
        assert time_since(date) == "1 hour ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_hours(self):
        """Test time since for hours."""
        date = datetime.datetime(2023, 5, 15, 10, 30, 0)
        assert time_since(date) == "4 hours ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_day(self):
        """Test time since for a day."""
        date = datetime.datetime(2023, 5, 14, 14, 30, 0)
        assert time_since(date) == "1 day ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_days(self):
        """Test time since for days."""
        date = datetime.datetime(2023, 5, 10, 14, 30, 0)
        assert time_since(date) == "5 days ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_month(self):
        """Test time since for a month."""
        date = datetime.datetime(2023, 4, 15, 14, 30, 0)
        assert time_since(date) == "1 month ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_months(self):
        """Test time since for months."""
        date = datetime.datetime(2023, 1, 15, 14, 30, 0)
        assert time_since(date) == "4 months ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_year(self):
        """Test time since for a year."""
        date = datetime.datetime(2022, 5, 15, 14, 30, 0)
        assert time_since(date) == "1 year ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_years(self):
        """Test time since for years."""
        date = datetime.datetime(2020, 5, 15, 14, 30, 0)
        assert time_since(date) == "3 years ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_future(self):
        """Test time since for a future date."""
        date = datetime.datetime(2023, 5, 16, 14, 30, 0)
        assert time_since(date) == "in 1 day"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_none(self):
        """Test time since for None date."""
        assert time_since(None) == ""


class TestParseDate:
    """Tests for the parse_date function."""

    def test_parse_date_iso_format(self):
        """Test parsing a date in ISO format."""
        assert parse_date("2023-05-15") == datetime.datetime(2023, 5, 15, 0, 0, 0)

    def test_parse_date_with_time(self):
        """Test parsing a date with time."""
        assert parse_date("2023-05-15 14:30:00") == datetime.datetime(2023, 5, 15, 14, 30, 0)

    def test_parse_date_custom_format(self):
        """Test parsing a date with a custom format."""
        assert parse_date("15/05/2023", format="%d/%m/%Y") == datetime.datetime(2023, 5, 15, 0, 0, 0)

    @freeze_time("2023-05-15 14:30:00")
    def test_parse_date_yesterday(self):
        """Test parsing 'yesterday'."""
        expected = datetime.datetime(2023, 5, 14, 0, 0, 0)
        assert parse_date("yesterday") == expected

    @freeze_time("2023-05-15 14:30:00")
    def test_parse_date_today(self):
        """Test parsing 'today'."""
        expected = datetime.datetime(2023, 5, 15, 0, 0, 0)
        assert parse_date("today") == expected

    @freeze_time("2023-05-15 14:30:00")
    def test_parse_date_tomorrow(self):
        """Test parsing 'tomorrow'."""
        expected = datetime.datetime(2023, 5, 16, 0, 0, 0)
        assert parse_date("tomorrow") == expected

    @freeze_time("2023-05-15 14:30:00")
    def test_parse_date_days_ago(self):
        """Test parsing 'X days ago'."""
        expected = datetime.datetime(2023, 5, 10, 0, 0, 0)
        assert parse_date("5 days ago") == expected
        
    @freeze_time("2023-05-15 14:30:00")
    def test_parse_date_invalid_days_ago_format(self):
        """Test parsing an invalid 'X days ago' format."""
        # This should fall through to the dateutil parser and raise ValueError
        with pytest.raises(ValueError):
            parse_date("invalid days ago")
            
    @freeze_time("2023-05-15 14:30:00")
    def test_parse_date_empty_days_ago_format(self):
        """Test parsing an empty 'days ago' format."""
        # This should fall through to the dateutil parser and raise ValueError
        with pytest.raises(ValueError):
            parse_date(" days ago")

    def test_parse_date_datetime_object(self):
        """Test parsing a datetime object."""
        dt = datetime.datetime(2023, 5, 15, 14, 30, 0)
        assert parse_date(dt) is dt

    def test_parse_date_invalid(self):
        """Test parsing an invalid date."""
        with pytest.raises(ValueError):
            parse_date("not a date")

    def test_parse_date_none(self):
        """Test parsing None."""
        assert parse_date(None) is None

    def test_parse_date_empty(self):
        """Test parsing an empty string."""
        assert parse_date("") is None
