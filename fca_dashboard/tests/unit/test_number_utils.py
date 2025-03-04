"""Unit tests for number utilities."""
import re
from decimal import Decimal

import pytest

from fca_dashboard.utils.number_utils import format_currency, random_number, round_to


class TestFormatCurrency:
    """Test cases for currency formatting function."""

    def test_integer_values(self):
        """Test formatting integer values as currency."""
        assert format_currency(1234) == "$1,234.00"
        assert format_currency(0) == "$0.00"
        assert format_currency(-1234) == "-$1,234.00"

    def test_float_values(self):
        """Test formatting float values as currency."""
        assert format_currency(1234.56) == "$1,234.56"
        assert format_currency(1234.5) == "$1,234.50"
        assert format_currency(0.99) == "$0.99"
        assert format_currency(-1234.56) == "-$1,234.56"

    def test_decimal_values(self):
        """Test formatting Decimal values as currency."""
        assert format_currency(Decimal("1234.56")) == "$1,234.56"
        assert format_currency(Decimal("1234.5")) == "$1,234.50"
        assert format_currency(Decimal("0.99")) == "$0.99"
        assert format_currency(Decimal("-1234.56")) == "-$1,234.56"

    def test_custom_currency_symbol(self):
        """Test formatting with custom currency symbols."""
        assert format_currency(1234.56, symbol="€") == "€1,234.56"
        assert format_currency(1234.56, symbol="£") == "£1,234.56"
        assert format_currency(1234.56, symbol="¥") == "¥1,234.56"
        assert format_currency(1234.56, symbol="") == "1,234.56"

    def test_custom_decimal_places(self):
        """Test formatting with custom decimal places."""
        assert format_currency(1234.56, decimal_places=0) == "$1,235"
        assert format_currency(1234.56, decimal_places=1) == "$1,234.6"
        assert format_currency(1234.56, decimal_places=3) == "$1,234.560"
        assert format_currency(1234.56789, decimal_places=4) == "$1,234.5679"

    def test_custom_thousands_separator(self):
        """Test formatting with custom thousands separator."""
        assert format_currency(1234567.89, thousands_sep=".") == "$1.234.567.89"
        assert format_currency(1234567.89, thousands_sep=" ") == "$1 234 567.89"
        assert format_currency(1234567.89, thousands_sep="") == "$1234567.89"

    def test_custom_decimal_separator(self):
        """Test formatting with custom decimal separator."""
        assert format_currency(1234.56, decimal_sep=",") == "$1,234,56"
        assert format_currency(1234.56, decimal_sep=" ") == "$1,234 56"

    def test_none_input(self):
        """Test that None input is handled correctly."""
        assert format_currency(None) == ""
        assert format_currency(None, default="N/A") == "N/A"

    def test_non_numeric_input(self):
        """Test that non-numeric inputs are handled correctly."""
        with pytest.raises(TypeError):
            format_currency("not a number")
        with pytest.raises(TypeError):
            format_currency([])


class TestRoundTo:
    """Test cases for number rounding function."""

    def test_round_to_zero_places(self):
        """Test rounding to zero decimal places."""
        assert round_to(1.4, 0) == 1
        assert round_to(1.5, 0) == 2
        assert round_to(-1.5, 0) == -2
        assert round_to(0, 0) == 0

    def test_round_to_positive_places(self):
        """Test rounding to positive decimal places."""
        assert round_to(1.234, 2) == 1.23
        assert round_to(1.235, 2) == 1.24
        assert round_to(-1.235, 2) == -1.24
        assert round_to(1.2, 2) == 1.20

    def test_round_to_negative_places(self):
        """Test rounding to negative decimal places (tens, hundreds, etc.)."""
        # Test rounding to the nearest 10
        assert round_to(123, -1) == 120
        assert round_to(125, -1) == 130
        # Test rounding to the nearest 100
        assert round_to(1234, -2) == 1200
        assert round_to(1250, -2) == 1300
        assert round_to(-1250, -2) == -1300

    def test_round_decimal_type(self):
        """Test rounding Decimal objects."""
        assert round_to(Decimal("1.234"), 2) == Decimal("1.23")
        assert round_to(Decimal("1.235"), 2) == Decimal("1.24")
        assert round_to(Decimal("-1.235"), 2) == Decimal("-1.24")

    def test_return_type(self):
        """Test that the return type matches the input type."""
        assert isinstance(round_to(1.5, 0), int)
        assert isinstance(round_to(1.5, 1), float)
        assert isinstance(round_to(Decimal("1.5"), 1), Decimal)

    def test_none_input(self):
        """Test that None input is handled correctly."""
        with pytest.raises(TypeError):
            round_to(None, 2)

    def test_non_numeric_input(self):
        """Test that non-numeric inputs are handled correctly."""
        with pytest.raises(TypeError):
            round_to("not a number", 2)
        with pytest.raises(TypeError):
            round_to([], 2)


class TestRandomNumber:
    """Test cases for random number generation function."""

    def test_within_range(self):
        """Test that generated numbers are within the specified range."""
        for _ in range(100):  # Run multiple times to increase confidence
            num = random_number(1, 10)
            assert 1 <= num <= 10

    def test_min_equals_max(self):
        """Test when min equals max."""
        assert random_number(5, 5) == 5

    def test_negative_range(self):
        """Test with negative numbers in the range."""
        for _ in range(100):
            num = random_number(-10, -1)
            assert -10 <= num <= -1

    def test_mixed_range(self):
        """Test with a range that includes both negative and positive numbers."""
        for _ in range(100):
            num = random_number(-5, 5)
            assert -5 <= num <= 5

    def test_large_range(self):
        """Test with a large range."""
        for _ in range(10):
            num = random_number(-1000000, 1000000)
            assert -1000000 <= num <= 1000000

    def test_distribution(self):
        """Test that the distribution is roughly uniform."""
        # Generate a large number of random values between 1 and 10
        results = [random_number(1, 10) for _ in range(1000)]
        
        # Count occurrences of each value
        counts = {}
        for num in range(1, 11):
            counts[num] = results.count(num)
        
        # Check that each number appears roughly the expected number of times
        # (100 times each, with some tolerance for randomness)
        for num, count in counts.items():
            assert 70 <= count <= 130, f"Number {num} appeared {count} times, expected roughly 100"

    def test_invalid_range(self):
        """Test with invalid range (min > max)."""
        with pytest.raises(ValueError):
            random_number(10, 1)

    def test_non_integer_input(self):
        """Test with non-integer inputs."""
        with pytest.raises(TypeError):
            random_number(1.5, 10)
        with pytest.raises(TypeError):
            random_number(1, 10.5)
        with pytest.raises(TypeError):
            random_number("1", 10)
        with pytest.raises(TypeError):
            random_number(1, "10")
