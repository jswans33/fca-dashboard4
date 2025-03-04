"""Tests for string utility functions."""
import pytest

from fca_dashboard.utils.string_utils import capitalize, is_empty, slugify, truncate


class TestCapitalize:
    """Tests for the capitalize function."""

    def test_capitalize_lowercase(self):
        """Test capitalizing a lowercase string."""
        assert capitalize("hello") == "Hello"

    def test_capitalize_already_capitalized(self):
        """Test capitalizing an already capitalized string."""
        assert capitalize("Hello") == "Hello"

    def test_capitalize_empty_string(self):
        """Test capitalizing an empty string."""
        assert capitalize("") == ""

    def test_capitalize_single_char(self):
        """Test capitalizing a single character."""
        assert capitalize("a") == "A"

    def test_capitalize_with_spaces(self):
        """Test capitalizing a string with leading spaces."""
        assert capitalize("  hello") == "  Hello"

    def test_capitalize_with_numbers(self):
        """Test capitalizing a string starting with numbers."""
        assert capitalize("123abc") == "123abc"


class TestSlugify:
    """Tests for the slugify function."""

    def test_slugify_simple_string(self):
        """Test slugifying a simple string."""
        assert slugify("Hello World") == "hello-world"

    def test_slugify_with_special_chars(self):
        """Test slugifying a string with special characters."""
        assert slugify("Hello, World!") == "hello-world"

    def test_slugify_with_multiple_spaces(self):
        """Test slugifying a string with multiple spaces."""
        assert slugify("Hello   World") == "hello-world"

    def test_slugify_with_dashes(self):
        """Test slugifying a string that already has dashes."""
        assert slugify("Hello-World") == "hello-world"

    def test_slugify_with_underscores(self):
        """Test slugifying a string with underscores."""
        assert slugify("Hello_World") == "hello-world"

    def test_slugify_empty_string(self):
        """Test slugifying an empty string."""
        assert slugify("") == ""

    def test_slugify_with_accents(self):
        """Test slugifying a string with accented characters."""
        assert slugify("Héllö Wörld") == "hello-world"


class TestTruncate:
    """Tests for the truncate function."""

    def test_truncate_short_string(self):
        """Test truncating a string shorter than the limit."""
        assert truncate("Hello", 10) == "Hello"

    def test_truncate_exact_length(self):
        """Test truncating a string of exact length."""
        assert truncate("Hello", 5) == "Hello"

    def test_truncate_long_string(self):
        """Test truncating a string longer than the limit."""
        assert truncate("Hello World", 5) == "Hello..."

    def test_truncate_with_custom_suffix(self):
        """Test truncating with a custom suffix."""
        assert truncate("Hello World", 5, suffix="...more") == "Hello...more"

    def test_truncate_empty_string(self):
        """Test truncating an empty string."""
        assert truncate("", 5) == ""

    def test_truncate_with_zero_length(self):
        """Test truncating with zero length."""
        assert truncate("Hello", 0) == "..."


class TestIsEmpty:
    """Tests for the is_empty function."""

    def test_is_empty_with_empty_string(self):
        """Test checking if an empty string is empty."""
        assert is_empty("") is True

    def test_is_empty_with_whitespace(self):
        """Test checking if a whitespace string is empty."""
        assert is_empty("   ") is True
        assert is_empty("\t\n") is True

    def test_is_empty_with_text(self):
        """Test checking if a non-empty string is empty."""
        assert is_empty("Hello") is False

    def test_is_empty_with_whitespace_and_text(self):
        """Test checking if a string with whitespace and text is empty."""
        assert is_empty("  Hello  ") is False

    def test_is_empty_with_none(self):
        """Test checking if None is empty."""
        with pytest.raises(TypeError):
            is_empty(None)
