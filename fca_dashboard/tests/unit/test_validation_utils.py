"""Unit tests for validation utilities."""
import pytest

from fca_dashboard.utils.validation_utils import is_valid_email, is_valid_phone, is_valid_url


class TestEmailValidation:
    """Test cases for email validation function."""

    def test_valid_emails(self):
        """Test that valid email addresses are correctly identified."""
        valid_emails = [
            "user@example.com",
            "user.name@example.com",
            "user+tag@example.com",
            "user-name@example.co.uk",
            "user_name@example-domain.com",
            "123456@example.com",
            "user@subdomain.example.com",
        ]
        for email in valid_emails:
            assert is_valid_email(email), f"Email should be valid: {email}"

    def test_invalid_emails(self):
        """Test that invalid email addresses are correctly rejected."""
        invalid_emails = [
            "",  # Empty string
            "user",  # Missing @ and domain
            "user@",  # Missing domain
            "@example.com",  # Missing username
            "user@.com",  # Missing domain name
            "user@example",  # Missing TLD
            "user@example..com",  # Double dot
            "user@example.com.",  # Trailing dot
            "user name@example.com",  # Space in username
            "user@exam ple.com",  # Space in domain
            "user@-example.com",  # Domain starts with hyphen
            "user@example-.com",  # Domain ends with hyphen
        ]
        for email in invalid_emails:
            assert not is_valid_email(email), f"Email should be invalid: {email}"

    def test_none_input(self):
        """Test that None input is handled correctly."""
        assert not is_valid_email(None), "None should be invalid"

    def test_non_string_input(self):
        """Test that non-string inputs are handled correctly."""
        assert not is_valid_email(123), "Integer should be invalid"
        assert not is_valid_email(True), "Boolean should be invalid"
        assert not is_valid_email([]), "List should be invalid"


class TestPhoneValidation:
    """Test cases for phone number validation function."""

    def test_valid_phone_numbers(self):
        """Test that valid phone numbers are correctly identified."""
        valid_phones = [
            "1234567890",  # Simple 10-digit
            "123-456-7890",  # Hyphenated
            "(123) 456-7890",  # Parentheses
            "+1 123-456-7890",  # International format
            "123.456.7890",  # Dots
            "123 456 7890",  # Spaces
            "+12345678901",  # International without separators
        ]
        for phone in valid_phones:
            assert is_valid_phone(phone), f"Phone should be valid: {phone}"

    def test_invalid_phone_numbers(self):
        """Test that invalid phone numbers are correctly rejected."""
        invalid_phones = [
            "",  # Empty string
            "123",  # Too short
            "123456",  # Too short
            "abcdefghij",  # Letters
            "123-abc-7890",  # Mixed letters and numbers
            "123-456-789",  # Too short with separators
            "123-456-78901",  # Too long with separators
            "(123)456-7890",  # Missing space after parentheses
            "123 - 456 - 7890",  # Spaces around hyphens
        ]
        for phone in invalid_phones:
            assert not is_valid_phone(phone), f"Phone should be invalid: {phone}"

    def test_none_input(self):
        """Test that None input is handled correctly."""
        assert not is_valid_phone(None), "None should be invalid"

    def test_non_string_input(self):
        """Test that non-string inputs are handled correctly."""
        assert not is_valid_phone(123), "Integer should be invalid"
        assert not is_valid_phone(True), "Boolean should be invalid"
        assert not is_valid_phone([]), "List should be invalid"


class TestUrlValidation:
    """Test cases for URL validation function."""

    def test_valid_urls(self):
        """Test that valid URLs are correctly identified."""
        valid_urls = [
            "http://example.com",
            "https://example.com",
            "http://www.example.com",
            "https://example.com/path",
            "https://example.com/path?query=value",
            "https://example.com/path#fragment",
            "https://example.com:8080",
            "https://subdomain.example.com",
            "https://example-domain.com",
            "https://example.co.uk",
            "http://localhost",
            "http://localhost:8080",
            "http://127.0.0.1",
            "http://127.0.0.1:8080",
        ]
        for url in valid_urls:
            assert is_valid_url(url), f"URL should be valid: {url}"

    def test_invalid_urls(self):
        """Test that invalid URLs are correctly rejected."""
        invalid_urls = [
            "",  # Empty string
            "example.com",  # Missing protocol
            "http://",  # Missing domain
            "http:/example.com",  # Missing slash
            "http://example",  # Missing TLD
            "http://.com",  # Missing domain name
            "http://example..com",  # Double dot
            "http://example.com.",  # Trailing dot
            "http://exam ple.com",  # Space in domain
            "http://-example.com",  # Domain starts with hyphen
            "http://example-.com",  # Domain ends with hyphen
            "htp://example.com",  # Typo in protocol
            "http:example.com",  # Missing slashes
            "http//example.com",  # Missing colon
        ]
        for url in invalid_urls:
            assert not is_valid_url(url), f"URL should be invalid: {url}"

    def test_none_input(self):
        """Test that None input is handled correctly."""
        assert not is_valid_url(None), "None should be invalid"

    def test_non_string_input(self):
        """Test that non-string inputs are handled correctly."""
        assert not is_valid_url(123), "Integer should be invalid"
        assert not is_valid_url(True), "Boolean should be invalid"
        assert not is_valid_url([]), "List should be invalid"
