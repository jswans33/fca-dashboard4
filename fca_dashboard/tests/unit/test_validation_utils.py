"""Unit tests for validation utilities."""

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
            
    def test_email_with_consecutive_dots(self):
        """Test email validation with consecutive dots."""
        # This specifically tests the '..' in email condition (line 35)
        email_with_consecutive_dots = "user@example..com"
        assert not is_valid_email(email_with_consecutive_dots), "Email with consecutive dots should be invalid"
        
    def test_email_with_trailing_dot(self):
        """Test email validation with trailing dot."""
        # This specifically tests the email.endswith('.') condition (line 37)
        email_with_trailing_dot = "user@example.com."
        assert not is_valid_email(email_with_trailing_dot), "Email with trailing dot should be invalid"
        
    def test_email_with_spaces(self):
        """Test email validation with spaces."""
        # This specifically tests the ' ' in email condition (line 39)
        email_with_spaces = "user name@example.com"
        assert not is_valid_email(email_with_spaces), "Email with spaces should be invalid"
        
    def test_email_domain_with_hyphens(self):
        """Test email validation with domain containing hyphens."""
        # This specifically tests the domain.startswith('-') or domain.endswith('-') condition (line 43)
        email_with_leading_hyphen = "user@-example.com"
        assert not is_valid_email(email_with_leading_hyphen), "Email with domain starting with hyphen should be invalid"
        
        email_with_trailing_hyphen = "user@example-.com"
        assert not is_valid_email(email_with_trailing_hyphen), "Email with domain ending with hyphen should be invalid"
            
    def test_domain_parts_with_hyphens(self):
        """Test email validation with domain parts containing hyphens."""
        # Test domain parts ending with hyphens (should be invalid)
        assert not is_valid_email("user@sub-.example.com"), \
            "Email with domain part ending with hyphen should be invalid"
        
        # Test the all() function in the return statement
        assert is_valid_email("user@valid-subdomain.example.com"), "Email with valid domain parts should be valid"

    def test_none_input(self):
        """Test that None input is handled correctly."""
        assert not is_valid_email(None), "None should be invalid"

    def test_non_string_input(self):
        """Test that non-string inputs are handled correctly."""
        assert not is_valid_email(123), "Integer should be invalid"
        assert not is_valid_email(True), "Boolean should be invalid"
        assert not is_valid_email([]), "List should be invalid"


class TestEmailValidationSpecificCases:
    """Specific test cases covering individual 'return False' conditions."""

    def test_non_string_input(self):
        assert not is_valid_email(123), "Should return False when input is not a string"

    def test_empty_string_input(self):
        assert not is_valid_email(""), "Empty string should be invalid"

    def test_missing_at_symbol(self):
        assert not is_valid_email("userexample.com"), "Email missing '@' should be invalid"

    def test_missing_domain_part(self):
        assert not is_valid_email("user@"), "Email missing domain should be invalid"

    def test_missing_username_part(self):
        assert not is_valid_email("@example.com"), "Email missing username should be invalid"

    def test_consecutive_dots_in_email(self):
        assert not is_valid_email("user..name@example.com"), "Email with consecutive dots should be invalid"

    def test_trailing_dot(self):
        assert not is_valid_email("user@example.com."), "Email with trailing dot should be invalid"

    def test_spaces_in_email(self):
        assert not is_valid_email("user name@example.com"), "Email with spaces should be invalid"

    def test_domain_starts_with_hyphen(self):
        assert not is_valid_email("user@-example.com"), "Domain starting with hyphen should be invalid"

    def test_domain_ends_with_hyphen(self):
        assert not is_valid_email("user@example-.com"), "Email domain ending with hyphen should be invalid"

    def test_domain_part_ends_with_hyphen(self):
        assert not is_valid_email("user@sub-.example.com"), "Domain part ending with hyphen should be invalid"


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
            
    def test_international_phone_formats(self):
        """Test various international phone number formats."""
        # Valid international formats
        assert is_valid_phone("+12345678901"), "International number with 11 digits should be valid"
        assert is_valid_phone("+44 20 1234 5678"), "UK format should be valid"
        
        # Invalid international formats
        assert not is_valid_phone("+1234"), "International number that's too short should be invalid"
        assert not is_valid_phone("+abcdefghijk"), "International number with letters should be invalid"
        
        # Test international number with non-digit characters after the plus
        phone_with_letters = "+abc1234567890"
        assert not is_valid_phone(phone_with_letters), "International number with letters after + should be invalid"

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
