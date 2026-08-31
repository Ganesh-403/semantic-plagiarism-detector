import pytest

from src.db.auth import _validate_password_complexity, generate_secure_password


def test_validate_password_complexity_rejects_common_passwords():
    """Verify that common dictionary and weak passwords like Password123! are rejected."""
    common_passwords = [
        "Password123!",
        "Admin123!",
        "Welcome123!",
        "Password@1",
        "Qwerty123456!",
        "12345678Abc!",
    ]
    for pwd in common_passwords:
        with pytest.raises(ValueError, match="weak|common|character|letter|number"):
            _validate_password_complexity(pwd)


def test_validate_password_complexity_accepts_strong_passwords():
    """Verify that high-entropy, strong passwords pass complexity validation."""
    strong_passwords = [
        "Correct-Horse-Battery-Staple-99!",
        "SecureP@ssw0rd2026!",
        "X#9mP$2vL&8qR!x4",
        "K9#vL2$mP8&xR!q4",
    ]
    for pwd in strong_passwords:
        assert _validate_password_complexity(pwd) == pwd


def test_validate_password_complexity_length_and_classes():
    """Verify that missing character classes or short lengths are rejected."""
    with pytest.raises(ValueError, match="at least 8 characters"):
        _validate_password_complexity("Ab1!")

    with pytest.raises(ValueError, match="uppercase"):
        _validate_password_complexity("lowercase123!@#")

    with pytest.raises(ValueError, match="number"):
        _validate_password_complexity("NoNumbersHere!@#")

    with pytest.raises(ValueError, match="special character"):
        _validate_password_complexity("NoSpecialChars123")


def test_generate_secure_password_satisfies_complexity():
    """Verify that generated passwords always satisfy the complexity rules."""
    for _ in range(5):
        pwd = generate_secure_password(32)
        assert _validate_password_complexity(pwd) == pwd


if __name__ == "__main__":
    test_validate_password_complexity_rejects_common_passwords()
    test_validate_password_complexity_accepts_strong_passwords()
    test_validate_password_complexity_length_and_classes()
    test_generate_secure_password_satisfies_complexity()
    print("ALL PASSWORD COMPLEXITY TESTS PASSED!")

