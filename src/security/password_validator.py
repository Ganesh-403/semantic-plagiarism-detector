import re

from zxcvbn import zxcvbn

def validate_password_complexity(password: str) -> bool:
    """
    Validate that a password meets complexity rules.
    Rules:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one number
    - At least one special character
    """
    if not isinstance(password, str):
        raise ValueError("Password must be a string.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain at least one uppercase letter.")
        
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain at least one lowercase letter.")
        
    if not re.search(r"\d", password):
        raise ValueError("Password must contain at least one number.")
        
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        raise ValueError("Password must contain at least one special character.")
        
    # zxcvbn accepts up to 72 characters; additional suffix characters can
    # only increase strength and need not be sent to the estimator.
    if zxcvbn(password[:72])["score"] < 3:
        raise ValueError("Password is too weak or common. Choose a longer, unpredictable password.")
    return True
