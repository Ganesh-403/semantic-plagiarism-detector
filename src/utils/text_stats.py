import re


def get_word_count(text: str) -> int:
    return len(re.findall(r'\w+', text))

def get_char_count(text: str) -> int:
    return len(text)

def get_reading_time_minutes(text: str) -> int:
    # Average reading speed is roughly 200-250 words per minute.
    # We'll use 200 for a conservative estimate.
    word_count = get_word_count(text)
    minutes = max(1, round(word_count / 200))
    return minutes

def format_text_stats(text: str) -> str:
    words = get_word_count(text)
    chars = get_char_count(text)
    time = get_reading_time_minutes(text)
    return f"**Words:** {words} | **Characters:** {chars} | **Est. Reading Time:** {time} min"
