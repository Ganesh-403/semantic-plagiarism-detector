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
    reading_ease, grade_level = get_readability_metrics(text)
    return f"**Words:** {words} | **Characters:** {chars} | **Est. Reading Time:** {time} min | **Flesch Reading Ease:** {reading_ease} | **Flesch-Kincaid Grade:** {grade_level}"


def count_syllables_in_word(word: str) -> int:
    """Estimate the syllable count of a single word using basic heuristics."""
    word = word.lower().strip()
    if not word:
        return 0
    word = "".join([c for c in word if c.isalpha()])
    if not word:
        return 0
    
    vowels = "aeiouy"
    count = 0
    is_prev_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not is_prev_vowel:
            count += 1
        is_prev_vowel = is_vowel
        
    if word.endswith("e"):
        count -= 1
        
    if count <= 0:
        count = 1
        
    return count


def get_syllable_count(text: str) -> int:
    """Return the total syllable count for the text."""
    words = re.findall(r'\w+', text)
    return sum(count_syllables_in_word(w) for w in words)


def get_sentence_count(text: str) -> int:
    """Return the total sentence count for the text."""
    sentences = [s for s in re.split(r'[.!?]+(?:\s+|$)', text) if s.strip()]
    return max(1, len(sentences)) if text.strip() else 0


def get_readability_metrics(text: str) -> tuple[float, float]:
    """Calculate Flesch Reading Ease and Flesch-Kincaid Grade Level.

    Returns (flesch_reading_ease, flesch_kincaid_grade).
    """
    words = get_word_count(text)
    sentences = get_sentence_count(text)
    syllables = get_syllable_count(text)

    if words == 0 or sentences == 0:
        return 0.0, 0.0

    reading_ease = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    grade_level = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

    return round(reading_ease, 2), round(grade_level, 2)
