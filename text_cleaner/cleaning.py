import re

def remove_tags(text: str) -> str:
    """
    Removes remnant wiki-style and HTML-like tags, such as <templatestyles>
    or other tags that BeautifulSoup might have missed.
    This uses a regular expression to find and remove any text enclosed in <...>.
    """
    return re.sub(r'<[^>]*?>', '', text)

def remove_extra_whitespace(text: str) -> str:
    """
    Removes extra whitespace, including multiple spaces, tabs, and newlines.
    - Strips leading/trailing whitespace.
    - Replaces multiple newlines with a single newline.
    - Replaces multiple spaces/tabs with a single space.
    """
    text = text.strip()
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text

def remove_repeated_title(text: str, title: str) -> str:
    """
    Removes the title if it's repeated as the first line of the text.
    Many articles start with the title, which is redundant since we
    already have it in our metadata.
    """
    first_line = text.split('\n', 1)[0].strip()
    if first_line == title:
        return text.split('\n', 1)[1] if '\n' in text else ""
    return text

def filter_by_ratio(text: str, threshold: float = 0.8) -> bool:
    """
    Checks if the ratio of Malayalam characters in the text is above a threshold.

    Args:
        text: The text to analyze.
        threshold: The minimum required ratio of Malayalam characters (0.0 to 1.0).

    Returns:
        True if the text passes the filter (is mostly Malayalam), False otherwise.
    """
    if not text:
        return False
    
    # The Unicode range for Malayalam characters is U+0D00 to U+0D7F
    malayalam_chars = 0
    total_chars = 0
    
    for char in text:
        # Ignore spaces and newlines when calculating the ratio
        if not char.isspace():
            total_chars += 1
            if '\u0D00' <= char <= '\u0D7F':
                malayalam_chars += 1
    
    if total_chars == 0:
        return False
        
    ratio = malayalam_chars / total_chars
    return ratio >= threshold

def filter_by_word_count(text: str, min_words: int = 5) -> bool:
    """
    Checks if the text meets a minimum word count.

    Args:
        text: The text to analyze.
        min_words: The minimum number of words required.

    Returns:
        True if the text has enough words, False otherwise.
    """
    word_count = len(text.split())
    return word_count >= min_words

def strict_mal_chars(text: str) -> str:
    """
    Removes all characters that are not in the Malayalam Unicode range,
    digits, or basic punctuation. This is an aggressive cleaning step.
    
    Keeps:
    - Malayalam characters (U+0D00 to U+0D7F)
    - Whitespace characters (space, newline, tab)
    - Digits (0-9)
    - Common punctuation (. , ? ! -)
    """

    return re.sub(r'[^\u0D00-\u0D7F\s\d.,?!-]', '', text)