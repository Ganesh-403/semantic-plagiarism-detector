"""Translation utility for cross-lingual plagiarism alignment."""

from __future__ import annotations

import logging
import os
import re
import time

from deep_translator import DeeplTranslator, GoogleTranslator

logger = logging.getLogger(__name__)

# In-memory pipeline cache for MarianMT models: (source_lang, target_lang) -> pipeline
_MARIAN_PIPELINES: dict[tuple[str, str], object] = {}

# Comprehensive ISO-639-1 (and ISO-639-2) language code database
ISO_639_LANGUAGES: dict[str, dict[str, str]] = {
    "aa": {"name": "Afar", "native": "Afaraf"},
    "ab": {"name": "Abkhazian", "native": "аҧсуа бызшәа"},
    "ae": {"name": "Avestan", "native": "avesta"},
    "af": {"name": "Afrikaans", "native": "Afrikaans"},
    "ak": {"name": "Akan", "native": "Akan"},
    "am": {"name": "Amharic", "native": "አማርኛ"},
    "an": {"name": "Aragonese", "native": "aragonés"},
    "ar": {"name": "Arabic", "native": "العربية"},
    "as": {"name": "Assamese", "native": "অসমীয়া"},
    "av": {"name": "Avaric", "native": "авар мацӀ"},
    "ay": {"name": "Aymara", "native": "aymar aru"},
    "az": {"name": "Azerbaijani", "native": "azərbaycan dili"},
    "ba": {"name": "Bashkir", "native": "башҡорт теле"},
    "be": {"name": "Belarusian", "native": "беларуская мова"},
    "bg": {"name": "Bulgarian", "native": "български език"},
    "bh": {"name": "Bihari languages", "native": "भोजपुरी"},
    "bi": {"name": "Bislama", "native": "Bislama"},
    "bm": {"name": "Bambara", "native": "bamanankan"},
    "bn": {"name": "Bengali", "native": "বাংলা"},
    "bo": {"name": "Tibetan", "native": "bod skad"},
    "br": {"name": "Breton", "native": "brezhoneg"},
    "bs": {"name": "Bosnian", "native": "bosanski jezik"},
    "ca": {"name": "Catalan", "native": "català"},
    "ce": {"name": "Chechen", "native": "нохчийн мотт"},
    "ch": {"name": "Chamorro", "native": "Chamoru"},
    "co": {"name": "Corsican", "native": "corsu"},
    "cr": {"name": "Cree", "native": "ᓀᐦᐃᔭᐍᐏᐣ"},
    "cs": {"name": "Czech", "native": "čeština"},
    "cu": {"name": "Church Slavic", "native": "ѩзыкъ словѣньскъ"},
    "cv": {"name": "Chuvash", "native": "чӑваш чӗлхи"},
    "cy": {"name": "Welsh", "native": "Cymraeg"},
    "da": {"name": "Danish", "native": "dansk"},
    "de": {"name": "German", "native": "Deutsch"},
    "dv": {"name": "Divehi", "native": "Dhivehi"},
    "dz": {"name": "Dzongkha", "native": "རྫོང་ཁ"},
    "ee": {"name": "Ewe", "native": "Eʋegbe"},
    "el": {"name": "Greek", "native": "Ελληνικά"},
    "en": {"name": "English", "native": "English"},
    "eo": {"name": "Esperanto", "native": "Esperanto"},
    "es": {"name": "Spanish", "native": "Español"},
    "et": {"name": "Estonian", "native": "eesti"},
    "eu": {"name": "Basque", "native": "euskara"},
    "fa": {"name": "Persian", "native": "فارسی"},
    "ff": {"name": "Fulah", "native": "Fulfulde"},
    "fi": {"name": "Finnish", "native": "suomi"},
    "fj": {"name": "Fijian", "native": "vosa Vakaviti"},
    "fo": {"name": "Faroese", "native": "føroyskt"},
    "fr": {"name": "French", "native": "Français"},
    "fy": {"name": "Western Frisian", "native": "Frysk"},
    "ga": {"name": "Irish", "native": "Gaeilge"},
    "gd": {"name": "Gaelic", "native": "Gàidhlig"},
    "gl": {"name": "Galician", "native": "galego"},
    "gn": {"name": "Guarani", "native": "Avañe'ẽ"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી"},
    "gv": {"name": "Manx", "native": "Gaelg"},
    "ha": {"name": "Hausa", "native": "هَوُسَ"},
    "he": {"name": "Hebrew", "native": "עברית"},
    "hi": {"name": "Hindi", "native": "हिन्दी"},
    "ho": {"name": "Hiri Motu", "native": "Hiri Motu"},
    "hr": {"name": "Croatian", "native": "hrvatski jezik"},
    "ht": {"name": "Haitian Creole", "native": "Kreyòl ayisyen"},
    "hu": {"name": "Hungarian", "native": "magyar"},
    "hy": {"name": "Armenian", "native": "Հայերեն"},
    "hz": {"name": "Herero", "native": "Otjiherero"},
    "ia": {"name": "Interlingua", "native": "Interlingua"},
    "id": {"name": "Indonesian", "native": "Bahasa Indonesia"},
    "ie": {"name": "Interlingue", "native": "Interlingue"},
    "ig": {"name": "Igbo", "native": "Asụsụ Igbo"},
    "ii": {"name": "Sichuan Yi", "native": "ꆈꌠ꒿ Nuosuhxop"},
    "ik": {"name": "Inupiaq", "native": "Iñupiaq"},
    "io": {"name": "Ido", "native": "Ido"},
    "is": {"name": "Icelandic", "native": "Íslenska"},
    "it": {"name": "Italian", "native": "Italiano"},
    "iu": {"name": "Inuktitut", "native": "ᐃᓄᒃᑎᑐᑦ"},
    "ja": {"name": "Japanese", "native": "日本語"},
    "jv": {"name": "Javanese", "native": "basa Jawa"},
    "ka": {"name": "Georgian", "native": "ქართული"},
    "kg": {"name": "Kongo", "native": "Kikongo"},
    "ki": {"name": "Kikuyu", "native": "Gĩkũyũ"},
    "kj": {"name": "Kuanyama", "native": "Kuanyama"},
    "kk": {"name": "Kazakh", "native": "қазақ тілі"},
    "kl": {"name": "Kalaallisut", "native": "kalaallisut"},
    "km": {"name": "Central Khmer", "native": "ភាសាខ្មែរ"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ"},
    "ko": {"name": "Korean", "native": "한국어"},
    "kr": {"name": "Kanuri", "native": "Kanuri"},
    "ks": {"name": "Kashmiri", "native": "کأشُر"},
    "ku": {"name": "Kurdish", "native": "Kurdî"},
    "kv": {"name": "Komi", "native": "коми кыв"},
    "kw": {"name": "Cornish", "native": "Kernewek"},
    "ky": {"name": "Kirghiz", "native": "Кыргызча"},
    "la": {"name": "Latin", "native": "Latine"},
    "lb": {"name": "Luxembourgish", "native": "Lëtzebuergesch"},
    "lg": {"name": "Ganda", "native": "Luganda"},
    "li": {"name": "Limburgish", "native": "Limburgs"},
    "ln": {"name": "Lingala", "native": "Lingála"},
    "lo": {"name": "Lao", "native": "ພາສາລາວ"},
    "lt": {"name": "Lithuanian", "native": "lietuvių kalba"},
    "lu": {"name": "Luba-Katanga", "native": "Tshiluba"},
    "lv": {"name": "Latvian", "native": "latviešu valoda"},
    "mg": {"name": "Malagasy", "native": "fiteny malagasy"},
    "mh": {"name": "Marshallese", "native": "Kajin M̧ajeļ"},
    "mi": {"name": "Maori", "native": "te reo Māori"},
    "mk": {"name": "Macedonian", "native": "македонски јазик"},
    "ml": {"name": "Malayalam", "native": "മലയാളം"},
    "mn": {"name": "Mongolian", "native": "Монгол хэл"},
    "mr": {"name": "Marathi", "native": "मराठी"},
    "ms": {"name": "Malay", "native": "Bahasa Melayu"},
    "mt": {"name": "Maltese", "native": "Malti"},
    "my": {"name": "Burmese", "native": "ဗမာစာ"},
    "na": {"name": "Nauru", "native": "Dorerin Naoero"},
    "nb": {"name": "Bokmål, Norwegian", "native": "Norsk Bokmål"},
    "nd": {"name": "North Ndebele", "native": "isiNdebele"},
    "ne": {"name": "Nepali", "native": "नेपाली"},
    "ng": {"name": "Ndonga", "native": "Owambo"},
    "nl": {"name": "Dutch", "native": "Nederlands"},
    "nn": {"name": "Norwegian Nynorsk", "native": "Norsk Nynorsk"},
    "no": {"name": "Norwegian", "native": "Norsk"},
    "nr": {"name": "South Ndebele", "native": "isiNdebele"},
    "nv": {"name": "Navajo", "native": "Diné bizaad"},
    "ny": {"name": "Chichewa", "native": "chiCheŵa"},
    "oc": {"name": "Occitan", "native": "occitan"},
    "oj": {"name": "Ojibwa", "native": "ᐊᓂShakespeare ᓂᓈᐯᒧᐎᓐ"},
    "om": {"name": "Oromo", "native": "Afaan Oromoo"},
    "or": {"name": "Oriya", "native": "ଓଡ଼ିଆ"},
    "os": {"name": "Ossetian", "native": "ирон æвзаг"},
    "pa": {"name": "Panjabi", "native": "ਪੰਜਾਬੀ"},
    "pi": {"name": "Pali", "native": "पालि"},
    "pl": {"name": "Polish", "native": "Język polski"},
    "ps": {"name": "Pashto", "native": "پښتو"},
    "pt": {"name": "Portuguese", "native": "Português"},
    "qu": {"name": "Quechua", "native": "Runa Simi"},
    "rm": {"name": "Romansh", "native": "rumantsch grischun"},
    "rn": {"name": "Rundi", "native": "Ikirundi"},
    "ro": {"name": "Romanian", "native": "Română"},
    "ru": {"name": "Russian", "native": "Русский"},
    "rw": {"name": "Kinyarwanda", "native": "Ikinyarwanda"},
    "sa": {"name": "Sanskrit", "native": "संस्कृतम्"},
    "sc": {"name": "Sardinian", "native": "sardu"},
    "sd": {"name": "Sindhi", "native": "سنڌي"},
    "se": {"name": "Northern Sami", "native": "Davvisámegiella"},
    "sg": {"name": "Sango", "native": "yângâ tî sängö"},
    "si": {"name": "Sinhala", "native": "සිංහල"},
    "sk": {"name": "Slovak", "native": "slovenčina"},
    "sl": {"name": "Slovenian", "native": "slovenski jezik"},
    "sm": {"name": "Samoan", "native": "gagana fa'a Samoa"},
    "sn": {"name": "Shona", "native": "chiShona"},
    "so": {"name": "Somali", "native": "Soomaaliga"},
    "sq": {"name": "Albanian", "native": "Shqip"},
    "sr": {"name": "Serbian", "native": "српски језик"},
    "ss": {"name": "Swati", "native": "SiSwati"},
    "st": {"name": "Southern Sotho", "native": "Sesotho"},
    "su": {"name": "Sundanese", "native": "Basa Sunda"},
    "sv": {"name": "Swedish", "native": "Svenska"},
    "sw": {"name": "Swahili", "native": "Kiswahili"},
    "ta": {"name": "Tamil", "native": "தமிழ்"},
    "te": {"name": "Telugu", "native": "తెలుగు"},
    "tg": {"name": "Tajik", "native": "тоҷикӣ"},
    "th": {"name": "Thai", "native": "ไทย"},
    "ti": {"name": "Tigrinya", "native": "ትግርኛ"},
    "tk": {"name": "Turkmen", "native": "Türkmen"},
    "tl": {"name": "Tagalog", "native": "Wikang Tagalog"},
    "tn": {"name": "Tswana", "native": "Setswana"},
    "to": {"name": "Tonga", "native": "faka Tonga"},
    "tr": {"name": "Turkish", "native": "Türkçe"},
    "ts": {"name": "Tsonga", "native": "Xitsonga"},
    "tt": {"name": "Tatar", "native": "татар теле"},
    "tw": {"name": "Twi", "native": "Twi"},
    "ty": {"name": "Tahitian", "native": "Reo Tahiti"},
    "ug": {"name": "Uighur", "native": "ئۇيغۇرچە"},
    "uk": {"name": "Ukrainian", "native": "Українська"},
    "ur": {"name": "Urdu", "native": "اردو"},
    "uz": {"name": "Uzbek", "native": "Oʻzbekcha"},
    "ve": {"name": "Venda", "native": "Tshivenenda"},
    "vi": {"name": "Vietnamese", "native": "Tiếng Việt"},
    "vo": {"name": "Volapük", "native": "Volapük"},
    "wa": {"name": "Walloon", "native": "walon"},
    "wo": {"name": "Wolof", "native": "Wollof"},
    "xh": {"name": "Xhosa", "native": "isiXhosa"},
    "yi": {"name": "Yiddish", "native": "ייִדיש"},
    "yo": {"name": "Yoruba", "native": "Yorùbá"},
    "za": {"name": "Zhuang", "native": "Saɯcueŋƅ"},
    "zh": {"name": "Chinese", "native": "中文"},
    "zu": {"name": "Zulu", "native": "isiZulu"},
}

# Simple lookup map from ISO code to English name
LANGUAGE_NAME_MAP: dict[str, str] = {
    code: info["name"] for code, info in ISO_639_LANGUAGES.items()
}


def _batch_sentences(text: str, max_chars: int = 4500) -> list[str]:
    """Helper to split long text into smaller batches by sentence boundaries."""
    # Split by punctuation followed by space (basic sentence boundary)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    batches = []
    current_batch = ""
    
    for sentence in sentences:
        if len(current_batch) + len(sentence) + 1 <= max_chars:
            current_batch += (sentence + " ").lstrip()
        else:
            if current_batch:
                batches.append(current_batch.strip())
            current_batch = sentence + " "
            
    if current_batch.strip():
        batches.append(current_batch.strip())
        
    # Safety net: If a single sentence without punctuation is still longer than max_chars
    final_batches = []
    for batch in batches:
        if len(batch) > max_chars:
            for i in range(0, len(batch), max_chars):
                final_batches.append(batch[i:i + max_chars])
        else:
            final_batches.append(batch)

    return final_batches


def translate_text_marian(
    text: str,
    source_lang: str = "es",
    target_lang: str = "en",
) -> str:
    """Translate text offline using local HuggingFace MarianMT (Helsinki-NLP/opus-mt) models.

    Args:
        text: Input text string to translate.
        source_lang: Source language ISO code (e.g., 'es', 'fr', 'de').
        target_lang: Target language ISO code (e.g., 'en').

    Returns:
        Translated text string, or error marker on model loading failure.
    """
    if not text or not str(text).strip():
        return str(text or "")

    src = (source_lang or "es").strip().lower()
    tgt = (target_lang or "en").strip().lower()
    if src == "auto":
        src = "es"

    cache_key = (src, tgt)
    pipe = _MARIAN_PIPELINES.get(cache_key)

    if pipe is None:
        try:
            from transformers import pipeline

            model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
            pipe = pipeline("translation", model=model_name)
            _MARIAN_PIPELINES[cache_key] = pipe
        except Exception as exc:
            logger.error("Failed to load MarianMT model for %s->%s: %s", src, tgt, exc)
            return f"(Translation Error: MarianMT {exc})"

    try:
        results = pipe(str(text))
        if isinstance(results, list) and len(results) > 0 and "translation_text" in results[0]:
            return str(results[0]["translation_text"]).strip()
        return str(results).strip()
    except Exception as exc:
        logger.error("MarianMT translation execution error: %s", exc)
        return f"(Translation Error: {exc})"


def translate_text(
    text: str | None,
    target_lang: str = "en",
    source_lang: str = "auto",
) -> str | None:
    """Translate text while preserving the repository's public API.

    Compatibility guarantees:
    - ``None`` returns ``None``.
    - An empty string returns an empty string.
    - Provider/configuration failures return a human-readable string containing
      ``"Translation Error"``.

    The cross-lingual preprocessing layer detects that error prefix and falls
    back to the original source text before embedding, so error messages never
    contaminate FAISS vectors.
    """
    if text is None:
        return None

    original = str(text)
    if not original.strip():
        return original

    # Chunking logic for texts exceeding API limits
    if len(original) > 4500:
        batches = _batch_sentences(original, max_chars=4500)
        translated_batches = []
        for batch in batches:
            chunk_trans = translate_text(batch, target_lang=target_lang, source_lang=source_lang)
            if chunk_trans and str(chunk_trans).startswith("(Translation Error:"):
                # If any chunk fails with a network/translation error, return the error
                return chunk_trans
            translated_batches.append(chunk_trans or "")
        return " ".join(translated_batches).strip()

    # Check for offline translation via HuggingFace MarianMT (#3988)
    if os.getenv("OFFLINE_TRANSLATION_ENABLED", "").lower() in ("true", "1", "yes"):
        res = translate_text_marian(original, source_lang=source_lang, target_lang=target_lang)
        return res

    deepl_api_key = os.getenv("DEEPL_API_KEY")
    translated = None

    if deepl_api_key:
        try:
            # DeeplTranslator uses api_key parameter
            deepl_source = "auto" if not source_lang or source_lang == "auto" else source_lang
            translated = DeeplTranslator(
                api_key=deepl_api_key,
                source=deepl_source,
                target=target_lang,
            ).translate(original)
        except Exception as exc:
            logger.warning("DeepL translation failed, falling back to GoogleTranslator: %s", exc)
            translated = None

    if translated is None:
        max_retries = 3
        last_exc = None

        for attempt in range(max_retries):
            try:
                translated = GoogleTranslator(
                    source=source_lang or "auto",
                    target=target_lang,
                ).translate(original)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    sleep_time = 2**attempt  # 1s, 2s, 4s
                    logger.warning(
                        "Translation failed on attempt %d/%d (%s). Retrying in %ds...",
                        attempt + 1,
                        max_retries,
                        exc,
                        sleep_time,
                    )
                    time.sleep(sleep_time)

        if last_exc and translated is None:
            return f"(Translation Error: {last_exc})"

    translated = str(translated or "").strip()
    if not translated:
        return f"(Translation Error: empty response for target '{target_lang}')"

    return translated


def translate_text_secondary(
    text: str,
    target_lang: str = "en",
    source_lang: str = "auto",
) -> str:
    """Fallback translation service using an offline/secondary translator.

    If MyMemoryTranslator is available, it uses it. Swaps to a mock
    offline fallback translation string on connection/provider failures.
    """
    if not text:
        return ""

    try:
        from deep_translator import MyMemoryTranslator
        translated = MyMemoryTranslator(
            source=source_lang or "auto",
            target=target_lang,
        ).translate(text)
        if translated:
            return str(translated).strip()
    except Exception:
        pass

    return f"[Offline Fallback -> {target_lang}]: {text}"


def translate_text_batch(
    texts: list[str],
    target_lang: str = "en",
    source_lang: str = "auto",
) -> list[str]:
    """Translate a batch list of text strings.

    Args:
        texts: List of strings to translate.
        target_lang: Target language code.
        source_lang: Source language code.

    Returns:
        List of translated strings.
    """
    if not texts:
        return []

    validate_target_language_code(target_lang)

    deepl_api_key = os.getenv("DEEPL_API_KEY")
    if deepl_api_key:
        try:
            deepl_source = "auto" if not source_lang or source_lang == "auto" else source_lang
            translated = DeeplTranslator(
                api_key=deepl_api_key,
                source=deepl_source,
                target=target_lang,
            ).translate_batch(texts)
            return [str(t or "").strip() for t in translated]
        except Exception as exc:
            logger.warning("DeepL batch translation failed, falling back to GoogleTranslator: %s", exc)

    try:
        translated = GoogleTranslator(
            source=source_lang or "auto",
            target=target_lang,
        ).translate_batch(texts)
        return [str(t or "").strip() for t in translated]
    except Exception as exc:
        logger.error("Batch translation error: %s", exc)
        # Fallback to translate_text individually for resilience
        results = []
        for text in texts:
            try:
                res = translate_text(text, target_lang=target_lang, source_lang=source_lang)
                results.append(res or "")
            except Exception:
                results.append(f"(Translation Error: {exc})")
        return results


def get_language_name(code: str) -> str:
    """Convert a two-letter ISO-639-1 language code to a human-readable language name.

    Args:
        code: ISO-639-1 language code string (e.g., 'en', 'es', 'fr', 'de').

    Returns:
        Human-readable language name if found, or uppercase code if unknown.
    """
    if not code or not isinstance(code, str):
        return ""
    normalized = code.strip().lower()
    return LANGUAGE_NAME_MAP.get(normalized, code.strip().upper())


def get_language_native_name(code: str) -> str:
    """Convert an ISO-639-1 language code to its native language name.

    Args:
        code: ISO-639-1 language code string.

    Returns:
        Native language name if found in registry, or uppercase code as fallback.
    """
    if not code or not isinstance(code, str):
        return ""
    normalized = code.strip().lower()
    info = ISO_639_LANGUAGES.get(normalized)
    if info and "native" in info:
        return info["native"]
    return code.strip().upper()


def get_language_info(code: str) -> dict[str, str] | None:
    """Fetch complete metadata for a specified ISO-639-1 language code.

    Args:
        code: ISO-639-1 language code string.

    Returns:
        Dictionary containing 'name' and 'native' keys, or None if unknown.
    """
    if not code or not isinstance(code, str):
        return None
    normalized = code.strip().lower()
    info = ISO_639_LANGUAGES.get(normalized)
    if info:
        return {"code": normalized, "name": info["name"], "native": info["native"]}
    return None


def is_valid_language_code(code: str) -> bool:
    """Check whether a given language code is recognized in the ISO-639-1 database.

    Args:
        code: Language code to check.

    Returns:
        True if valid code, False otherwise.
    """
    if not code or not isinstance(code, str):
        return False
    return code.strip().lower() in ISO_639_LANGUAGES


def validate_target_language_code(lang_code: str) -> bool:
    """Validate a target language code against the ISO-639-1 language code set.

    Args:
        lang_code: Target language code to validate (case-insensitive).

    Returns:
        True if the code is a supported ISO-639-1 language code.

    Raises:
        ValueError: If the language code is not a supported ISO-639-1 code.
    """
    if (
        not isinstance(lang_code, str)
        or lang_code.strip().lower() not in ISO_639_LANGUAGES
    ):
        raise ValueError(f"Unsupported target language code: {lang_code}")
    return True


def get_supported_language_codes() -> list[str]:
    """Return a sorted list of all supported ISO-639-1 language codes.

    Returns:
        Sorted list of two-letter language code strings.
    """
    return sorted(ISO_639_LANGUAGES.keys())


def get_all_languages() -> dict[str, str]:
    """Return a copy of the complete language code to English name mapping.

    Returns:
        Dictionary mapping ISO codes to human-readable names.
    """
    return dict(LANGUAGE_NAME_MAP)


def normalize_language_code(code: str) -> str:
    """Normalize a raw language string into a clean lowercase ISO code.

    Args:
        code: Raw language input string.

    Returns:
        Cleaned lowercase language code or 'en' fallback if empty.
    """
    if not code or not isinstance(code, str):
        return "en"
    clean = code.strip().lower()
    base = clean.split("-")[0].split("_")[0]
    if base in ISO_639_LANGUAGES:
        return base
    return clean or "en"


def search_languages_by_name(query: str) -> list[tuple[str, str]]:
    """Search for languages matching a name substring.

    Args:
        query: Query string to match against English or native language names.

    Returns:
        List of tuples (iso_code, english_name).
    """
    if not query or not isinstance(query, str):
        return []
    q = query.strip().lower()
    matches = []
    for code, info in ISO_639_LANGUAGES.items():
        if q in info["name"].lower() or q in info["native"].lower() or q == code:
            matches.append((code, info["name"]))
    return sorted(matches, key=lambda item: item[1])


def batch_convert_language_codes(codes: list[str]) -> dict[str, str]:
    """Convert a batch list of ISO language codes to English language names.

    Args:
        codes: List of language code strings.

    Returns:
        Dictionary mapping input code to human-readable language name.
    """
    if not codes or not isinstance(codes, list):
        return {}
    results = {}
    for code in codes:
        if isinstance(code, str):
            results[code] = get_language_name(code)
    return results


def format_language_display(code: str, include_native: bool = True) -> str:
    """Format a language code for display in user interface components.

    Args:
        code: ISO-639-1 language code.
        include_native: Whether to append native name in parentheses if available.

    Returns:
        Formatted display string, e.g. "Spanish (Español)" or "Spanish".
    """
    if not code or not isinstance(code, str):
        return ""
    name = get_language_name(code)
    if not include_native:
        return name
    native = get_language_native_name(code)
    if native and native.lower() != name.lower() and native != code.upper():
        return f"{name} ({native})"
    return name


def get_language_display_name(code: str) -> str:
    """
    Map ISO-639-1 code to full language name (e.g. 'de' -> 'German').
    Return uppercase code string if language is unmapped.

    Args:
        code: ISO-639-1 language code.

    Returns:
        Full language name or uppercase code.
    """
    if not code or not isinstance(code, str):
        return ""
    normalized = code.strip().lower()
    return LANGUAGE_NAME_MAP.get(normalized, code.strip().upper())


def get_common_translation_pairs() -> list[tuple[str, str]]:
    """Return a list of primary cross-lingual translation language code pairs.

    Returns:
        List of tuples containing (source_code, target_code).
    """
    return [
        ("es", "en"),
        ("fr", "en"),
        ("de", "en"),
        ("zh", "en"),
        ("ja", "en"),
        ("ru", "en"),
        ("ar", "en"),
        ("hi", "en"),
        ("pt", "en"),
        ("it", "en"),
    ]
# Helper function verified for issue 3993
