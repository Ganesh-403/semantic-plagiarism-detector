# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
tests/core/test_encoding_clean_text.py
--------------------------------------
Regression tests covering the guarantee that ``normalize_encoding()`` never
damages text that is already correctly encoded.

The mojibake table used to contain two single-character entries, ``"Â": ""``
and ``"Ã": "Ã"``. The first deleted every U+00C2 in the document, so ordinary
French, Portuguese and Romanian words lost letters ("Âme" became "me"); the
second was a no-op that only widened the compiled alternation. These tests pin
down the behaviour that replaced them:

* a "Â" is only removed when the character after it proves the pair was a
  UTF-8 two-byte sequence,
* no table key is short enough to match a lone letter,
* repair is a fixed point, so running it twice changes nothing.
"""

import pytest

from src.core.encoding import MOJIBAKE_REPLACEMENTS, detect_mojibake, normalize_encoding

# Real words from Latin-script languages that legitimately contain the
# characters the old table treated as garbage.
CLEAN_TEXTS_WITH_A_CIRCUMFLEX = [
    "Âme",
    "Ângela",
    "Ângelo Moreira",
    "Un Âne gris",
    "Â propos de rien",
    "Îmi place România",
    "Ancien Régime: l'Âme française",
    "Ângela e Ântonio foram à praça.",
]

CLEAN_TEXTS_WITH_A_TILDE = [
    "Ã bientôt",
    "São Paulo",
    "Ãnimo",
]


class TestCleanTextIsPreserved:
    """Correctly encoded text must survive normalization untouched."""

    @pytest.mark.parametrize("text", CLEAN_TEXTS_WITH_A_CIRCUMFLEX)
    def test_capital_a_circumflex_is_not_deleted(self, text):
        """A lone Â is a letter, not a mojibake artifact."""
        assert normalize_encoding(text) == text

    @pytest.mark.parametrize("text", CLEAN_TEXTS_WITH_A_TILDE)
    def test_capital_a_tilde_is_not_rewritten(self, text):
        """A lone Ã is a letter and must pass through unchanged."""
        assert normalize_encoding(text) == text

    def test_accented_prose_round_trips_unchanged(self):
        """A paragraph of ordinary accented prose is left exactly as-is."""
        text = (
            "L'Âme du poète résonne à travers les siècles. "
            "Ângela Ferreira écrivait déjà en 1893, à Coimbra, "
            "que « l'être humain » cherche naïvement l'idéal."
        )
        assert normalize_encoding(text) == text

    def test_ascii_text_is_unchanged(self):
        """Plain ASCII has nothing to repair."""
        text = "The quick brown fox jumps over the lazy dog."
        assert normalize_encoding(text) == text

    def test_cjk_text_is_unchanged(self):
        """Non-Latin scripts are outside the Windows-1252 corruption path."""
        text = "这是一个测试文档。テストです。"
        assert normalize_encoding(text) == text

    def test_emoji_text_is_unchanged(self):
        """Astral-plane characters are not representable in cp1252."""
        text = "Submitted on time 🎉 with a full bibliography 📚"
        assert normalize_encoding(text) == text


class TestContextualPrefixRemoval:
    """The Â prefix is stripped only when the next character justifies it."""

    @pytest.mark.parametrize(
        "garbled,expected",
        [
            ("Â© 2024 University Press", "© 2024 University Press"),
            ("Â£40 per term", "£40 per term"),
            ("30Â° north", "30° north"),
            ("Â«citationÂ»", "«citation»"),
            ("Section Â§4.2", "Section §4.2"),
            ("100Â€", "100€"),
        ],
    )
    def test_prefix_before_symbol_is_removed(self, garbled, expected):
        """Â followed by a Latin-1 symbol is a genuine two-byte artifact."""
        assert normalize_encoding(garbled) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Âme",
            "Âb",
            "Âz",
            "Â9",
            "Â.",
            "Â",
        ],
    )
    def test_prefix_before_plain_character_is_kept(self, text):
        """Â followed by an ordinary character is real text."""
        assert normalize_encoding(text) == text


class TestTableIntegrity:
    """Structural guarantees about the generated replacement table."""

    def test_no_single_character_keys(self):
        """A one-character key would match ordinary letters and corrupt them."""
        short_keys = [key for key in MOJIBAKE_REPLACEMENTS if len(key) < 2]
        assert short_keys == []

    def test_no_identity_mappings(self):
        """An entry that maps a sequence to itself only costs match time."""
        identities = {
            key: value for key, value in MOJIBAKE_REPLACEMENTS.items() if key == value
        }
        assert identities == {}

    def test_every_replacement_shortens_the_text(self):
        """Each repair must shrink the string, which is what bounds the loop."""
        for key, value in MOJIBAKE_REPLACEMENTS.items():
            assert len(value) < len(key), f"{key!r} -> {value!r} does not shorten"

    def test_covers_the_common_accents(self):
        """Spot-check that the generated table did not come out empty-handed."""
        for garbled, expected in [
            ("Ã©", "é"),
            ("Ã¯", "ï"),
            ("Ã±", "ñ"),
            ("ÃŸ", "ß"),
            ("Ã¸", "ø"),
        ]:
            assert MOJIBAKE_REPLACEMENTS[garbled] == expected


class TestIdempotence:
    """Repair must reach a fixed point, as the docstring promises."""

    @pytest.mark.parametrize(
        "text",
        [
            "The cafÃ© was beautifÃ»l.",
            "Âme et Ângela",
            "He said, â€œHello!â€",
            "Â© 2024",
            "EspaÃ±ol y portuguÃªs",
            "naÃ¯ve",
            "plain ascii",
            "",
        ],
    )
    def test_second_pass_is_a_no_op(self, text):
        """f(f(x)) == f(x) for garbled, clean and empty input alike."""
        once = normalize_encoding(text)
        assert normalize_encoding(once) == once

    def test_repeated_application_converges(self):
        """Ten passes give the same answer as one."""
        text = "The cafÃ© on RÃ©publique serves crÃ¨me brÃ»lÃ©e."
        result = normalize_encoding(text)

        repeated = text
        for _ in range(10):
            repeated = normalize_encoding(repeated)

        assert repeated == result


class TestDetectMojibakeOnCleanText:
    """Detection must not flag ordinary accented documents."""

    @pytest.mark.parametrize(
        "text",
        CLEAN_TEXTS_WITH_A_CIRCUMFLEX + CLEAN_TEXTS_WITH_A_TILDE,
    )
    def test_clean_latin_text_is_not_flagged(self, text):
        """A document full of Â and Ã letters is not mojibake."""
        assert detect_mojibake(text) is False

    def test_genuinely_garbled_text_is_flagged(self):
        """The heuristic still fires on real garbling."""
        text = "The cafÃ© on RÃ©publique serves crÃ¨me brÃ»lÃ©e and Ã¼ber-dÃ¶ner."
        assert detect_mojibake(text) is True

    def test_symbol_prefix_counts_toward_the_ratio(self):
        """Contextual Â artifacts are mojibake and should count."""
        assert detect_mojibake("Â©Â£Â°", threshold=0.5) is True
