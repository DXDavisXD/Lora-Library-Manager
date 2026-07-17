"""
src/providers/tests/test_arcenciel_provider.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Concrete implementation of the provider contract for Arc en Ciel.

Run with:
    cd src
    python -m pytest providers/tests/test_arcenciel_provider.py -v

Known fixtures
--------------
Model  : "Gaston18 Style (Mashima) [Anima/Illustrious/Pony]"
         arcenciel.io/models/14794

Version: Pony v3.0 — id=16788
SHA-256: 881cde5056eb1328c2de3f20c802ab0235cc36fecbdd24249b37cfda590d8fe8
AutoV3 : aab44495803a97c30d251745e7706cc5ee49b3b3bee5bd0e71bc733dca70812a

These IDs were verified against the live API on 2026-07-07 and are expected
to be stable as long as the model is not deleted.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from providers.tests.provider_contract import ProviderContractTests  # noqa: E402


class TestArcencielProviderContract(ProviderContractTests, unittest.TestCase):
    """Arc en Ciel satisfies the ProviderBase contract."""

    # ── Fixtures ──────────────────────────────────────────────────────────────
    _KNOWN_SHA256    = "881cde5056eb1328c2de3f20c802ab0235cc36fecbdd24249b37cfda590d8fe8"
    _KNOWN_MODEL_ID  = "14794"
    _KNOWN_VERSION_ID = "16788"
    _SEARCH_QUERY    = "character lora"   # 370+ results verified 2026-07-07

    @property
    def provider(self):
        from providers.arcenciel_provider import ArcencielProvider
        p = ArcencielProvider()
        p._clear_cache()
        return p

    @property
    def known_sha256(self) -> str:
        return self._KNOWN_SHA256

    @property
    def known_model_id(self) -> str:
        return self._KNOWN_MODEL_ID

    @property
    def known_version_id(self) -> str:
        return self._KNOWN_VERSION_ID

    @property
    def search_query(self) -> str:
        return self._SEARCH_QUERY

    # ── Arcenciel-specific unit tests ─────────────────────────────────────────

    def test_strip_html_no_tags(self) -> None:
        """[UNIT] _strip_html returns plain text when no tags are present."""
        from providers.arcenciel_provider import _strip_html
        self.assertEqual(_strip_html("Hello world"), "Hello world")

    def test_strip_html_removes_tags(self) -> None:
        """[UNIT] _strip_html extracts plain text from HTML."""
        from providers.arcenciel_provider import _strip_html
        result = _strip_html("<p>Hello <strong>world</strong></p>")
        self.assertIn("Hello", result)
        self.assertIn("world", result)
        self.assertNotIn("<p>", result)
        self.assertNotIn("<strong>", result)

    def test_strip_html_decodes_entities(self) -> None:
        """[UNIT] _strip_html decodes HTML entities correctly."""
        from providers.arcenciel_provider import _strip_html
        self.assertIn("&", _strip_html("Tom &amp; Jerry"))
        self.assertIn("<", _strip_html("A &lt; B"))

    def test_strip_html_empty_string(self) -> None:
        """[UNIT] _strip_html returns empty string for empty/None-like input."""
        from providers.arcenciel_provider import _strip_html
        self.assertEqual(_strip_html(""), "")

    def test_parse_activation_tags_plain(self) -> None:
        """[UNIT] _parse_activation_tags handles plain comma-delimited tags."""
        from providers.arcenciel_provider import _parse_activation_tags
        result = _parse_activation_tags(["tag1, tag2, tag3"])
        self.assertEqual(result, ["tag1", "tag2", "tag3"])

    def test_parse_activation_tags_with_labels(self) -> None:
        """[UNIT] _parse_activation_tags strips label prefixes."""
        from providers.arcenciel_provider import _parse_activation_tags
        result = _parse_activation_tags([
            "Default | tao, black hair",
            "Panagia | panag, headgear",
        ])
        self.assertIn("tao", result)
        self.assertIn("black hair", result)
        self.assertIn("panag", result)
        self.assertIn("headgear", result)
        self.assertNotIn("Default", result)
        self.assertNotIn("Panagia", result)

    def test_parse_activation_tags_deduplicates(self) -> None:
        """[UNIT] _parse_activation_tags deduplicates across label groups."""
        from providers.arcenciel_provider import _parse_activation_tags
        result = _parse_activation_tags([
            "Group A | long hair, black hair",
            "Group B | long hair, blue eyes",
        ])
        self.assertEqual(result.count("long hair"), 1, "long hair must appear only once")

    def test_model_cache_populated_after_fetch(self) -> None:
        """[LIVE] _fetch_model populates the cache and avoids a second HTTP call."""
        self._skip_if_offline()
        p = self.provider
        p._clear_cache()

        # First call — populates cache
        m1 = p._fetch_model(self._KNOWN_MODEL_ID)
        if m1 is None:
            self.skipTest(  # type: ignore[attr-defined]
                "Arcenciel /api/models endpoint returned None — transient API failure"
            )
        self.assertIn(self._KNOWN_MODEL_ID, p._model_cache)

        # Second call — must come from cache (we can verify by checking object identity)
        m2 = p._fetch_model(self._KNOWN_MODEL_ID)
        self.assertIs(m1, m2, "second fetch should return the cached object")  # type: ignore[attr-defined]

    def test_fetch_by_autov3_hash(self) -> None:
        """[LIVE] fetch_by_hash resolves a model via its AutoV3 (sha256webui) hash."""
        self._skip_if_offline()
        autov3 = "aab44495803a97c30d251745e7706cc5ee49b3b3bee5bd0e71bc733dca70812a"
        info = self.provider.fetch_by_hash(autov3)
        self.assertIsNotNone(info, "should find a model by AutoV3 hash")
        self._assert_model_info_valid(info)

    def test_download_url_is_cdn_path(self) -> None:
        """[LIVE] get_download_url returns a working /api/models/.../versions/.../download URL.

        The old /uploads/ CDN path returns 404; Arcenciel now serves downloads
        via GET /api/models/{model_id}/versions/{version_id}/download.
        """
        self._skip_if_offline()
        url = self.provider.get_download_url(self._KNOWN_MODEL_ID, self._KNOWN_VERSION_ID)
        self.assertTrue(url.startswith("https://arcenciel.io/api/"), f"unexpected URL: {url!r}")
        self.assertIn("/versions/", url, "URL must contain /versions/")
        self.assertIn("/download", url, "URL must end with /download")

    def test_get_download_url_requires_model_id(self) -> None:
        """[UNIT] get_download_url returns empty string when model_id is None."""
        p = self.provider
        result = p.get_download_url(None, self._KNOWN_VERSION_ID)
        self.assertEqual(result, "")

    def test_provider_name(self) -> None:
        """[UNIT] get_provider_name() returns 'arcenciel'."""
        self.assertEqual(self.provider.get_provider_name(), "arcenciel")

    def test_provider_display_name_in_model_info(self) -> None:
        """[LIVE] ModelInfo.provider_display_name is 'Arc en Ciel'."""
        self._skip_if_offline()
        info = self.provider.fetch_by_id(self._KNOWN_MODEL_ID, self._KNOWN_VERSION_ID)
        self.assertIsNotNone(info)
        self.assertEqual(info.provider_display_name, "Arc en Ciel")


if __name__ == "__main__":
    unittest.main()
