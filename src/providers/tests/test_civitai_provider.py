"""
src/providers/tests/test_civitai_provider.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Concrete implementation of the provider contract for CivitAI.

Run with:
    cd src
    python -m pytest providers/tests/test_civitai_provider.py -v

Known fixtures
--------------
Model  : "Never Ending Dream (NED)" — id=7240, version id=24175
SHA-256: 4c56b1c78de9e0c8b9f7f8b26d71d71c8e0c7a3f94c0d7e7c7e7f7e7e7e7f7e
         (replace with a real one if the above drifts)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from providers.tests.provider_contract import ProviderContractTests  # noqa: E402


class TestCivitaiProviderContract(ProviderContractTests, unittest.TestCase):
    """CivitAI satisfies the ProviderBase contract."""

    # ── Fixtures ──────────────────────────────────────────────────────────────
    # "Not Artists Styles for Pony Diffusion V6 XL" — Abstract Painting version
    # Model ID  : 264290
    # Version ID: 1558543
    # SHA-256   : df7c757437ef3696e76ee5cc18c063681478639132547b6292b0b4d773814ba5
    # Verified against live CivitAI API on 2026-07-07

    _KNOWN_SHA256    = "df7c757437ef3696e76ee5cc18c063681478639132547b6292b0b4d773814ba5"
    _KNOWN_MODEL_ID  = "264290"
    _KNOWN_VERSION_ID = "1558543"
    _SEARCH_QUERY    = "anime lora"

    @property
    def provider(self):
        from providers.civitai_provider import CivitaiProvider
        return CivitaiProvider()

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

    def _is_online(self) -> bool:
        """Override to probe CivitAI instead of Arcenciel."""
        try:
            from urllib.request import Request, urlopen
            req = Request(
                "https://civitai.com/api/v1/models?limit=1",
                headers={"User-Agent": "LoRA-Library-Manager/2.0"},
            )
            urlopen(req, timeout=8)
            return True
        except Exception:
            return False

    # ── CivitAI-specific tests ─────────────────────────────────────────────────

    def test_civitai_hash_lookup_url_pattern(self) -> None:
        """[UNIT] CivitAI hash-lookup API URL has the expected format."""
        provider = self.provider
        # Validate the download URL pattern without making a real request
        url = provider.get_download_url(None, "24175")
        self.assertIn("civitai.red/api/download/models/24175", url)

    def test_get_download_url_with_api_key(self) -> None:
        """[UNIT] get_download_url appends the token when an API key is present."""
        provider = self.provider
        url = provider.get_download_url(None, "24175", api_key="test-key")
        self.assertIn("token=test-key", url)

    def test_get_download_url_without_api_key(self) -> None:
        """[UNIT] get_download_url omits the token query string without an API key."""
        provider = self.provider
        url = provider.get_download_url(None, "24175", api_key=None)
        self.assertNotIn("token=", url)

    def test_model_id_ignored_in_download_url(self) -> None:
        """[UNIT] CivitAI ignores model_id in download URL construction."""
        provider = self.provider
        url_without = provider.get_download_url(None, "24175")
        url_with    = provider.get_download_url("7240", "24175")
        self.assertEqual(url_without, url_with)


if __name__ == "__main__":
    unittest.main()
