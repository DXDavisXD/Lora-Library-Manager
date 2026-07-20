"""
src/providers/tests/provider_contract.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract test suite that every provider must satisfy.

Usage
-----
Subclass ``ProviderContractTests``, implement the required abstract
properties, and mix in ``unittest.TestCase``:

    class TestCivitai(ProviderContractTests, unittest.TestCase):
        @property
        def provider(self): ...

The abstract class deliberately does NOT inherit from ``unittest.TestCase``
so that the test runner does not try to execute it directly.

All tests are tagged with what they require:
  - [LIVE] — makes real HTTP requests; skipped when offline
  - [UNIT] — fully offline; uses only in-memory data

The metadata round-trip test ([ROUNDTRIP]) writes a temporary sidecar
file and reads it back through the application's own ``load_metadata``
and ``build_gallery_entry`` pipeline to verify end-to-end normalisation.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from abc import ABC, abstractmethod
from pathlib import Path


# Add src/ to sys.path so this module can be run directly from the project root
_SRC = Path(__file__).parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class ProviderContractTests(ABC):
    """
    Abstract contract that every ProviderBase implementation must satisfy.

    Concrete subclasses must implement all abstract properties.
    """

    # ── Abstract fixtures ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def provider(self):
        """Return an instance of the provider under test."""
        ...

    @property
    @abstractmethod
    def known_sha256(self) -> str:
        """A SHA-256 hash that is known to resolve to a model on this provider."""
        ...

    @property
    @abstractmethod
    def known_model_id(self) -> str:
        """A model ID that exists on this provider."""
        ...

    @property
    @abstractmethod
    def known_version_id(self) -> str:
        """A version ID that belongs to known_model_id."""
        ...

    @property
    @abstractmethod
    def search_query(self) -> str:
        """A search term expected to return at least one result."""
        ...

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_online(self) -> bool:
        """Return True if the provider's API is reachable."""
        try:
            from urllib.request import Request, urlopen
            req = Request(
                "https://arcenciel.io/api/models/search?search=test&limit=1",
                headers={"User-Agent": "LoRA-Library-Manager/2.0"},
            )
            urlopen(req, timeout=8)
            return True
        except Exception:
            return False

    def _skip_if_offline(self) -> None:  # type: ignore[return]
        """Skip the test if the network is unavailable."""
        if not self._is_online():
            self.skipTest("Network unavailable — skipping live provider test")  # type: ignore[attr-defined]

    def _assert_model_info_valid(self, info) -> None:
        """Assert that *info* is a well-formed ModelInfo with all required fields."""
        from providers.provider_base import ModelInfo
        self.assertIsInstance(info, ModelInfo)  # type: ignore[attr-defined]

        # Provider identity
        self.assertTrue(info.provider, "provider must be non-empty")  # type: ignore[attr-defined]
        self.assertTrue(info.provider_display_name, "provider_display_name must be non-empty")  # type: ignore[attr-defined]
        self.assertTrue(info.provider_model_id, "provider_model_id must be non-empty")  # type: ignore[attr-defined]
        self.assertTrue(info.provider_version_id, "provider_version_id must be non-empty")  # type: ignore[attr-defined]

        # Model metadata — names are required; others are allowed to be empty
        self.assertIsInstance(info.name, str)  # type: ignore[attr-defined]
        self.assertIsInstance(info.version_name, str)  # type: ignore[attr-defined]
        self.assertIsInstance(info.trained_words, list)  # type: ignore[attr-defined]
        self.assertIsInstance(info.preview_urls, list)  # type: ignore[attr-defined]
        self.assertIsInstance(info.tags, list)  # type: ignore[attr-defined]

        # file_sha256 must be lowercase hex if present
        if info.file_sha256:
            self.assertEqual(  # type: ignore[attr-defined]
                info.file_sha256, info.file_sha256.lower(),
                "file_sha256 must be lowercase",
            )

    # ── Contract tests ────────────────────────────────────────────────────────

    def test_capabilities_declared(self) -> None:
        """[UNIT] Provider advertises a ProviderCapabilities object."""
        from providers.provider_base import ProviderCapabilities
        caps = self.provider.capabilities
        self.assertIsInstance(caps, ProviderCapabilities)  # type: ignore[attr-defined]

    def test_provider_name_non_empty(self) -> None:
        """[UNIT] get_provider_name() returns a non-empty string."""
        name = self.provider.get_provider_name()
        self.assertIsInstance(name, str)  # type: ignore[attr-defined]
        self.assertTrue(name, "provider name must be non-empty")  # type: ignore[attr-defined]

    def test_fetch_by_hash(self) -> None:
        """[LIVE] fetch_by_hash returns a valid ModelInfo for a known SHA-256."""
        self._skip_if_offline()
        if not self.provider.capabilities.hash_lookup:
            self.skipTest("Provider does not support hash lookup")  # type: ignore[attr-defined]
        info = self.provider.fetch_by_hash(self.known_sha256)
        self.assertIsNotNone(info, "fetch_by_hash should return a ModelInfo for a known hash")  # type: ignore[attr-defined]
        self._assert_model_info_valid(info)

    def test_fetch_by_id(self) -> None:
        """[LIVE] fetch_by_id returns a valid ModelInfo for known IDs."""
        self._skip_if_offline()
        info = self.provider.fetch_by_id(self.known_model_id, self.known_version_id)
        self.assertIsNotNone(info, "fetch_by_id should return a ModelInfo for known IDs")  # type: ignore[attr-defined]
        self._assert_model_info_valid(info)
        self.assertEqual(info.provider_model_id, str(self.known_model_id))  # type: ignore[attr-defined]
        self.assertEqual(info.provider_version_id, str(self.known_version_id))  # type: ignore[attr-defined]

    def test_search_returns_list(self) -> None:
        """[LIVE] search() returns a non-empty list of ModelInfo objects."""
        self._skip_if_offline()
        if not self.provider.capabilities.search:
            self.skipTest("Provider does not support search")  # type: ignore[attr-defined]
        if self.provider.get_provider_name() == "civitai":
            try:
                from urllib.request import Request, urlopen
                req = Request(
                    f"https://civitai.com/api/v1/models?query={self.search_query}&limit=1",
                    headers={"User-Agent": "LoRA-Library-Manager/2.0"},
                )
                with urlopen(req, timeout=8):
                    pass
            except Exception as e:
                self.skipTest(f"CivitAI search query API is currently unavailable: {e}")
        results = self.provider.search(self.search_query, limit=5)
        self.assertIsInstance(results, list)  # type: ignore[attr-defined]
        self.assertGreater(len(results), 0, "search should return at least one result")  # type: ignore[attr-defined]
        for r in results:
            self._assert_model_info_valid(r)

    def test_fetch_model_versions(self) -> None:
        """[LIVE] fetch_model_versions() returns a non-empty list of ModelInfo."""
        self._skip_if_offline()
        if not self.provider.capabilities.update_check:
            self.skipTest("Provider does not support version listing")  # type: ignore[attr-defined]
        versions = self.provider.fetch_model_versions(self.known_model_id)
        self.assertIsInstance(versions, list)  # type: ignore[attr-defined]
        self.assertGreater(len(versions), 0, "should return at least one version")  # type: ignore[attr-defined]
        for v in versions:
            self._assert_model_info_valid(v)

    def test_download_url_non_empty(self) -> None:
        """[LIVE] get_download_url() returns a non-empty URL for a known version."""
        self._skip_if_offline()
        if not self.provider.capabilities.model_download:
            self.skipTest("Provider does not support model download")  # type: ignore[attr-defined]
        url = self.provider.get_download_url(self.known_model_id, self.known_version_id)
        self.assertIsInstance(url, str)  # type: ignore[attr-defined]
        self.assertTrue(url.startswith("http"), f"download URL must be an http(s) URL, got: {url!r}")  # type: ignore[attr-defined]

    def test_provider_presentation(self) -> None:
        """[LIVE] presentation property returns correct dynamic values."""
        pres = self.provider.presentation
        self.assertIsNotNone(pres)  # type: ignore[attr-defined]
        self.assertIsInstance(pres.display_name, str)  # type: ignore[attr-defined]
        self.assertIsInstance(pres.accent_color, str)  # type: ignore[attr-defined]
        self.assertTrue(pres.accent_color.startswith("#"))  # type: ignore[attr-defined]
        self.assertIsInstance(pres.icon, str)  # type: ignore[attr-defined]
        self.assertIsInstance(pres.website, str)  # type: ignore[attr-defined]
        self.assertTrue(pres.website.startswith("http"))  # type: ignore[attr-defined]

    def test_get_model_page_url(self) -> None:
        """[LIVE] get_model_page_url() returns a valid URL for known versions."""
        url = self.provider.get_model_page_url(self.known_model_id, self.known_version_id)
        self.assertIsInstance(url, str)  # type: ignore[attr-defined]
        self.assertTrue(url.startswith("http"), f"page URL must be an http(s) URL, got: {url!r}")  # type: ignore[attr-defined]
        self.assertIn(str(self.known_model_id), url)  # type: ignore[attr-defined]

    def test_preview_urls_non_empty(self) -> None:
        """[LIVE] fetch_by_id returns at least one preview URL."""
        self._skip_if_offline()
        if not self.provider.capabilities.preview_download:
            self.skipTest("Provider does not support preview download")  # type: ignore[attr-defined]
        info = self.provider.fetch_by_id(self.known_model_id, self.known_version_id)
        self.assertIsNotNone(info)  # type: ignore[attr-defined]
        self.assertGreater(len(info.preview_urls), 0, "at least one preview URL expected")  # type: ignore[attr-defined]
        for url in info.preview_urls:
            self.assertTrue(url.startswith("http"), f"preview URL must be http(s): {url!r}")  # type: ignore[attr-defined]

    def test_to_sidecar_dict_roundtrip(self) -> None:
        """[LIVE] to_sidecar_dict() produces a dict that can be serialised/deserialised as JSON."""
        self._skip_if_offline()
        info = self.provider.fetch_by_id(self.known_model_id, self.known_version_id)
        self.assertIsNotNone(info)  # type: ignore[attr-defined]

        sidecar = info.to_sidecar_dict()
        self.assertIsInstance(sidecar, dict)  # type: ignore[attr-defined]

        # Must round-trip through JSON cleanly
        serialised = json.dumps(sidecar)
        reloaded = json.loads(serialised)

        # Key normalised fields must survive the round-trip
        for field in (
            "provider", "provider_display_name",
            "provider_model_id", "provider_version_id",
            "name", "version_name", "base_model",
            "trained_words", "tags", "file_sha256",
        ):
            self.assertIn(field, reloaded, f"sidecar must contain '{field}'")  # type: ignore[attr-defined]
            self.assertEqual(  # type: ignore[attr-defined]
                reloaded[field], sidecar[field],
                f"sidecar field '{field}' must survive JSON round-trip",
            )

        # download_url must NOT be in the sidecar (it's ephemeral)
        self.assertNotIn("download_url", reloaded)  # type: ignore[attr-defined]

    def test_metadata_full_roundtrip(self) -> None:
        """
        [ROUNDTRIP] Provider → ModelInfo → sidecar JSON → load_metadata() → verify fields.

        This test writes the sidecar produced by to_sidecar_dict() next to a
        temporary placeholder .safetensors file, then reads it back through
        the application's own ``load_metadata()`` function and verifies that
        all normalised fields survive intact.

        This catches any mismatch between the field names written by
        ``to_sidecar_dict()`` and the field names expected by the gallery
        layer.
        """
        self._skip_if_offline()

        info = self.provider.fetch_by_id(self.known_model_id, self.known_version_id)
        self.assertIsNotNone(info)  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # Create a placeholder model file
            model_file = tmp / f"test_model_{info.provider_version_id}.safetensors"
            model_file.touch()

            # Write the sidecar (same stem, .json extension)
            sidecar_file = model_file.with_suffix(".json")
            sidecar_data = info.to_sidecar_dict()
            with sidecar_file.open("w", encoding="utf-8") as f:
                json.dump(sidecar_data, f, indent=2)

            # Load back through the application's own pipeline
            from gallery import load_metadata
            loaded = load_metadata(model_file)

            self.assertIsInstance(loaded, dict)  # type: ignore[attr-defined]
            self.assertTrue(loaded, "load_metadata must return non-empty dict")  # type: ignore[attr-defined]

            # Normalised fields that must survive the round-trip
            roundtrip_fields = [
                ("provider",                info.provider),
                ("provider_display_name",   info.provider_display_name),
                ("provider_model_id",       info.provider_model_id),
                ("provider_version_id",     info.provider_version_id),
                ("name",                    info.name),
                ("version_name",            info.version_name),
                ("base_model",              info.base_model),
                ("file_sha256",             info.file_sha256),
            ]
            for field, expected in roundtrip_fields:
                self.assertIn(field, loaded, f"loaded sidecar must contain '{field}'")  # type: ignore[attr-defined]
                self.assertEqual(  # type: ignore[attr-defined]
                    loaded[field], expected,
                    f"field '{field}' must survive the sidecar round-trip",
                )

            # List fields: check they are lists with the same contents
            for list_field, expected_list in [
                ("trained_words", info.trained_words),
                ("tags",          info.tags),
            ]:
                self.assertIn(list_field, loaded)  # type: ignore[attr-defined]
                self.assertEqual(sorted(loaded[list_field]), sorted(expected_list))  # type: ignore[attr-defined]

            # download_url must NOT appear in the persisted sidecar
            self.assertNotIn("download_url", loaded)  # type: ignore[attr-defined]
