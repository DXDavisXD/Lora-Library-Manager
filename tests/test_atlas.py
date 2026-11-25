"""Test suite for Lora Atlas.

Every test builds a throwaway library on disk and points the app at it, so
nothing here touches a real LoRA folder or reaches the network.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas import backup, config, db, enrich, hashing, query, scanner, sidecar  # noqa: E402
from atlas.providers.base import ModelInfo, RateLimiter                        # noqa: E402


class LibraryFixture(unittest.TestCase):
    """Creates a temporary library and rebinds config to it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "lora"
        self.root.mkdir(parents=True)

        self._config_path = Path(self._tmp.name) / "atlas.json"
        self._saved_config_path = config._CONFIG_PATH
        config._CONFIG_PATH = self._config_path
        config.set_many({"library_root": str(self.root)})

        db._initialised.clear()
        db.close_thread_connections()
        self.conn = db.connect()

    def tearDown(self) -> None:
        db.close_thread_connections()
        db._initialised.clear()
        config._CONFIG_PATH = self._saved_config_path
        self._tmp.cleanup()

    # -- helpers ------------------------------------------------------------

    def make_model(self, relative: str, content: bytes = b"weights",
                   preview: bool = False, meta: dict | None = None) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if preview:
            path.with_suffix(".png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        if meta is not None:
            path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
        return path

    def rescan(self) -> scanner.ScanResult:
        return scanner.sync(self.conn, self.root)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class TestScanner(LibraryFixture):

    def test_finds_models_and_ignores_other_files(self):
        self.make_model("a/one.safetensors")
        self.make_model("b/two.ckpt")
        (self.root / "notes.txt").write_text("ignore me", encoding="utf-8")

        result = self.rescan()
        self.assertEqual(result.added, 2)
        rows = self.conn.execute("SELECT path FROM model_file ORDER BY path").fetchall()
        self.assertEqual([r["path"] for r in rows], ["a/one.safetensors", "b/two.ckpt"])

    def test_second_scan_reports_unchanged(self):
        self.make_model("a/one.safetensors")
        self.rescan()
        result = self.rescan()
        self.assertEqual((result.added, result.modified, result.unchanged), (0, 0, 1))

    def test_modified_file_invalidates_hash(self):
        path = self.make_model("a/one.safetensors", b"first")
        self.rescan()
        hashing.hash_pending(self.conn, self.root)
        before = self.conn.execute("SELECT sha256 FROM model_file").fetchone()["sha256"]
        self.assertIsNotNone(before)

        os.utime(path, (0, 0))
        path.write_bytes(b"second content, different length")
        result = self.rescan()

        self.assertEqual(result.modified, 1)
        row = self.conn.execute("SELECT sha256, hash_state FROM model_file").fetchone()
        self.assertIsNone(row["sha256"], "a changed file must not keep its old hash")
        self.assertEqual(row["hash_state"], "pending")

    def test_deleted_file_is_marked_missing_not_removed(self):
        path = self.make_model("a/one.safetensors")
        self.rescan()
        path.unlink()
        result = self.rescan()
        self.assertEqual(result.missing, 1)
        row = self.conn.execute("SELECT present FROM model_file").fetchone()
        self.assertEqual(row["present"], 0)

    def test_state_directory_is_skipped(self):
        state = self.root / config.STATE_DIRNAME
        state.mkdir(exist_ok=True)
        (state / "decoy.safetensors").write_bytes(b"x")
        self.make_model("real.safetensors")
        result = self.rescan()
        self.assertEqual(result.added, 1)

    def test_preview_matched_by_stem(self):
        self.make_model("a/one.safetensors", preview=True)
        self.rescan()
        row = self.conn.execute("SELECT path FROM preview").fetchone()
        self.assertEqual(row["path"], "a/one.png")

    def test_generic_preview_only_when_folder_has_one_model(self):
        self.make_model("solo/model.safetensors")
        (self.root / "solo" / "preview.png").write_bytes(b"img")
        self.make_model("shared/a.safetensors")
        self.make_model("shared/b.safetensors")
        (self.root / "shared" / "preview.png").write_bytes(b"img")

        self.rescan()
        solo = self.conn.execute(
            "SELECT COUNT(*) c FROM preview p JOIN model_file f ON f.id=p.file_id "
            "WHERE f.path LIKE 'solo/%'").fetchone()["c"]
        shared = self.conn.execute(
            "SELECT COUNT(*) c FROM preview p JOIN model_file f ON f.id=p.file_id "
            "WHERE f.path LIKE 'shared/%'").fetchone()["c"]
        self.assertEqual(solo, 1)
        self.assertEqual(shared, 0, "an ambiguous preview must not be attached to both models")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

class TestHashing(LibraryFixture):

    def test_hash_matches_hashlib(self):
        import hashlib
        payload = b"some model bytes" * 1000
        path = self.make_model("a.safetensors", payload)
        self.assertEqual(hashing.hash_file(path), hashlib.sha256(payload).hexdigest())

    def test_hash_pending_marks_done(self):
        self.make_model("a.safetensors", b"one")
        self.make_model("b.safetensors", b"two")
        self.rescan()
        self.assertEqual(hashing.pending_count(self.conn), 2)
        self.assertEqual(hashing.hash_pending(self.conn, self.root), 2)
        self.assertEqual(hashing.pending_count(self.conn), 0)

    def test_duplicates_grouped_by_content_not_name(self):
        self.make_model("x/same.safetensors", b"identical")
        self.make_model("y/renamed.safetensors", b"identical")
        self.make_model("z/other.safetensors", b"different")
        self.rescan()
        hashing.hash_pending(self.conn, self.root)

        groups = hashing.find_duplicates(self.conn)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["files"]), 2)
        self.assertEqual(groups[0]["reclaimable"], len(b"identical"))

    def test_cancellation_stops_hashing(self):
        path = self.make_model("big.safetensors", b"x" * (1024 * 64))
        self.assertIsNone(hashing.hash_file(path, cancel=lambda: True))


# ---------------------------------------------------------------------------
# Sidecars
# ---------------------------------------------------------------------------

class TestSidecar(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self, text: str):
        path = self.dir / "x.json"
        path.write_text(text, encoding="utf-8")
        return sidecar.load_lenient(path)

    def test_clean_json(self):
        data, repair = self._load('{"a": 1}')
        self.assertEqual(data, {"a": 1})
        self.assertIsNone(repair)

    def test_trailing_comma_repaired(self):
        data, repair = self._load('{"a": 1, "b": 2,}')
        self.assertEqual(data, {"a": 1, "b": 2})
        self.assertIsNotNone(repair)

    def test_missing_comma_repaired(self):
        data, repair = self._load('{\n"a": "x"\n"b": "y"\n}')
        self.assertEqual(data, {"a": "x", "b": "y"})
        self.assertIsNotNone(repair)

    def test_truncated_file_recovers_leading_object(self):
        data, repair = self._load('{"a": 1}{"b": 2')
        self.assertEqual(data, {"a": 1})
        self.assertEqual(repair, "truncated tail discarded")

    def test_unparseable_reports_failure(self):
        data, repair = self._load("not json at all")
        self.assertIsNone(data)
        self.assertEqual(repair, "unparseable")

    def test_atomic_write_leaves_no_temp_files(self):
        target = self.dir / "out.json"
        sidecar.write_atomic(target, {"k": "v"})
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"k": "v"})
        self.assertEqual(list(self.dir.glob("*.tmp")), [])
        self.assertEqual(list(self.dir.glob(".atlas-*")), [])

    def test_normalise_reads_webui_activation_text(self):
        # A flat activation-text string is one group, not split into words.
        fields = sidecar.normalise(
            {"activation text": "trigger one, trigger two", "preferred weight": 0.8,
             "sd version": "SDXL"},
            "model.safetensors")
        self.assertEqual(fields["trained_words"], ["trigger one, trigger two"])
        self.assertEqual(fields["preferred_weight"], 0.8)

    def test_normalise_reads_previous_manager_flat_format(self):
        """The previous app's own export: flat provider_* / file_sha256 keys.

        Missing this graded a fully-described model 'partial' for lack of
        provenance, even though the ids were right there in the sidecar.
        """
        fields = sidecar.normalise({
            "provider": "civitai",
            "provider_model_id": "2402420",
            "provider_version_id": "2701215",
            "name": "Sasha Necron",
            "version_name": "v1.0",
            "creator": "KurutAI",
            "base_model": "Illustrious",
            "trained_words": ["MGNSasha", "long hair"],
            "file_sha256": "7F94525784D14CA1",
            "negative_prompt": "blurry",
        }, "MGNSasha_ILXL.safetensors")
        self.assertEqual(fields["provider"], "civitai")
        self.assertEqual(fields["provider_model_id"], "2402420")
        self.assertEqual(fields["provider_version_id"], "2701215")
        self.assertEqual(fields["sha256"], "7f94525784d14ca1")
        self.assertEqual(fields["trained_words"], ["MGNSasha", "long hair"])
        self.assertEqual(sidecar.completeness(fields), "complete")

    def test_trigger_groups_are_preserved_not_split(self):
        """CivitAI ships grouped triggers; each comma-joined group is one chip.

        And when both a grouped list and a flat activation-text string are
        present, only the groups survive — never both.
        """
        fields = sidecar.normalise({
            "trainedWords": ["long hair, blue eyes, sailor uniform",
                             "twin tails, school uniform"],
            "activation text": "long hair, blue eyes, sailor uniform, twin tails, school uniform",
        }, "m.safetensors")
        self.assertEqual(fields["trained_words"],
                         ["long hair, blue eyes, sailor uniform", "twin tails, school uniform"])

    def test_flat_activation_text_becomes_one_group(self):
        # No structured list, so the flat string is kept whole as a single
        # group chip rather than exploded into loose words.
        fields = sidecar.normalise(
            {"activation text": "one, two, three"}, "m.safetensors")
        self.assertEqual(fields["trained_words"], ["one, two, three"])

    def test_normalise_reads_civitai_shape(self):
        fields = sidecar.normalise({
            "id": 999, "modelId": 111, "name": "v2",
            "model": {"name": "Cool LoRA", "creator": {"username": "someone"}},
            "baseModel": "Illustrious",
            "trainedWords": ["magic"],
            "files": [{"primary": True, "hashes": {"SHA256": "ABCDEF"}}],
        }, "model.safetensors")
        self.assertEqual(fields["name"], "Cool LoRA")
        self.assertEqual(fields["creator"], "someone")
        self.assertEqual(fields["base_model"], "Illustrious")
        self.assertEqual(fields["sha256"], "abcdef")
        self.assertEqual(fields["provider"], "civitai")

    def test_html_is_stripped_from_description(self):
        self.assertEqual(
            sidecar.clean_text("<p>Hello <b>world</b></p>"), "Hello world")

    def test_metadata_sidecar_is_rich_and_preserves_user_keys(self):
        model = self.dir / "m.safetensors"
        model.write_bytes(b"x")
        # Pre-existing file with the user's own curation.
        (self.dir / "m.json").write_text(json.dumps({
            "custom_field": "keep me",
            "activation text": "my curated triggers",
            "notes": "my note",
            "preferred weight": 0.7,
        }), encoding="utf-8")

        sidecar.write_metadata_sidecar(
            model,
            {"provider": "civitai", "provider_version_id": "9", "name": "Cool",
             "base_model": "SDXL", "trained_words": ["official group"], "sha256": "ab"},
            raw={"description": "a long description", "images": [{"url": "x"}], "stats": {"d": 1}},
        )
        data = json.loads((self.dir / "m.json").read_text(encoding="utf-8"))
        # Rich: the full provider payload is present.
        self.assertEqual(data["description"], "a long description")
        self.assertIn("images", data)
        self.assertIn("stats", data)
        self.assertEqual(data["provider_version_id"], "9")
        self.assertEqual(data["file_sha256"], "ab")
        # User-owned keys are untouched — never overwritten by a refresh.
        self.assertEqual(data["custom_field"], "keep me")
        self.assertEqual(data["activation text"], "my curated triggers")
        self.assertEqual(data["notes"], "my note")
        self.assertEqual(data["preferred weight"], 0.7)

    def test_retire_legacy_atlas_sidecar(self):
        """The old two-file design left a thin <stem>.atlas.json behind.

        It holds only machine-written provenance, so retiring it is safe — but
        anything the main file is missing is folded in first.
        """
        model = self.dir / "m.safetensors"
        model.write_bytes(b"x")
        (self.dir / "m.json").write_text(json.dumps({
            "name": "Existing name",
            "activation text": "my curated triggers",
        }), encoding="utf-8")
        (self.dir / "m.atlas.json").write_text(json.dumps({
            "_atlas_provider": "civitai",
            "_atlas_model_id": "111",
            "_atlas_version_id": "222",
            "name": "Should not overwrite",
            "versionName": "v2",
            "sha256": "abc",
        }), encoding="utf-8")

        summary = sidecar.retire_legacy_atlas(model)
        self.assertIsNotNone(summary)
        self.assertTrue(summary["removed"])
        self.assertFalse((self.dir / "m.atlas.json").exists())

        main = json.loads((self.dir / "m.json").read_text(encoding="utf-8"))
        # Gaps filled...
        self.assertEqual(main["provider"], "civitai")
        self.assertEqual(main["provider_model_id"], "111")
        self.assertEqual(main["provider_version_id"], "222")
        self.assertEqual(main["version_name"], "v2")
        self.assertEqual(main["file_sha256"], "abc")
        # ...but existing values and user-owned keys untouched.
        self.assertEqual(main["name"], "Existing name")
        self.assertEqual(main["activation text"], "my curated triggers")

    def test_retire_legacy_atlas_is_a_noop_without_one(self):
        model = self.dir / "n.safetensors"
        model.write_bytes(b"x")
        self.assertIsNone(sidecar.retire_legacy_atlas(model))

    def test_retire_legacy_atlas_dry_run_keeps_the_file(self):
        model = self.dir / "d.safetensors"
        model.write_bytes(b"x")
        (self.dir / "d.atlas.json").write_text('{"_atlas_provider": "civitai"}',
                                               encoding="utf-8")
        summary = sidecar.retire_legacy_atlas(model, dry_run=True)
        self.assertFalse(summary["removed"])
        self.assertTrue((self.dir / "d.atlas.json").exists())

    def test_writing_metadata_retires_the_legacy_file(self):
        """A metadata refresh should tidy the old file as a side effect."""
        model = self.dir / "r.safetensors"
        model.write_bytes(b"x")
        (self.dir / "r.atlas.json").write_text('{"_atlas_provider": "civitai"}',
                                               encoding="utf-8")
        sidecar.write_metadata_sidecar(
            model, {"name": "N", "base_model": "SDXL", "trained_words": []}, raw={})
        self.assertFalse((self.dir / "r.atlas.json").exists())
        self.assertTrue((self.dir / "r.json").exists())

    def test_metadata_sidecar_never_creates_activation_text(self):
        """A fresh download must leave the WebUI's own field empty.

        Seeding it from the official trigger list makes a provider's guess
        indistinguishable from something the user typed, and there is then no
        way to recover which entries they actually chose.
        """
        model = self.dir / "n.safetensors"
        model.write_bytes(b"x")
        sidecar.write_metadata_sidecar(
            model, {"name": "N", "base_model": "SDXL", "trained_words": ["a, b", "c"]}, raw={})
        data = json.loads((self.dir / "n.json").read_text(encoding="utf-8"))
        self.assertNotIn("activation text", data)
        # The official list is still recorded, just under its own key.
        self.assertEqual(data["trained_words"], ["a, b", "c"])

    def test_metadata_sidecar_keeps_activation_text_the_user_wrote(self):
        model = self.dir / "n.safetensors"
        model.write_bytes(b"x")
        (self.dir / "n.json").write_text(
            json.dumps({"activation text": "my own words", "preferred weight": 0.8}),
            encoding="utf-8")
        sidecar.write_metadata_sidecar(
            model, {"name": "N", "base_model": "SDXL", "trained_words": ["official"]}, raw={})
        data = json.loads((self.dir / "n.json").read_text(encoding="utf-8"))
        self.assertEqual(data["activation text"], "my own words")
        self.assertEqual(data["preferred weight"], 0.8)

    def test_metadata_sidecar_creates_no_empty_user_fields(self):
        """None of the WebUI-editable keys should appear until a person fills one."""
        model = self.dir / "n.safetensors"
        model.write_bytes(b"x")
        sidecar.write_metadata_sidecar(
            model, {"name": "N", "base_model": "SDXL", "trained_words": ["a"]}, raw={})
        data = json.loads((self.dir / "n.json").read_text(encoding="utf-8"))
        for key in sidecar.USER_OWNED_KEYS:
            self.assertNotIn(key, data, f"{key} must not be created on a fresh download")


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

class TestEnrich(LibraryFixture):

    def test_resolve_without_a_key_does_not_crash(self):
        """Regression: resolve/full-sync raised KeyError:'configured' when no
        provider key was set, breaking every metadata-fetching action.

        The faulty line ran before the resolve loop, so an already-complete
        library (nothing to resolve, hence no network) still exercises it.
        """
        from atlas import providers

        self.make_model("a.safetensors")
        self.rescan()
        file_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]
        with db.transaction(self.conn):
            enrich.upsert_model(self.conn, file_id, {
                "name": "Done", "base_model": "SDXL", "sha256": "aa",
                "provider": "civitai", "provider_version_id": "1",
                "trained_words": ["x"]}, "provider")

        # No key, and no rows need resolving, so no request is ever made.
        self.assertFalse(any(p.presentation()["configured"] for p in providers.all_providers()))
        stats = enrich.resolve_online(self.conn, self.root, None, force=False)
        self.assertEqual(stats["attempted"], 0)

    def test_ingest_populates_model_rows(self):
        self.make_model("a.safetensors", meta={
            "activation text": "hello", "sd version": "SD 1.5"})
        self.rescan()
        stats = enrich.ingest_sidecars(self.conn, self.root)
        self.assertEqual(stats["described"], 1)
        row = self.conn.execute("SELECT trained_words FROM model").fetchone()
        self.assertEqual(db.jload(row["trained_words"]), ["hello"])

    def test_upsert_does_not_blank_existing_fields(self):
        self.make_model("a.safetensors")
        self.rescan()
        file_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]

        with db.transaction(self.conn):
            enrich.upsert_model(self.conn, file_id, {
                "name": "Full Name", "base_model": "SDXL", "creator": "someone",
                "trained_words": ["a", "b"]}, "provider")
        with db.transaction(self.conn):
            enrich.upsert_model(self.conn, file_id, {"name": "Full Name"}, "sidecar")

        row = self.conn.execute("SELECT * FROM model").fetchone()
        self.assertEqual(row["base_model"], "SDXL")
        self.assertEqual(row["creator"], "someone")
        self.assertEqual(db.jload(row["trained_words"]), ["a", "b"])

    def test_curated_triggers_and_notes_survive_a_provider_refresh(self):
        """The regression that matters most: years of hand-curated metadata.

        A user's personal triggers and notes (from the previous manager's
        user_data.json) must survive a provider resolve that returns entirely
        different official trigger words.
        """
        self.make_model("fav.safetensors", meta={
            "activation text": "official one, official two",
            "my_own_field": "written by some other tool",
        })
        # The user's own curated data lives in user_data.json.
        (self.root / "user_data.json").write_text(json.dumps({
            "schema_version": 1,
            "personal_trigger_words": ["my custom trigger", "secret sauce"],
            "notes": "Best at 0.6 with DPM++",
            "preferred_weight": 0.6,
        }), encoding="utf-8")
        self.rescan()
        enrich.ingest_sidecars(self.conn, self.root)
        file_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]

        before = query.get_model(self.conn, file_id)
        self.assertEqual(before["personal_triggers"],
                         ["my custom trigger", "secret sauce"])
        self.assertEqual(before["notes"], "Best at 0.6 with DPM++")

        # A provider now returns completely different official triggers.
        official = {
            "provider": "civitai", "provider_model_id": "1", "provider_version_id": "2",
            "name": "Official Name", "base_model": "SDXL 1.0",
            "trained_words": ["official_a", "official_b"], "sha256": "ab",
        }
        with db.transaction(self.conn):
            enrich.upsert_model(self.conn, file_id, official, "provider")

        after = query.get_model(self.conn, file_id)
        self.assertEqual(after["personal_triggers"],
                         ["my custom trigger", "secret sauce"],
                         "curated personal triggers must survive in the database")
        self.assertEqual(after["notes"], "Best at 0.6 with DPM++",
                         "user notes must survive in the database")
        self.assertEqual(after["trained_words"], ["official_a", "official_b"],
                         "official triggers are tracked separately")

    def test_activation_text_is_not_adopted_as_personal_triggers(self):
        """Activation text is official trigger data, so it must not also land
        in personal triggers as a duplicate, split-up list."""
        self.make_model("a.safetensors", meta={
            "activation text": "MGN, long hair, black hair, braid"})
        self.rescan()
        enrich.ingest_sidecars(self.conn, self.root)
        file_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]
        model = query.get_model(self.conn, file_id)
        self.assertEqual(model["personal_triggers"], [])
        # It becomes the official list instead — as one group.
        self.assertEqual(model["trained_words"], ["MGN, long hair, black hair, braid"])

    def test_atlas_written_sidecars_are_not_treated_as_user_authored(self):
        model = self.root / "m.safetensors"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"x")
        self.assertFalse(sidecar.is_user_authored(
            self.root / "m.civitai.info", model))
        self.assertFalse(sidecar.is_user_authored(
            self.root / "m.atlas.json", model))
        self.assertTrue(sidecar.is_user_authored(self.root / "m.json", model))
        self.assertTrue(sidecar.is_user_authored(self.root / "metadata.json", model))

    def test_adoption_never_overwrites_existing_atlas_user_data(self):
        self.make_model("a.safetensors")
        (self.root / "user_data.json").write_text(json.dumps({
            "personal_trigger_words": ["from sidecar"], "notes": "sidecar note"}),
            encoding="utf-8")
        self.rescan()
        file_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]
        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT INTO user_data(file_id, notes, updated_at) VALUES (?,?,?)",
                (file_id, "notes I typed in Atlas", db.now()))

        enrich.ingest_sidecars(self.conn, self.root)

        row = self.conn.execute(
            "SELECT notes, personal_triggers FROM user_data WHERE file_id=?",
            (file_id,)).fetchone()
        self.assertEqual(row["notes"], "notes I typed in Atlas",
                         "Atlas-side notes win; adoption only fills gaps")
        self.assertIn("from sidecar", db.jload(row["personal_triggers"]))

    def test_completeness_grading(self):
        self.assertEqual(sidecar.completeness({}), "missing")
        self.assertEqual(sidecar.completeness(
            {"name": "x", "base_model": "SDXL"}), "partial")
        self.assertEqual(sidecar.completeness(
            {"name": "x", "base_model": "SDXL", "sha256": "aa", "trained_words": ["t"]}),
            "complete")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class TestQuery(LibraryFixture):

    def seed(self):
        self.make_model("anime/rin.safetensors", b"a", preview=True, meta={
            "model": {"name": "Rin Tohsaka", "creator": {"username": "alice"}},
            "baseModel": "Illustrious", "trainedWords": ["rin tohsaka", "twintails"],
            "id": 1, "modelId": 10,
        })
        self.make_model("style/watercolor.safetensors", b"bb", meta={
            "model": {"name": "Watercolour Style", "creator": {"username": "bob"}},
            "baseModel": "SDXL 1.0", "trainedWords": ["watercolour"],
            "id": 2, "modelId": 20,
        })
        self.make_model("misc/unknown.safetensors", b"ccc")
        self.rescan()
        enrich.ingest_sidecars(self.conn, self.root)

    def test_lists_everything_by_default(self):
        self.seed()
        result = query.list_models(self.conn, {})
        self.assertEqual(result["total"], 3)

    def test_pagination_is_stable_and_disjoint(self):
        self.seed()
        first = query.list_models(self.conn, {"page": 1, "page_size": 2})
        second = query.list_models(self.conn, {"page": 2, "page_size": 2})
        self.assertEqual(first["total"], 3)
        self.assertEqual(len(first["items"]), 2)
        self.assertEqual(len(second["items"]), 1)
        ids = {i["id"] for i in first["items"]} & {i["id"] for i in second["items"]}
        self.assertEqual(ids, set(), "pages must not overlap")

    def test_full_text_search(self):
        self.seed()
        result = query.list_models(self.conn, {"q": "tohsaka"})
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["name"], "Rin Tohsaka")

    def test_search_matches_trigger_words(self):
        self.seed()
        result = query.list_models(self.conn, {"q": "twintails"})
        self.assertEqual(result["total"], 1)

    def test_short_search_falls_back_to_like(self):
        self.seed()
        result = query.list_models(self.conn, {"q": "un"})
        self.assertGreaterEqual(result["total"], 1)

    def test_search_with_sql_metacharacters_is_safe(self):
        self.seed()
        for term in ["100%", "_test", "a' OR 1=1 --", '"quoted"', "a*b"]:
            with self.subTest(term=term):
                result = query.list_models(self.conn, {"q": term})
                self.assertLessEqual(result["total"], 3)

    def test_filter_by_base_model(self):
        self.seed()
        result = query.list_models(self.conn, {"base_model": "SDXL 1.0"})
        self.assertEqual(result["total"], 1)

    def test_uncategorised_filter_and_count(self):
        """The pile nothing has claimed is directly askable for."""
        self.seed()
        ids = [r["id"] for r in self.conn.execute("SELECT id FROM model_file ORDER BY path")]
        with db.transaction(self.conn):
            cid = self.conn.execute(
                "INSERT INTO collection(name, created_at) VALUES ('Anime', ?)",
                (db.now(),)).lastrowid
            self.conn.execute(
                "INSERT INTO collection_member(collection_id, file_id) VALUES (?,?)",
                (cid, ids[0]))

        self.assertEqual(query.facets(self.conn)["uncategorised"], 2)
        result = query.list_models(self.conn, {"uncategorised": "1"})
        self.assertEqual(result["total"], 2)
        # And the filed one is precisely what it excludes.
        filed = self.conn.execute(
            "SELECT path FROM model_file WHERE id=?", (ids[0],)).fetchone()["path"]
        self.assertNotIn(filed, [i["path"] for i in result["items"]])

    def test_uncategorised_is_its_own_flag_not_a_reserved_name(self):
        """A collection really can be called anything, so the filter must not
        depend on a category name being unreachable."""
        self.seed()
        # Even a collection named like a sentinel is treated as an ordinary one.
        with db.transaction(self.conn):
            cid = self.conn.execute(
                "INSERT INTO collection(name, created_at) VALUES ('..uncategorised', ?)",
                (db.now(),)).lastrowid
            fid = self.conn.execute("SELECT id FROM model_file LIMIT 1").fetchone()["id"]
            self.conn.execute(
                "INSERT INTO collection_member(collection_id, file_id) VALUES (?,?)",
                (cid, fid))
        # Filtering by that literal name finds the member, not the unfiled pile.
        named = query.list_models(self.conn, {"category": "..uncategorised"})
        self.assertEqual(named["total"], 1)
        # The flag still reports only genuinely unfiled models.
        unfiled = query.list_models(self.conn, {"uncategorised": "1"})
        self.assertEqual(unfiled["total"], 2)

    def test_filter_by_creator(self):
        self.seed()
        result = query.list_models(self.conn, {"creator": "alice"})
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["name"], "Rin Tohsaka")

    def test_filter_by_creator_excludes_others(self):
        self.seed()
        result = query.list_models(self.conn, {"creator": "bob"})
        self.assertEqual({i["name"] for i in result["items"]}, {"Watercolour Style"})

    def test_filter_missing_preview(self):
        self.seed()
        result = query.list_models(self.conn, {"missing_preview": "1"})
        self.assertEqual(result["total"], 2)

    def _make_collections(self):
        """Two nested collections across two categories."""
        with db.transaction(self.conn):
            for name in ("Anime / Fate", "Game / Genshin", "Anime / Type-Moon"):
                self.conn.execute(
                    "INSERT INTO collection(name, created_at) VALUES (?,?)", (name, db.now()))
            ids = {r["name"]: r["id"] for r in self.conn.execute("SELECT id, name FROM collection")}
            files = [r["id"] for r in self.conn.execute("SELECT id FROM model_file ORDER BY path")]
            # file0 -> Anime/Fate + Anime/Type-Moon, file1 -> Game/Genshin
            self.conn.execute("INSERT INTO collection_member VALUES (?,?)", (ids["Anime / Fate"], files[0]))
            self.conn.execute("INSERT INTO collection_member VALUES (?,?)", (ids["Anime / Type-Moon"], files[0]))
            self.conn.execute("INSERT INTO collection_member VALUES (?,?)", (ids["Game / Genshin"], files[1]))
        return files

    def test_category_filter_matches_all_nested(self):
        self.seed()
        self._make_collections()
        self.assertEqual(query.list_models(self.conn, {"category": "Anime"})["total"], 1)
        self.assertEqual(query.list_models(self.conn, {"category": "Game"})["total"], 1)

    def test_category_filter_counts_a_model_once(self):
        """A model in two sub-categories of one category must not double-count."""
        self.seed()
        self._make_collections()
        result = query.list_models(self.conn, {"category": "Anime"})
        self.assertEqual(result["total"], 1)
        self.assertEqual(len(result["items"]), 1)

    def test_subcategory_narrows(self):
        self.seed()
        self._make_collections()
        self.assertEqual(
            query.list_models(self.conn, {"category": "Anime", "subcategory": "Fate"})["total"], 1)
        self.assertEqual(
            query.list_models(self.conn, {"category": "Anime", "subcategory": "Nonexistent"})["total"], 0)

    def test_category_tree_shape(self):
        self.seed()
        self._make_collections()
        tree = query.category_tree(self.conn)
        names = {c["name"]: c for c in tree}
        self.assertIn("Anime", names)
        self.assertIn("Game", names)
        self.assertEqual(names["Anime"]["count"], 1)
        subs = {s["name"] for s in names["Anime"]["subcategories"]}
        self.assertEqual(subs, {"Fate", "Type-Moon"})

    def test_has_notes_rated_and_triggers_filters(self):
        self.seed()
        fid = self.conn.execute("SELECT id FROM model_file ORDER BY path").fetchone()["id"]
        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT INTO user_data(file_id, notes, rating, personal_triggers, updated_at) "
                "VALUES (?,?,?,?,?)",
                (fid, "a note", 5, db.jdump(["my trigger"]), db.now()))
        self.assertEqual(query.list_models(self.conn, {"has_notes": "1"})["total"], 1)
        self.assertEqual(query.list_models(self.conn, {"rated": "1"})["total"], 1)
        self.assertEqual(query.list_models(self.conn, {"has_triggers": "1"})["total"], 1)

    def test_sorts_are_all_valid_sql(self):
        self.seed()
        for sort in query.SORTS:
            with self.subTest(sort=sort):
                query.list_models(self.conn, {"sort": sort})

    def test_unknown_sort_falls_back_to_name(self):
        self.seed()
        result = query.list_models(self.conn, {"sort": "; DROP TABLE model_file"})
        self.assertEqual(result["total"], 3)
        self.assertTrue(self.conn.execute("SELECT 1 FROM model_file LIMIT 1").fetchone())

    def test_page_size_is_clamped(self):
        self.seed()
        result = query.list_models(self.conn, {"page_size": 99999})
        self.assertLessEqual(result["page_size"], 500)

    def test_stats_reflect_library(self):
        self.seed()
        stats = query.stats(self.conn)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["missing_preview"], 2)
        self.assertEqual(stats["missing_metadata"], 1)

    def test_missing_files_excluded_from_listing(self):
        self.seed()
        (self.root / "misc/unknown.safetensors").unlink()
        self.rescan()
        self.assertEqual(query.list_models(self.conn, {})["total"], 2)
        self.assertEqual(query.stats(self.conn)["missing_files"], 1)


# ---------------------------------------------------------------------------
# User data
# ---------------------------------------------------------------------------

class TestUserData(LibraryFixture):

    def test_user_data_survives_rename(self):
        """The point of hash-keyed orphan storage: renaming must not lose ratings."""
        self.make_model("old_name.safetensors", b"stable content")
        self.rescan()
        hashing.hash_pending(self.conn, self.root)
        file_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]
        sha = self.conn.execute("SELECT sha256 FROM model_file").fetchone()["sha256"]

        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT INTO user_data(file_id, rating, favorite, notes, updated_at) "
                "VALUES (?,5,1,'my notes',?)", (file_id, db.now()))
            # Simulate what a delete-detection sweep would archive.
            self.conn.execute(
                "INSERT INTO user_data_orphan(sha256, payload, orphaned_at) VALUES (?,?,?)",
                (sha, db.jdump({"rating": 5, "favorite": 1, "notes": "my notes"}), db.now()))

        (self.root / "old_name.safetensors").rename(self.root / "new_name.safetensors")
        self.rescan()
        hashing.hash_pending(self.conn, self.root)

        new_id = self.conn.execute(
            "SELECT id FROM model_file WHERE path='new_name.safetensors'").fetchone()["id"]
        scanner._reattach_user_data(self.conn, new_id)
        restored = self.conn.execute(
            "SELECT rating, notes FROM user_data WHERE file_id=?", (new_id,)).fetchone()
        self.assertIsNotNone(restored)
        self.assertEqual(restored["rating"], 5)
        self.assertEqual(restored["notes"], "my notes")


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

class TestBackup(LibraryFixture):

    def test_roundtrip_restores_user_data_after_reorganisation(self):
        self.make_model("a/model.safetensors", b"content-a")
        self.rescan()
        hashing.hash_pending(self.conn, self.root)
        file_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]
        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT INTO user_data(file_id, rating, favorite, notes, updated_at) "
                "VALUES (?,4,1,'keep this',?)", (file_id, db.now()))
            self.conn.execute(
                "INSERT INTO collection(name, created_at) VALUES ('Favourites', ?)", (db.now(),))
            cid = self.conn.execute("SELECT id FROM collection").fetchone()["id"]
            self.conn.execute(
                "INSERT INTO collection_member(collection_id, file_id) VALUES (?,?)",
                (cid, file_id))

        archive = Path(self._tmp.name) / "test.atlasbak"
        result = backup.export(self.conn, self.root, archive)
        self.assertEqual(result["models"], 1)
        self.assertTrue(archive.is_file())

        info = backup.inspect(archive)
        self.assertEqual(info["model_count"], 1)
        self.assertEqual(info["format"], backup.FORMAT)

        # Move the file and wipe local state — the restore must still match.
        (self.root / "a" / "model.safetensors").parent.mkdir(exist_ok=True)
        (self.root / "a" / "model.safetensors").rename(self.root / "relocated.safetensors")
        with db.transaction(self.conn):
            self.conn.execute("DELETE FROM user_data")
            self.conn.execute("DELETE FROM collection_member")
        self.rescan()
        hashing.hash_pending(self.conn, self.root)

        stats = backup.restore(self.conn, self.root, archive)
        self.assertEqual(stats["matched"], 1)

        new_id = self.conn.execute(
            "SELECT id FROM model_file WHERE path='relocated.safetensors'").fetchone()["id"]
        row = self.conn.execute(
            "SELECT rating, notes FROM user_data WHERE file_id=?", (new_id,)).fetchone()
        self.assertEqual(row["rating"], 4)
        self.assertEqual(row["notes"], "keep this")
        members = self.conn.execute(
            "SELECT COUNT(*) c FROM collection_member WHERE file_id=?", (new_id,)).fetchone()["c"]
        self.assertEqual(members, 1)

    def test_unmatched_entries_are_parked_for_later(self):
        self.make_model("present.safetensors", b"here")
        self.rescan()
        hashing.hash_pending(self.conn, self.root)
        archive = Path(self._tmp.name) / "b.atlasbak"

        file_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]
        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT INTO user_data(file_id, rating, updated_at) VALUES (?,3,?)",
                (file_id, db.now()))
        backup.export(self.conn, self.root, archive)

        # Remove the file entirely, then restore into the now-empty library.
        (self.root / "present.safetensors").unlink()
        self.rescan()
        with db.transaction(self.conn):
            self.conn.execute("DELETE FROM model_file")

        stats = backup.restore(self.conn, self.root, archive)
        self.assertEqual(stats["unmatched"], 1)
        parked = self.conn.execute("SELECT COUNT(*) c FROM user_data_orphan").fetchone()["c"]
        self.assertEqual(parked, 1)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class TestFolderAdoption(LibraryFixture):

    TREE = [
        "Anime/Fate/Rin/rin.safetensors",
        "Anime/Fate/Rin/rin_v2.safetensors",
        "Anime/Fate/Saber/saber.safetensors",
        "Anime/Genshin/Raiden/raiden.safetensors",
        "Style/Watercolour/wc.safetensors",
        "Style/Retro/80s/retro.safetensors",
        "unsorted/random.safetensors",
        "loose.safetensors",
    ]

    def seed_tree(self):
        for rel in self.TREE:
            self.make_model(rel)
        self.rescan()

    def test_analysis_is_read_only(self):
        from atlas import folders
        self.seed_tree()
        before = self.conn.execute("SELECT COUNT(*) c FROM collection").fetchone()["c"]
        report = folders.analyse(self.conn)
        after = self.conn.execute("SELECT COUNT(*) c FROM collection").fetchone()["c"]
        self.assertEqual(before, after, "analysis must not create anything")
        self.assertEqual(report["total_models"], len(self.TREE))
        self.assertEqual(report["models_at_root"], 1)
        self.assertEqual(report["max_depth"], 3)

    def test_adopt_creates_nested_collections_without_moving_files(self):
        from atlas import folders
        self.seed_tree()
        folders.import_as_collections(self.conn, None, max_depth=2)

        for rel in self.TREE:
            self.assertTrue((self.root / rel).is_file(),
                            f"{rel} must not be moved or renamed")

        names = {r["name"] for r in self.conn.execute("SELECT name FROM collection")}
        self.assertIn("Anime", names)
        self.assertIn("Anime / Fate", names)
        self.assertIn("Style", names)

    def test_parent_collection_includes_all_descendants(self):
        from atlas import folders
        self.seed_tree()
        folders.import_as_collections(self.conn, None, max_depth=2)
        anime = self.conn.execute(
            "SELECT id FROM collection WHERE name='Anime'").fetchone()["id"]
        result = query.list_models(self.conn, {"collection": anime})
        self.assertEqual(result["total"], 4,
                         "filtering by a parent must find everything beneath it")

    def test_noise_folders_are_skipped(self):
        from atlas import folders
        self.seed_tree()
        folders.import_as_collections(self.conn, None, max_depth=2)
        names = {r["name"] for r in self.conn.execute("SELECT name FROM collection")}
        self.assertNotIn("unsorted", names)

    def test_adopt_is_idempotent(self):
        from atlas import folders
        self.seed_tree()
        folders.import_as_collections(self.conn, None, max_depth=2)
        second = folders.import_as_collections(self.conn, None, max_depth=2)
        self.assertEqual(second["collections_created"], 0)
        self.assertEqual(second["assignments_added"], 0)

    def test_adopt_does_not_drop_hand_added_members_by_default(self):
        """A user's manual collection assignments must survive a re-adopt."""
        from atlas import folders
        self.seed_tree()
        folders.import_as_collections(self.conn, None, max_depth=2)
        anime = self.conn.execute(
            "SELECT id FROM collection WHERE name='Anime'").fetchone()["id"]
        loose = self.conn.execute(
            "SELECT id FROM model_file WHERE path='loose.safetensors'").fetchone()["id"]
        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT INTO collection_member(collection_id, file_id) VALUES (?,?)",
                (anime, loose))

        folders.import_as_collections(self.conn, None, max_depth=2)
        still = self.conn.execute(
            "SELECT 1 FROM collection_member WHERE collection_id=? AND file_id=?",
            (anime, loose)).fetchone()
        self.assertIsNotNone(still, "a hand-added member must not be pruned silently")

        folders.import_as_collections(self.conn, None, max_depth=2, prune=True)
        gone = self.conn.execute(
            "SELECT 1 FROM collection_member WHERE collection_id=? AND file_id=?",
            (anime, loose)).fetchone()
        self.assertIsNone(gone, "prune=True is the opt-in that removes it")

    def test_preview_matches_what_adopt_creates(self):
        from atlas import folders
        self.seed_tree()
        preview = folders.preview_import(self.conn, max_depth=2)
        stats = folders.import_as_collections(self.conn, None, max_depth=2)
        self.assertEqual(preview["total_collections"], stats["collections_created"])
        self.assertEqual(preview["total_assignments"], stats["assignments_added"])


class TestLegacyUserData(LibraryFixture):
    """Importing the previous manager's user_data.json."""

    LEGACY = {
        "schema_version": 1,
        "last_modified": "2026-06-29T05:31:22.340068Z",
        "personal_trigger_words": ["my trigger", "another"],
        "notes": "Years of accumulated notes",
        "rating": 4,
        "collections": ["Anime / Kiss X Sis"],
        "preferred_weight": 0.75,
        "negative_prompt": "blurry",
    }

    def seed(self, folder="Anime/Kiss X Sis/Some Model"):
        self.make_model(f"{folder}/m.safetensors", preview=True)
        (self.root / folder / "user_data.json").write_text(
            json.dumps(self.LEGACY), encoding="utf-8")
        self.rescan()
        enrich.ingest_sidecars(self.conn, self.root)
        return self.conn.execute("SELECT id FROM model_file").fetchone()["id"]

    def test_legacy_user_data_is_imported(self):
        file_id = self.seed()
        model = query.get_model(self.conn, file_id)
        self.assertEqual(model["rating"], 4)
        self.assertEqual(model["notes"], "Years of accumulated notes")
        self.assertEqual(model["personal_triggers"], ["my trigger", "another"])
        self.assertEqual(model["weight"], 0.75)

    def test_legacy_collections_are_recreated(self):
        self.seed()
        row = self.conn.execute(
            "SELECT id FROM collection WHERE name='Anime / Kiss X Sis'").fetchone()
        self.assertIsNotNone(row, "collections in user_data.json must be restored")
        members = self.conn.execute(
            "SELECT COUNT(*) c FROM collection_member WHERE collection_id=?",
            (row["id"],)).fetchone()["c"]
        self.assertEqual(members, 1)

    def test_legacy_zero_rating_is_treated_as_unset(self):
        folder = "X/Y"
        self.make_model(f"{folder}/m.safetensors")
        (self.root / folder / "user_data.json").write_text(
            json.dumps({**self.LEGACY, "rating": 0, "preferred_weight": 0.0}),
            encoding="utf-8")
        self.rescan()
        enrich.ingest_sidecars(self.conn, self.root)
        file_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]
        model = query.get_model(self.conn, file_id)
        self.assertIsNone(model["rating"])
        self.assertIsNone(model["weight"])

    def _write_user(self, rel, **payload):
        base = {"rating": 0, "notes": "", "personal_triggers": [], "collections": [],
                "weight": None, "negative_prompt": "", "favorite": False}
        sidecar.write_legacy_user_data(self.root / rel, {**base, **payload})

    def test_versions_in_one_folder_keep_separate_user_data(self):
        """Two versions sharing a folder must not overwrite each other.

        Versions of one model deliberately live together, so the per-folder
        user_data.json is ambiguous — each model gets its own file instead.
        """
        folder = "Anime/Angelina"
        self.make_model(f"{folder}/angelina_v1.safetensors")
        self.make_model(f"{folder}/angelina_anima.safetensors")
        self._write_user(f"{folder}/angelina_v1.safetensors",
                         rating=2, notes="v1 is weaker", personal_triggers=["v1 trig"])
        self._write_user(f"{folder}/angelina_anima.safetensors",
                         rating=5, notes="anima is my favourite",
                         personal_triggers=["anima trig"])

        on_disk = sorted(p.name for p in (self.root / folder).glob("*.json"))
        self.assertEqual(on_disk, ["angelina_anima.user.json", "angelina_v1.user.json"],
                         "each version needs its own file, and no ambiguous shared one")

        # Rebuilding purely from disk must give each version its own data back.
        self.rescan()
        enrich.ingest_sidecars(self.conn, self.root, only_unknown=False)
        got = {r["filename"]: (r["rating"], r["notes"])
               for r in self.conn.execute(
                   "SELECT f.filename, u.rating, u.notes FROM model_file f "
                   "JOIN user_data u ON u.file_id=f.id")}
        self.assertEqual(got["angelina_v1.safetensors"], (2, "v1 is weaker"))
        self.assertEqual(got["angelina_anima.safetensors"], (5, "anima is my favourite"))

    def test_single_model_folder_still_writes_the_shared_file(self):
        """A lone model keeps user_data.json current, so the old app still works."""
        self.make_model("Anime/Solo/m.safetensors")
        self._write_user("Anime/Solo/m.safetensors", rating=3, notes="hello")
        shared = self.root / "Anime/Solo/user_data.json"
        self.assertTrue(shared.exists())
        self.assertEqual(json.loads(shared.read_text(encoding="utf-8"))["rating"], 3)

    def test_per_model_file_wins_over_the_shared_one(self):
        folder = "Anime/Both"
        self.make_model(f"{folder}/m.safetensors")
        (self.root / folder / "user_data.json").write_text(
            json.dumps({**self.LEGACY, "rating": 1, "notes": "stale shared"}),
            encoding="utf-8")
        (self.root / folder / "m.user.json").write_text(
            json.dumps({**self.LEGACY, "rating": 5, "notes": "mine"}), encoding="utf-8")
        self.rescan()
        enrich.ingest_sidecars(self.conn, self.root, only_unknown=False)
        file_id = self.conn.execute(
            "SELECT id FROM model_file WHERE filename='m.safetensors'").fetchone()["id"]
        model = query.get_model(self.conn, file_id)
        self.assertEqual((model["rating"], model["notes"]), (5, "mine"))

    def test_user_data_notes_never_leak_into_provider_description(self):
        """user_data.json shares key names with metadata; it must not be merged."""
        folder = "Anime/Leak"
        self.make_model(f"{folder}/m.safetensors")
        (self.root / folder / "user_data.json").write_text(
            json.dumps({"notes": "private", "tags": ["mine"]}), encoding="utf-8")
        # The scanner hands user_data.json to read_all as a known sidecar, so
        # that is the path that has to keep it out of the metadata blob.
        merged, _, parsed = sidecar.read_all(
            self.root / folder / "m.safetensors",
            known=[self.root / folder / "user_data.json"])
        self.assertNotIn("notes", merged)
        self.assertNotIn("tags", merged)
        self.assertTrue(any(p.name == "user_data.json" for p, _ in parsed),
                        "the file must still be offered to the adoption step")

    def test_provider_refresh_cannot_touch_legacy_user_data(self):
        file_id = self.seed()
        with db.transaction(self.conn):
            enrich.upsert_model(self.conn, file_id, {
                "name": "Official", "base_model": "SDXL", "tags": ["official"],
                "trained_words": ["official_only"], "sha256": "aa"}, "provider")
        model = query.get_model(self.conn, file_id)
        self.assertEqual(model["personal_triggers"], ["my trigger", "another"])
        self.assertEqual(model["notes"], "Years of accumulated notes")
        self.assertEqual(model["rating"], 4)


class TestUserTags(LibraryFixture):

    def test_user_tags_are_separate_from_provider_tags(self):
        self.make_model("a.safetensors")
        self.rescan()
        file_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]
        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT INTO user_data(file_id, tags, updated_at) VALUES (?,?,?)",
                (file_id, db.jdump(["mine", "waifu"]), db.now()))
            enrich.upsert_model(self.conn, file_id, {
                "name": "N", "base_model": "SDXL", "tags": ["provider-tag"],
                "sha256": "aa"}, "provider")
        model = query.get_model(self.conn, file_id)
        self.assertEqual(model["user_tags"], ["mine", "waifu"])
        self.assertEqual(model["tags"], ["provider-tag"])

    def test_user_tags_are_searchable(self):
        self.make_model("a.safetensors")
        self.rescan()
        file_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]
        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT INTO user_data(file_id, tags, updated_at) VALUES (?,?,?)",
                (file_id, db.jdump(["zephyrtag"]), db.now()))
            db.fts_replace(self.conn, file_id)
        self.assertEqual(query.list_models(self.conn, {"q": "zephyrtag"})["total"], 1)


class TestOrganizer(LibraryFixture):

    def seed(self):
        for rel in ["Anime/Fate/Rin/rin.safetensors",
                    "Anime/Fate/Saber/saber.safetensors"]:
            self.make_model(rel, preview=True, meta={
                "model": {"name": Path(rel).stem.title(),
                          "creator": {"username": "someone"}},
                "baseModel": "Illustrious", "trainedWords": ["t"],
                "id": 1, "modelId": 10 if "rin" in rel else 11})
        self.rescan()
        enrich.ingest_sidecars(self.conn, self.root)
        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT INTO collection(name, created_at) VALUES ('Anime / Fate', ?)",
                (db.now(),))
            cid = self.conn.execute("SELECT id FROM collection").fetchone()["id"]
            for r in self.conn.execute("SELECT id FROM model_file"):
                self.conn.execute(
                    "INSERT INTO collection_member(collection_id, file_id) VALUES (?,?)",
                    (cid, r["id"]))

    def test_safe_component_handles_illegal_and_unicode(self):
        from atlas import organizer
        self.assertEqual(organizer.safe_component('a<b>c:d/e'), "a b c d e")
        self.assertEqual(organizer.safe_component("trailing. "), "trailing")
        self.assertEqual(organizer.safe_component("Maō Gakuin"), "Maō Gakuin")
        self.assertTrue(organizer.safe_component("CON").startswith("_"))
        self.assertEqual(organizer.safe_component(""), "Unnamed")
        self.assertEqual(organizer.safe_component("dash - "), "dash")

    def test_long_folder_name_does_not_end_in_a_separator(self):
        """Truncating a long "Name - Base - Creator" must not leave a dangling
        " -" on the folder written to disk."""
        from atlas import organizer

        class Row(dict):
            def __getitem__(self, k):
                return dict.get(self, k)

        name = organizer.folder_name_for(Row(
            name="Eleanor Bianca - Maou Gakuin no Futekigousha "
                 "(The Misfit of Demon King Academy) - 2 Outfits - Illustrious/SDXL",
            base_model="Illustrious", creator="KurutAI",
            filename="MGNEleanor_ILXL.safetensors"))
        self.assertLessEqual(len(name), organizer.MAX_COMPONENT)
        self.assertNotIn(name.rstrip()[-1], "-–—_|.",
                         f"folder name ends with a separator: {name!r}")

    def test_folder_name_does_not_repeat_information(self):
        from atlas import organizer

        class Row(dict):
            def __getitem__(self, k):
                return dict.get(self, k)

        name = organizer.folder_name_for(Row(
            name="Rin | Illustrious | Fate", base_model="Illustrious",
            creator="alice", filename="rin.safetensors"))
        self.assertEqual(name.lower().count("illustrious"), 1,
                         "the base model must not be appended twice")
        self.assertIn("alice", name)

    def test_move_carries_every_asset_and_loses_nothing(self):
        from atlas import organizer
        self.seed()
        before = {p.read_bytes() for p in self.root.rglob("*") if p.is_file()
                  and ".atlas" not in p.parts}
        plan = organizer.plan_collection_layout(self.conn, self.root)
        organizer.apply_plan(self.conn, self.root, plan)
        after = {p.read_bytes() for p in self.root.rglob("*") if p.is_file()
                 and ".atlas" not in p.parts}
        self.assertEqual(before, after, "no file content may be lost in a move")

    def test_move_leaves_no_empty_directories(self):
        from atlas import organizer
        self.seed()
        plan = organizer.plan_collection_layout(self.conn, self.root)
        organizer.apply_plan(self.conn, self.root, plan)
        empty = [d for d in self.root.rglob("*")
                 if d.is_dir() and ".atlas" not in d.parts and not any(d.iterdir())]
        self.assertEqual(empty, [], "an organise pass must not strand empty folders")

    def test_move_back_into_a_vacated_path_does_not_collide(self):
        """Regression: UNIQUE constraint failed: model_file.path.

        Moving a file out of a folder leaves its old row as present=0. Moving
        the file back to that exact path used to collide with the stale row and
        abort the whole organise/onboard run with an IntegrityError.
        """
        import shutil as _shutil
        from atlas import organizer, scanner

        self.make_model("Foo/m.safetensors", b"stable bytes")
        self.rescan()
        _shutil.move(str(self.root / "Foo" / "m.safetensors"),
                     str(self.root / "m.safetensors"))
        scanner.sync(self.conn, self.root)

        rows = {r["path"]: r["present"] for r in self.conn.execute(
            "SELECT path, present FROM model_file")}
        self.assertEqual(rows.get("Foo/m.safetensors"), 0, "stale row should remain")

        live = self.conn.execute(
            "SELECT id, path, filename FROM model_file WHERE present=1").fetchone()
        plan = organizer.Plan(moves=[organizer.Move(
            file_id=live["id"], from_dir="", to_dir="Foo", assets=[], reason="test")])
        stats = organizer.apply_plan(self.conn, self.root, plan, remove_empty=False)

        self.assertEqual(stats["errors"], 0)
        self.assertEqual(stats["moved"], 1)
        self.assertTrue((self.root / "Foo" / "m.safetensors").is_file())
        # Exactly one row now owns that path.
        count = self.conn.execute(
            "SELECT COUNT(*) c FROM model_file WHERE path='Foo/m.safetensors'"
        ).fetchone()["c"]
        self.assertEqual(count, 1)

    def test_stale_row_user_data_is_inherited_on_move_back(self):
        """Ratings attached to the vacated path must not be lost."""
        import shutil as _shutil
        from atlas import organizer, scanner

        self.make_model("Foo/m.safetensors", b"stable bytes")
        self.rescan()
        old_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]
        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT INTO user_data(file_id, rating, notes, updated_at) "
                "VALUES (?,?,?,?)", (old_id, 5, "keep me", db.now()))

        _shutil.move(str(self.root / "Foo" / "m.safetensors"),
                     str(self.root / "m.safetensors"))
        scanner.sync(self.conn, self.root)
        live = self.conn.execute(
            "SELECT id FROM model_file WHERE present=1").fetchone()["id"]
        plan = organizer.Plan(moves=[organizer.Move(
            file_id=live, from_dir="", to_dir="Foo", assets=[], reason="test")])
        organizer.apply_plan(self.conn, self.root, plan, remove_empty=False)

        row = self.conn.execute(
            "SELECT rating, notes FROM user_data WHERE file_id=?", (live,)).fetchone()
        self.assertIsNotNone(row, "user data from the stale row should carry over")
        self.assertEqual(row["rating"], 5)
        self.assertEqual(row["notes"], "keep me")

    def test_orphans_detected_in_asset_only_folders(self):
        """The husk a move leaves behind: metadata with no model beside it."""
        from atlas import organizer
        self.seed()
        husk = self.root / "Anime/Fate/Ghost"
        husk.mkdir(parents=True, exist_ok=True)
        (husk / "leftover.json").write_text("{}", encoding="utf-8")
        orphans = organizer.find_orphans(self.conn, self.root)
        self.assertIn("Anime/Fate/Ghost/leftover.json", orphans)

    def test_orphan_detection_ignores_folders_with_other_content(self):
        from atlas import organizer
        self.seed()
        keep = self.root / "MyReferences"
        keep.mkdir(parents=True, exist_ok=True)
        (keep / "notes.md").write_text("mine", encoding="utf-8")
        (keep / "pic.png").write_bytes(b"img")
        orphans = organizer.find_orphans(self.conn, self.root)
        self.assertFalse([o for o in orphans if o.startswith("MyReferences")],
                         "a folder with unrecognised content must be left alone")

    def test_dry_run_moves_nothing(self):
        from atlas import organizer
        config.set_many({"dry_run": True})
        try:
            self.seed()
            paths_before = sorted(p.as_posix() for p in self.root.rglob("*.safetensors"))
            plan = organizer.plan_collection_layout(self.conn, self.root)
            stats = organizer.apply_plan(self.conn, self.root, plan)
            paths_after = sorted(p.as_posix() for p in self.root.rglob("*.safetensors"))
            self.assertTrue(stats["dry_run"])
            self.assertEqual(paths_before, paths_after)
        finally:
            config.set_many({"dry_run": False})

    def test_version_key_groups_same_model_different_versions(self):
        from atlas import organizer

        class Row(dict):
            def __getitem__(self, k):
                return dict.get(self, k)

        a = Row(provider="civitai", provider_model_id="55", creator="x", name="n")
        b = Row(provider="civitai", provider_model_id="55", creator="y", name="m")
        c = Row(provider="civitai", provider_model_id="99", creator="x", name="n")
        self.assertEqual(organizer.version_key(a), organizer.version_key(b))
        self.assertNotEqual(organizer.version_key(a), organizer.version_key(c))


class TestCollectionApi(LibraryFixture):
    """The create-collection-and-move flow the drawer UI drives."""

    def _request(self, method, segments, body=None, query_params=None):
        from atlas import api
        return api.Request(method, "/".join(segments), query_params or {}, body or {}, segments)

    def seed_two(self):
        for rel in ["A/one.safetensors", "B/two.safetensors"]:
            self.make_model(rel, preview=True, meta={
                "model": {"name": Path(rel).stem, "creator": {"username": "x"}},
                "baseModel": "SDXL", "id": 1})
        self.rescan()
        enrich.ingest_sidecars(self.conn, self.root)
        return [r["id"] for r in self.conn.execute("SELECT id FROM model_file ORDER BY path")]

    def test_create_collection_records_nesting_depth(self):
        from atlas import api
        req = self._request("POST", ["api", "collections"], {"name": "Anime / Fate / Rin"})
        result = api.create_collection(req)
        self.assertTrue(result["id"])
        row = self.conn.execute(
            "SELECT sort_order FROM collection WHERE id=?", (result["id"],)).fetchone()
        self.assertEqual(row["sort_order"], 2, "nesting depth should drive sort order")

    def test_add_member_without_move_returns_no_job(self):
        from atlas import api
        ids = self.seed_two()
        cid = api.create_collection(self._request(
            "POST", ["api", "collections"], {"name": "Faves"}))["id"]
        result = api.update_membership(self._request(
            "POST", ["api", "collections", str(cid), "members"],
            {"file_ids": [ids[0]], "add": True}))
        self.assertNotIn("job", result, "a virtual add must not move files")
        member = self.conn.execute(
            "SELECT 1 FROM collection_member WHERE collection_id=? AND file_id=?",
            (cid, ids[0])).fetchone()
        self.assertIsNotNone(member)

    def test_add_member_with_move_submits_a_job(self):
        from atlas import api
        from atlas.jobs import QUEUE
        ids = self.seed_two()
        cid = api.create_collection(self._request(
            "POST", ["api", "collections"], {"name": "Faves"}))["id"]
        result = api.update_membership(self._request(
            "POST", ["api", "collections", str(cid), "members"],
            {"file_ids": [ids[0]], "add": True, "move_files": True}))
        self.assertIn("job", result, "move_files must enqueue an organise job")
        QUEUE.cancel(result["job"]["id"])

    def test_remove_member(self):
        from atlas import api
        ids = self.seed_two()
        cid = api.create_collection(self._request(
            "POST", ["api", "collections"], {"name": "Faves"}))["id"]
        api.update_membership(self._request(
            "POST", ["api", "collections", str(cid), "members"],
            {"file_ids": ids, "add": True}))
        api.update_membership(self._request(
            "POST", ["api", "collections", str(cid), "members"],
            {"file_ids": [ids[0]], "add": False}))
        remaining = self.conn.execute(
            "SELECT COUNT(*) c FROM collection_member WHERE collection_id=?", (cid,)).fetchone()["c"]
        self.assertEqual(remaining, 1)

    # --- renaming a category -------------------------------------------------

    def _seed_nested(self):
        """One model under Anime / Kiss X Sis, with the folders to match."""
        from atlas import api
        self.make_model("Anime/Kiss X Sis/m/m.safetensors", preview=True)
        self.rescan()
        file_id = self.conn.execute("SELECT id FROM model_file").fetchone()["id"]
        for name in ("Anime", "Anime / Kiss X Sis"):
            cid = api.create_collection(self._request(
                "POST", ["api", "collections"], {"name": name}))["id"]
            api.update_membership(self._request(
                "POST", ["api", "collections", str(cid), "members"],
                {"file_ids": [file_id], "add": True}))
        return file_id

    def _rename(self, name, new_name):
        from atlas import api
        return api.rename_category(self._request(
            "POST", ["api", "collections", "rename-by-name"],
            {"name": name, "new_name": new_name}))

    def test_rename_subcategory_moves_the_folder_and_keeps_the_model(self):
        file_id = self._seed_nested()
        result = self._rename("Anime / Kiss X Sis", "Kissxsis")
        self.assertEqual(result["new_name"], "Anime / Kissxsis")
        self.assertTrue((self.root / "Anime/Kissxsis/m/m.safetensors").is_file())
        self.assertFalse((self.root / "Anime/Kiss X Sis").exists())
        row = self.conn.execute(
            "SELECT path, present FROM model_file WHERE id=?", (file_id,)).fetchone()
        self.assertEqual(row["path"], "Anime/Kissxsis/m/m.safetensors")
        self.assertEqual(row["present"], 1)

    def test_rename_parent_carries_its_children(self):
        self._seed_nested()
        self._rename("Anime", "Animation")
        names = {r["name"] for r in self.conn.execute("SELECT name FROM collection")}
        self.assertEqual(names, {"Animation", "Animation / Kiss X Sis"})
        self.assertTrue((self.root / "Animation/Kiss X Sis/m/m.safetensors").is_file())

    def test_rename_moves_non_members_sitting_in_the_folder(self):
        """A rename is a folder rename: nothing gets left under the old name.

        Only some models in a category folder are usually collection members,
        and moving just those would strand the rest in a folder the user
        believes they renamed.
        """
        self._seed_nested()
        self.make_model("Anime/Kiss X Sis/stray/stray.safetensors")
        self.rescan()
        result = self._rename("Anime / Kiss X Sis", "Kissxsis")
        self.assertEqual(result["models_moved"], 2)
        self.assertTrue((self.root / "Anime/Kissxsis/stray/stray.safetensors").is_file())
        self.assertFalse((self.root / "Anime/Kiss X Sis").exists())

    def test_rename_onto_an_existing_name_is_refused(self):
        from atlas import api
        self._seed_nested()
        api.create_collection(self._request(
            "POST", ["api", "collections"], {"name": "Manga"}))
        with self.assertRaises(api.ApiError) as caught:
            self._rename("Anime", "Manga")
        self.assertEqual(caught.exception.status, 409)

    def test_rename_rejects_a_path_separator(self):
        from atlas import api
        self._seed_nested()
        with self.assertRaises(api.ApiError) as caught:
            self._rename("Anime", "Anime/Sub")
        self.assertEqual(caught.exception.status, 400)

    def test_rename_preserves_user_data(self):
        file_id = self._seed_nested()
        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT INTO user_data(file_id, rating, notes, updated_at) "
                "VALUES (?,5,'keep me',?)", (file_id, db.now()))
        self._rename("Anime / Kiss X Sis", "Kissxsis")
        new_id = self.conn.execute(
            "SELECT id FROM model_file WHERE present=1").fetchone()["id"]
        row = self.conn.execute(
            "SELECT rating, notes FROM user_data WHERE file_id=?", (new_id,)).fetchone()
        self.assertEqual((row["rating"], row["notes"]), (5, "keep me"))


class TestVideoPreviews(unittest.TestCase):
    """A version's showcase media can lead with an animation.

    CivitAI image URLs carry no extension, so downloading entry zero blindly
    wrote an mp4 out as .png: a file on disk and a blank card in the gallery.
    """

    @staticmethod
    def _info(urls, meta=None):
        return ModelInfo(provider="civitai", preview_urls=urls,
                         preview_meta=meta or [])

    def test_skips_a_leading_video_and_takes_the_next(self):
        got = enrich._image_candidates(self._info(
            ["https://image.civitai.com/AAA/width=450",
             "https://image.civitai.com/BBB/width=450"],
            [{"url": "https://image.civitai.com/AAA/width=450", "type": "video"},
             {"url": "https://image.civitai.com/BBB/width=450", "type": "image"}]))
        self.assertEqual(got, ["https://image.civitai.com/BBB/width=450"])

    def test_falls_back_to_the_extension_when_untyped(self):
        # Arcenciel does not label entries, so the suffix is the only signal.
        got = enrich._image_candidates(self._info(
            ["https://ex.com/uploads/a.mp4", "https://ex.com/uploads/b.jpeg"]))
        self.assertEqual(got, ["https://ex.com/uploads/b.jpeg"])

    def test_all_video_yields_nothing_rather_than_a_broken_file(self):
        got = enrich._image_candidates(self._info(
            ["https://x/a.mp4"], [{"url": "https://x/a.mp4", "type": "video"}]))
        self.assertEqual(got, [])

    def test_ordinary_previews_keep_provider_order(self):
        urls = ["https://x/1.png", "https://x/2.png"]
        self.assertEqual(enrich._image_candidates(self._info(urls)), urls)

    def test_sniffs_real_bytes_not_the_file_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cases = {
                "a.png": (b"\x89PNG\r\n\x1a\n" + b"\0" * 16, True),
                "b.png": (b"\xff\xd8\xff\xe0" + b"\0" * 16, True),      # jpeg
                "c.png": (b"RIFF\0\0\0\0WEBPVP8 " + b"\0" * 8, True),   # webp
                # An mp4 and a webm wearing a .png name — the reported bug.
                "d.png": (b"\0\0\0\x20ftypisom" + b"\0" * 12, False),
                "e.png": (b"\x1a\x45\xdf\xa3" + b"\0" * 16, False),
                "f.png": (b"", False),
            }
            for name, (data, expected) in cases.items():
                path = tmp / name
                path.write_bytes(data)
                with self.subTest(name=name):
                    self.assertIs(enrich._looks_like_image(path), expected)


class TestDescriptionHtml(unittest.TestCase):
    """clean_html is the only thing between provider HTML and innerHTML."""

    def test_keeps_structure(self):
        out = sidecar.clean_html(
            "<h2>Usage</h2><p>Weight <b>0.7</b>.</p><ul><li>one</li><li>two</li></ul>")
        self.assertIn("<h2>Usage</h2>", out)
        self.assertIn("<b>0.7</b>", out)
        self.assertIn("<li>one</li>", out)

    def test_drops_scripts_and_keeps_no_handlers(self):
        out = sidecar.clean_html(
            '<p onclick="steal()">hi</p><script>alert(1)</script>'
            '<img src=x onerror="alert(1)">')
        self.assertNotIn("script", out.lower())
        self.assertNotIn("onclick", out.lower())
        self.assertNotIn("onerror", out.lower())
        self.assertIn("hi", out)

    def test_strips_attributes_from_allowed_tags(self):
        out = sidecar.clean_html('<p style="position:fixed" class="x" id="y">t</p>')
        self.assertEqual(out, "<p>t</p>")

    def test_link_schemes(self):
        safe = sidecar.clean_html('<a href="https://example.com">go</a>')
        self.assertIn('href="https://example.com"', safe)
        self.assertIn('rel="noopener noreferrer nofollow"', safe)
        for bad in ('javascript:alert(1)', 'data:text/html,<script>x</script>',
                    'vbscript:x', '/relative'):
            with self.subTest(href=bad):
                out = sidecar.clean_html(f'<a href="{bad}">go</a>')
                self.assertNotIn("href", out, f"{bad} should not survive")
                self.assertIn("go", out, "link text should still be kept")

    def test_unknown_tags_are_unwrapped_not_kept(self):
        out = sidecar.clean_html("<marquee><iframe>text</iframe></marquee>")
        self.assertNotIn("<", out.replace("&lt;", ""))
        self.assertIn("text", out)

    def test_stray_angle_bracket_is_escaped(self):
        out = sidecar.clean_html("<p>1 < 2 and 3 > 2</p>")
        self.assertIn("&lt; 2", out)

    def test_empty_input(self):
        self.assertEqual(sidecar.clean_html(""), "")
        self.assertEqual(sidecar.clean_html(None), "")

    def test_html_is_found_under_model_when_top_level_is_flattened(self):
        """Regression: real sidecars hold the description twice.

        write_metadata_sidecar puts the flattened text at the top level for the
        WebUI and leaves the provider's markup nested under `model`. Preferring
        the first non-empty candidate therefore always picked the flat copy, so
        the formatting fix silently did nothing on every existing library.
        """
        fields = sidecar.normalise({
            "name": "N", "baseModel": "SDXL",
            "description": "Order a commission here! Support my work by ...",
            "model": {"name": "N",
                      "description": "<p>Order a commission here!</p>"
                                     "<ul><li>Support my work</li></ul>"},
        }, "n.safetensors")
        self.assertIn("<ul>", fields["description_html"])
        self.assertIn("<li>Support my work</li>", fields["description_html"])
        self.assertNotIn("<", fields["description"])

    def test_richest_prefers_markup_over_order(self):
        self.assertIn("<p>", sidecar._richest("plain text", "<p>rich</p>"))
        self.assertIn("<p>", sidecar._richest("<p>rich</p>", "plain text"))
        # No markup anywhere: first non-empty wins.
        self.assertEqual(sidecar._richest("", "first", "second"), "first")
        self.assertEqual(sidecar._richest("", None), "")

    def test_refresh_without_markup_keeps_markup_on_disk(self):
        """A provider that returns no HTML must not blank what is already there."""
        model = Path(self.__class__.__name__)  # placeholder, replaced below
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            m = d / "n.safetensors"
            m.write_bytes(b"x")
            sidecar.write_metadata_sidecar(
                m, {"name": "N", "description_html": "<p>kept</p>"}, raw={})
            sidecar.write_metadata_sidecar(
                m, {"name": "N", "description_html": ""}, raw={})
            data = json.loads((d / "n.json").read_text(encoding="utf-8"))
            self.assertEqual(data["description_html"], "<p>kept</p>")

    def test_normalise_produces_both_forms(self):
        fields = sidecar.normalise(
            {"description": "<h3>Hi</h3><p>Line one</p>", "name": "N",
             "baseModel": "SDXL"}, "n.safetensors")
        self.assertIn("<h3>Hi</h3>", fields["description_html"])
        # The plain field stays plain — the WebUI and the search index read it.
        self.assertNotIn("<", fields["description"])
        self.assertIn("Hi", fields["description"])


class _StubHandle:
    """Enough of JobHandle for a task body to run outside the queue."""

    def __init__(self):
        self.logs = []

    def check_cancelled(self): pass
    def set_total(self, total): pass
    def advance(self, detail=None): pass
    def progress(self, current=None, message=None, detail=None): pass
    def log(self, message, level="info"): self.logs.append((level, message))


class TestFilingLooseModels(LibraryFixture):
    """Loose models must end up in their own folder, taking their files along."""

    def _loose_model(self):
        self.make_model("kanna.safetensors", preview=True, meta={
            "model": {"name": "Kanna", "creator": {"username": "someone"}},
            "baseModel": "SDXL", "id": 1, "modelId": 10})
        self.rescan()
        enrich.ingest_sidecars(self.conn, self.root, only_unknown=False)

    def test_file_loose_models_moves_the_model_and_its_assets(self):
        from atlas import tasks
        self._loose_model()
        stats = tasks.file_loose_models(_StubHandle())
        self.assertEqual(stats["moved_to_folders"], 1)
        self.assertEqual(stats["failed_moves"], 0)
        # Nothing left loose at the root.
        loose = [p.name for p in self.root.iterdir() if p.is_file()]
        self.assertEqual(loose, [], f"still loose at root: {loose}")
        # The model, its sidecar and its preview all travelled together.
        moved = list(self.root.glob("*/kanna.safetensors"))
        self.assertEqual(len(moved), 1)
        folder = moved[0].parent
        self.assertTrue((folder / "kanna.json").is_file(), "sidecar left behind")
        self.assertTrue((folder / "kanna.png").is_file(), "preview left behind")

    def test_update_everything_includes_the_filing_stage(self):
        """Regression: Update Everything fetched metadata but never filed
        anything, so freshly-onboarded models stayed loose at the root."""
        from atlas import tasks
        import inspect
        source = inspect.getsource(tasks.full_sync)
        self.assertIn("file_loose_models", source)
        # And the stage list actually runs it.
        self.assertIn("Filing loose models", source)

    def test_filing_is_a_no_op_in_dry_run(self):
        from atlas import tasks
        self._loose_model()
        config.set_many({"dry_run": True})
        try:
            stats = tasks.file_loose_models(_StubHandle())
        finally:
            config.set_many({"dry_run": False})
        self.assertEqual(stats["moved_to_folders"], 0)
        self.assertTrue((self.root / "kanna.safetensors").is_file(),
                        "dry run must not move anything")


class TestCivitaiBase(LibraryFixture):

    def test_default_base_is_civitai_red(self):
        from atlas import providers
        provider = providers.get("civitai")
        self.assertEqual(provider.base, "https://civitai.red")
        self.assertIn("civitai.red", provider.api)
        self.assertIn("civitai.red", provider.download_url("1", "2"))

    def test_base_is_configurable(self):
        from atlas import providers
        provider = providers.get("civitai")
        config.set_many({"civitai_base": "civitai.com"})
        try:
            self.assertEqual(provider.base, "https://civitai.com")
            self.assertIn("civitai.com", provider.api)
            self.assertIn("civitai.com", provider.download_url("1", "2"))
        finally:
            config.set_many({"civitai_base": ""})

    def test_api_downloads_and_links_share_one_host(self):
        from atlas import providers
        provider = providers.get("civitai")
        # Metadata, downloads and page links must never split across hosts.
        self.assertEqual(provider.api.split("/api")[0], provider.base)
        self.assertTrue(provider.download_url("1", "2").startswith(provider.base))

    def test_key_still_travels_in_a_header_not_the_url(self):
        from atlas import providers
        provider = providers.get("civitai")
        self.assertNotIn("token=", provider.download_url("1", "2"))
        self.assertEqual(
            provider.download_headers("SECRET")["Authorization"], "Bearer SECRET")

    def test_legacy_mirror_setting_still_honoured(self):
        from atlas import providers
        provider = providers.get("civitai")
        config.set_many({"civitai_mirror": "civitai.com"})
        try:
            self.assertEqual(provider.base, "https://civitai.com")
        finally:
            config.set_many({"civitai_mirror": ""})


class TestProviders(unittest.TestCase):

    def test_registry_resolves_by_domain(self):
        from atlas import providers
        self.assertEqual(providers.from_url("https://civitai.com/models/1").name, "civitai")
        self.assertEqual(providers.from_url("https://arcenciel.io/models/1").name, "arcenciel")
        self.assertIsNone(providers.from_url("https://example.com/models/1"))

    def test_registry_returns_shared_instances(self):
        from atlas import providers
        self.assertIs(providers.get("civitai"), providers.get("civitai"),
                      "a fresh instance would reset the shared rate limiter")

    def test_download_url_never_contains_the_key(self):
        from atlas import providers
        for provider in providers.all_providers():
            with self.subTest(provider=provider.name):
                url = provider.download_url("10", "20")
                self.assertNotIn("token=", url)
                self.assertNotIn("api_key", url)

    def test_auth_travels_in_a_header(self):
        from atlas.providers.civitai import CivitaiProvider
        headers = CivitaiProvider()._headers("secret-key")
        self.assertEqual(headers["Authorization"], "Bearer secret-key")

    def test_arcenciel_normalisation(self):
        """Arcenciel's schema differs from CivitAI's throughout.

        Built from the live API response shape: title/versionName/
        activationTags/uploader.username, and {id,name} tag objects.
        """
        from atlas.providers.arcenciel import ArcencielProvider

        model = {
            "id": 12762,
            "title": "Alice | Genshin Impact | Character LoRA",
            "description": "A description.",
            "type": "LORA",
            "tags": [{"id": 1, "name": "character"}, {"id": 2, "name": "genshin"}],
            "uploader": {"id": 1, "username": "FallenIncursio"},
        }
        version = {
            "id": 14033,
            "versionName": "v1",
            "baseModel": "NoobAI Eps",
            "sha256": "87D8A98367355B52",
            "sha256webui": "927294D416A90749",
            "fileName": "Alice_illusXLNoobAI_Incrs_v1.safetensors",
            "activationTags": ["Default | mature female, blonde hair",
                               "Outfit | white dress, red dress"],
            "images": [{"filePath": "images/a.png", "width": 1456, "height": 2128}],
            "publishedAt": "2025-01-01",
        }
        info = ArcencielProvider()._to_info(model, version)
        self.assertEqual(info.provider, "arcenciel")
        self.assertEqual(info.provider_model_id, "12762")
        self.assertEqual(info.provider_version_id, "14033")
        self.assertEqual(info.name, "Alice | Genshin Impact | Character LoRA")
        self.assertEqual(info.version_name, "v1")
        self.assertEqual(info.creator, "FallenIncursio")
        self.assertEqual(info.base_model, "NoobAI Eps")
        self.assertEqual(info.tags, ["character", "genshin"])
        self.assertEqual(info.file_sha256, "87d8a98367355b52")
        # Group labels are stripped; each entry stays one group.
        self.assertEqual(info.trained_words,
                         ["mature female, blonde hair", "white dress, red dress"])
        self.assertEqual(info.preview_urls,
                         ["https://arcenciel.io/uploads/images/a.png"])
        self.assertEqual(info.page_url, "https://arcenciel.io/models/12762")

    def test_arcenciel_picks_version_by_either_hash(self):
        from atlas.providers.arcenciel import ArcencielProvider

        model = {"versions": [
            {"id": 1, "sha256": "aaaa", "sha256webui": "bbbb", "publishedAt": "2024"},
            {"id": 2, "sha256": "cccc", "sha256webui": "dddd", "publishedAt": "2025"},
        ]}
        pick = ArcencielProvider._pick_version
        self.assertEqual(pick(model, sha256="cccc")["id"], 2)
        # sha256webui is the AutoV3-style hash and must match too.
        self.assertEqual(pick(model, sha256="bbbb")["id"], 1)
        self.assertEqual(pick(model, version_id="1")["id"], 1)
        self.assertIsNone(pick(model, version_id="99"))
        # No hint at all -> newest.
        self.assertEqual(pick(model)["id"], 2)

    def test_arcenciel_download_url_shape(self):
        from atlas import providers
        url = providers.get("arcenciel").download_url("12762", "14033")
        self.assertEqual(
            url, "https://arcenciel.io/api/models/12762/versions/14033/download")

    def test_civitai_normalisation(self):
        from atlas.providers.civitai import CivitaiProvider
        info = CivitaiProvider()._version_to_info({
            "id": 55, "name": "v3", "baseModel": "Pony",
            "trainedWords": ["word"],
            "model": {"id": 7, "name": "Thing", "creator": {"username": "creator"},
                      "tags": ["tag"], "nsfw": False},
            "files": [{"primary": True, "name": "thing.safetensors",
                       "hashes": {"SHA256": "AABB"}}],
            "images": [{"url": "https://img/1.png"}],
        })
        self.assertIsInstance(info, ModelInfo)
        self.assertEqual(info.provider_version_id, "55")
        self.assertEqual(info.provider_model_id, "7")
        self.assertEqual(info.name, "Thing")
        self.assertEqual(info.file_sha256, "aabb")
        self.assertEqual(info.preview_urls, ["https://img/1.png"])

    def test_rate_limiter_paces_requests(self):
        import time
        limiter = RateLimiter(rate_per_sec=20, burst=1)
        limiter.acquire()
        started = time.monotonic()
        limiter.acquire()
        limiter.acquire()
        self.assertGreater(time.monotonic() - started, 0.05)


# ---------------------------------------------------------------------------
# Config / secrets
# ---------------------------------------------------------------------------

class TestConfigSecrets(LibraryFixture):

    def test_public_settings_never_include_a_key(self):
        config.set_secret("civitai", "super-secret-value")
        public = config.public_settings()
        self.assertNotIn("civitai_api_key", public)
        self.assertNotIn("super-secret-value", json.dumps(public))

    def test_environment_wins_over_file(self):
        config.set_secret("civitai", "from-file")
        os.environ["ATLAS_CIVITAI_API_KEY"] = "from-env"
        try:
            self.assertEqual(config.get_secret("civitai"), "from-env")
            self.assertEqual(config.secret_source("civitai"), "env")
        finally:
            del os.environ["ATLAS_CIVITAI_API_KEY"]
        self.assertEqual(config.secret_source("civitai"), "file")

    def test_unset_secret_reports_unset(self):
        self.assertEqual(config.secret_source("arcenciel"), "unset")


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

class TestOpenFolder(LibraryFixture):
    """Revealing a model in the file manager."""

    def setUp(self):
        super().setUp()
        import os
        import subprocess
        from atlas import api

        self.api = api
        # Never actually launch a file manager during tests.
        self._popen, self._startfile = subprocess.Popen, getattr(os, "startfile", None)
        subprocess.Popen = lambda *a, **k: None
        if hasattr(os, "startfile"):
            os.startfile = lambda *a, **k: None

    def tearDown(self):
        import os
        import subprocess
        subprocess.Popen = self._popen
        if self._startfile is not None:
            os.startfile = self._startfile
        super().tearDown()

    def _open(self, path):
        req = self.api.Request("POST", "/api/open-folder", {}, {"path": path},
                               ["api", "open-folder"])
        return self.api.open_folder(req)

    def test_relative_model_path_is_anchored_to_the_library(self):
        """Regression: the gallery sends library-relative paths.

        Resolving one against the process working directory put it outside the
        library, so the endpoint rejected its own models with
        "path is outside the library and backup folders".
        """
        self.make_model("Anime/Rin/m.safetensors")
        result = self._open("Anime/Rin/m.safetensors")
        self.assertTrue(result["ok"])
        self.assertTrue(result["opened"].endswith("Rin"))

    def test_absolute_path_inside_the_library_is_allowed(self):
        self.make_model("Anime/Rin/m.safetensors")
        result = self._open(str(self.root / "Anime" / "Rin" / "m.safetensors"))
        self.assertTrue(result["ok"])

    def test_paths_outside_the_library_are_refused(self):
        self.make_model("Anime/Rin/m.safetensors")
        for attack in ["../../../../Windows", "C:/Windows", "/etc"]:
            with self.subTest(attack=attack):
                with self.assertRaises(self.api.ApiError) as caught:
                    self._open(attack)
                self.assertEqual(caught.exception.status, 403)


class TestServerSafety(unittest.TestCase):

    def test_safe_join_blocks_traversal(self):
        from atlas.server import _safe_join
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "ok.txt").write_text("fine", encoding="utf-8")
            self.assertIsNotNone(_safe_join(base, "ok.txt"))
            for attack in ["../secret", "../../etc/passwd", "a/../../out",
                           "..\\..\\windows", ""]:
                with self.subTest(attack=attack):
                    self.assertIsNone(_safe_join(base, attack))

    def test_no_options_handler_exists(self):
        from atlas.server import AtlasHandler
        self.assertFalse(hasattr(AtlasHandler, "do_OPTIONS"),
                         "a CORS preflight handler would defeat the origin isolation")

    def test_api_routes_resolve(self):
        from atlas import api
        self.assertIs(api.resolve("GET", ["api", "models"]), api.list_models)
        self.assertIs(api.resolve("GET", ["api", "models", "7"]), api.get_model)
        self.assertIs(api.resolve("POST", ["api", "models", "7", "user"]), api.update_user_data)
        self.assertIsNone(api.resolve("GET", ["api", "nope"]))
        self.assertIsNone(api.resolve("PUT", ["api", "models"]))

    def test_no_route_returns_an_api_key(self):
        """Structural guard: nothing in the API surface may emit a secret."""
        from atlas import api
        for _method, pattern, handler in api.ROUTES:
            source = (handler.__doc__ or "") + handler.__name__
            self.assertNotIn("get_secret", source)


# ---------------------------------------------------------------------------
# Regressions found in review: each of these failed before its fix.
# ---------------------------------------------------------------------------

class TestReviewRegressions(LibraryFixture):

    def _req(self, method, segments, body=None, query_params=None):
        from atlas import api
        return api.Request(method, "/" + "/".join(segments),
                           query_params or {}, body or {}, segments)

    def _rate(self, file_id, **fields):
        from atlas import api
        api.update_user_data(self._req(
            "POST", ["api", "models", str(file_id), "user"], fields))

    def _id(self, rel):
        return self.conn.execute(
            "SELECT id FROM model_file WHERE path=?", (rel,)).fetchone()["id"]

    def _user(self, rel):
        return self.conn.execute(
            "SELECT u.rating, u.notes FROM model_file f "
            "LEFT JOIN user_data u ON u.file_id=f.id WHERE f.path=?", (rel,)).fetchone()

    # -- rename outside Atlas ---------------------------------------------

    def test_rename_in_a_shared_folder_keeps_user_data(self):
        """End to end, no hand-inserted orphan rows: rate, rename, rescan.

        A folder holding several versions has no shared user_data.json to fall
        back on, so before the fix the rating was simply gone.
        """
        self.make_model("Chars/rin_v1.safetensors", b"version one")
        self.make_model("Chars/rin_v2.safetensors", b"version two")
        self.rescan()
        self._rate(self._id("Chars/rin_v1.safetensors"), rating=5, notes="my notes")

        (self.root / "Chars/rin_v1.safetensors").rename(
            self.root / "Chars/rin_old.safetensors")
        self.rescan()

        row = self._user("Chars/rin_old.safetensors")
        self.assertEqual((row["rating"], row["notes"]), (5, "my notes"))
        # Moved, not copied: the stale row no longer holds it.
        self.assertIsNone(self._user("Chars/rin_v1.safetensors")["rating"])

    def test_rename_carries_collection_membership(self):
        from atlas import api
        self.make_model("a/one.safetensors", b"one")
        self.make_model("a/two.safetensors", b"two!")
        self.rescan()
        fid = self._id("a/one.safetensors")
        self._rate(fid, rating=3)
        cid = api.create_collection(self._req("POST", ["api", "collections"],
                                              {"name": "Keep"}))["id"]
        api.update_membership(self._req("POST", ["api", "collections", str(cid), "members"],
                                        {"file_ids": [fid]}))
        (self.root / "a/one.safetensors").rename(self.root / "a/uno.safetensors")
        self.rescan()
        members = [r["path"] for r in self.conn.execute(
            "SELECT f.path FROM collection_member cm JOIN model_file f ON f.id=cm.file_id "
            "WHERE cm.collection_id=?", (cid,))]
        self.assertEqual(members, ["a/uno.safetensors"])

    def test_move_across_drives_is_matched_by_hash(self):
        """A cross-drive move changes mtime; the hash pass still reunites it."""
        path = self.make_model("a/one.safetensors", b"same bytes")
        self.make_model("a/two.safetensors", b"other!!!!!")
        self.rescan()
        hashing.hash_pending(self.conn, self.root)
        self._rate(self._id("a/one.safetensors"), rating=4)

        data = path.read_bytes()
        path.unlink()
        moved = self.root / "b/one.safetensors"
        moved.parent.mkdir()
        moved.write_bytes(data)
        os.utime(moved, (1_000_000, 1_000_000))
        self.rescan()
        self.assertIsNone(self._user("b/one.safetensors")["rating"])
        hashing.hash_pending(self.conn, self.root)
        self.assertEqual(self._user("b/one.safetensors")["rating"], 4)

    def test_ambiguous_size_and_mtime_match_is_not_guessed(self):
        a = self.make_model("x/a.safetensors", b"same")
        b = self.make_model("x/b.safetensors", b"same")
        os.utime(a, (2_000_000, 2_000_000))
        os.utime(b, (2_000_000, 2_000_000))
        self.make_model("x/c.safetensors", b"anchor")
        self.rescan()
        self._rate(self._id("x/a.safetensors"), rating=5)
        self._rate(self._id("x/b.safetensors"), rating=1)
        a.rename(self.root / "x/a2.safetensors")
        b.rename(self.root / "x/b2.safetensors")
        self.rescan()
        self.assertIsNone(self._user("x/a2.safetensors")["rating"])
        self.assertIsNone(self._user("x/b2.safetensors")["rating"])

    # -- delete model ------------------------------------------------------

    def test_delete_model_keeps_unrelated_files_in_its_folder(self):
        from atlas import api
        self.make_model("Chars/rin.safetensors", preview=True)
        (self.root / "Chars/my_prompts.txt").write_text("hand written")
        (self.root / "Chars/rin.civitai.info").write_text("{}")
        self.rescan()
        out = api.delete_model(self._req(
            "DELETE", ["api", "models", str(self._id("Chars/rin.safetensors"))]))
        self.assertTrue((self.root / "Chars/my_prompts.txt").exists())
        self.assertFalse((self.root / "Chars/rin.png").exists())
        self.assertFalse((self.root / "Chars/rin.civitai.info").exists())
        self.assertIn("my_prompts.txt", out["kept"])
        self.assertFalse(out["folder_removed"])

    def test_delete_model_still_removes_an_emptied_folder(self):
        from atlas import api
        self.make_model("Chars/rin.safetensors", preview=True)
        (self.root / "Chars/preview.png").write_bytes(b"img")
        self.rescan()
        out = api.delete_model(self._req(
            "DELETE", ["api", "models", str(self._id("Chars/rin.safetensors"))]))
        self.assertTrue(out["folder_removed"])
        self.assertFalse((self.root / "Chars").exists())

    # -- cleanup -----------------------------------------------------------

    def test_cleanup_leaves_user_files_beside_models(self):
        from atlas import organizer
        self.make_model("Chars/rin.safetensors", preview=True)
        (self.root / "Chars/README.txt").write_text("mine")
        (self.root / "Chars/showcase_01.png").write_bytes(b"img")
        self.rescan()
        self.assertEqual(organizer.find_orphans(self.conn, self.root), [])

    def test_cleanup_still_finds_a_renamed_models_sidecars(self):
        from atlas import organizer
        self.make_model("Chars/rin.safetensors", preview=True)
        for name in ("oldname.json", "oldname.civitai.info", "oldname.preview.png"):
            (self.root / "Chars" / name).write_bytes(b"{}")
        self.rescan()
        self.assertEqual(organizer.find_orphans(self.conn, self.root), [
            "Chars/oldname.civitai.info", "Chars/oldname.json", "Chars/oldname.preview.png"])

    def test_cleanup_leaves_a_folder_of_reference_images(self):
        from atlas import organizer
        self.make_model("a/one.safetensors")
        refs = self.root / "Refs"
        refs.mkdir()
        (refs / "pose1.png").write_bytes(b"img")
        (refs / "pose2.jpg").write_bytes(b"img")
        self.rescan()
        self.assertEqual(organizer.find_orphans(self.conn, self.root), [])

    # -- collections ---------------------------------------------------------

    def test_creating_a_collection_twice_keeps_one(self):
        from atlas import api
        first = api.create_collection(self._req("POST", ["api", "collections"], {"name": "Anime"}))
        second = api.create_collection(self._req("POST", ["api", "collections"], {"name": "Anime"}))
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM collection WHERE name='Anime'").fetchone()[0], 1)

    def test_existing_duplicate_collections_are_merged_on_open(self):
        self.make_model("a/one.safetensors")
        self.make_model("a/two.safetensors", b"two")
        self.rescan()
        ids = [r["id"] for r in self.conn.execute("SELECT id FROM model_file ORDER BY path")]
        with db.transaction(self.conn):
            self.conn.execute("DROP INDEX ux_collection_name")
            for fid in ids:
                cur = self.conn.execute(
                    "INSERT INTO collection(name, created_at) VALUES ('Dup', ?)", (db.now(),))
                self.conn.execute(
                    "INSERT INTO collection_member(collection_id, file_id) VALUES (?,?)",
                    (cur.lastrowid, fid))
        db.close_thread_connections()
        db._initialised.clear()
        self.conn = db.connect()
        rows = self.conn.execute(
            "SELECT c.id, COUNT(cm.file_id) n FROM collection c "
            "LEFT JOIN collection_member cm ON cm.collection_id=c.id "
            "WHERE c.name='Dup' GROUP BY c.id").fetchall()
        self.assertEqual([r["n"] for r in rows], [2])

    # -- thumbnails ----------------------------------------------------------

    def test_cached_thumbnail_is_served_without_pillow(self):
        """A Python without Pillow must still use thumbnails already on disk."""
        from atlas import thumbnails
        source = self.root / "a/one.png"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        cached = thumbnails.thumb_path(source, 320)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"RIFF....WEBP")
        saved = thumbnails.HAVE_PILLOW
        thumbnails.HAVE_PILLOW = False
        try:
            self.assertEqual(thumbnails.get_or_create(source, 320), cached)
            self.assertIsNone(thumbnails.get_or_create(source, 640),
                              "nothing cached at this size and no Pillow to make it")
        finally:
            thumbnails.HAVE_PILLOW = saved

    # -- bad parameters ----------------------------------------------------

    def test_non_numeric_filters_are_a_400(self):
        from atlas import api
        for params in ({"min_rating": "abc"}, {"collection": "x"}):
            with self.assertRaises(api.ApiError) as ctx:
                api.list_models(self._req("GET", ["api", "models"], query_params=params))
            self.assertEqual(ctx.exception.status, 400)
        api.list_models(self._req("GET", ["api", "models"], query_params={"min_rating": "3"}))

    def test_non_numeric_member_ids_are_a_400(self):
        from atlas import api
        cid = api.create_collection(self._req("POST", ["api", "collections"], {"name": "C"}))["id"]
        with self.assertRaises(api.ApiError) as ctx:
            api.update_membership(self._req(
                "POST", ["api", "collections", str(cid), "members"], {"file_ids": ["x"]}))
        self.assertEqual(ctx.exception.status, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
