import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

sys.path.append(str(Path(__file__).parent / "src"))

from downloader import save_metadata_if_missing, merge_metadata, import_metadata_to_user_data_dict, load_json_lenient
from config import set_dry_run, is_complete_metadata


class TestMetadataCompatibility(unittest.TestCase):
    def setUp(self):
        set_dry_run(False)  # Ensure files are actually written

    def test_merge_metadata_preserves_unknown_keys(self):
        # Test 1: Metadata refresh preserves unknown keys.
        old_meta = {
            "creator": "old_creator",
            "activation text": "old_activation",
            "preferred weight": 0.8,
            "negative text": "bad hands",
            "notes": "some notes",
            "custom_field": "custom_val"
        }
        new_meta = {
            "creator": "new_creator",
            "description": "new description",
            "trainedWords": ["trigger1", "trigger2"]
        }
        merged = merge_metadata(old_meta, new_meta)
        
        # authoritative source from new_meta
        self.assertEqual(merged["creator"], "new_creator")
        self.assertEqual(merged["description"], "new description")
        self.assertEqual(merged["trainedWords"], ["trigger1", "trigger2"])
        
        # custom fields preserved
        self.assertEqual(merged["activation text"], "old_activation")
        self.assertEqual(merged["preferred weight"], 0.8)
        self.assertEqual(merged["negative text"], "bad hands")
        self.assertEqual(merged["notes"], "some notes")
        self.assertEqual(merged["custom_field"], "custom_val")

    def test_import_activation_text_only_if_empty(self):
        # Test 2: Activation text is copied into personal triggers only when they are empty.
        # Case A: Empty triggers
        user_data = {"personal_trigger_words": []}
        meta = {"activation text": "triggerA, triggerB"}
        modified = import_metadata_to_user_data_dict(meta, user_data)
        self.assertTrue(modified)
        self.assertEqual(user_data["personal_trigger_words"], ["triggerA", "triggerB"])

        # Case B: Non-empty triggers (should not overwrite)
        user_data2 = {"personal_trigger_words": ["user_trigger"]}
        meta2 = {"activation text": "new_trigger"}
        modified2 = import_metadata_to_user_data_dict(meta2, user_data2)
        self.assertFalse(modified2)
        self.assertEqual(user_data2["personal_trigger_words"], ["user_trigger"])

    def test_import_preferred_weight(self):
        # Test 3: Preferred weight is imported into user_data.json.
        # Case A: Missing in user_data
        user_data = {}
        meta = {"preferred weight": 0.85}
        modified = import_metadata_to_user_data_dict(meta, user_data)
        self.assertTrue(modified)
        self.assertEqual(user_data["preferred_weight"], 0.85)

        # Case B: Already exists (should not overwrite)
        user_data2 = {"preferred_weight": 0.7}
        meta2 = {"preferred weight": 0.9}
        modified2 = import_metadata_to_user_data_dict(meta2, user_data2)
        self.assertFalse(modified2)
        self.assertEqual(user_data2["preferred_weight"], 0.7)

    def test_import_negative_prompt_only_if_non_empty(self):
        # Test 4: Negative prompt is imported only if non-empty.
        # Case A: Empty negative text (ignore)
        user_data = {}
        meta = {"negative text": ""}
        modified = import_metadata_to_user_data_dict(meta, user_data)
        self.assertFalse(modified)
        self.assertNotIn("negative_prompt", user_data)

        # Case B: Non-empty negative text
        user_data2 = {}
        meta2 = {"negative text": "bad anatomy"}
        modified2 = import_metadata_to_user_data_dict(meta2, user_data2)
        self.assertTrue(modified2)
        self.assertEqual(user_data2["negative_prompt"], "bad anatomy")

    def test_existing_notes_and_triggers_not_overwritten(self):
        # Test 5: Existing notes and personal trigger words are never overwritten.
        user_data = {
            "personal_trigger_words": ["original_trigger"],
            "notes": "original notes"
        }
        meta = {
            "activation text": "new_trigger",
            "notes": "new notes"
        }
        modified = import_metadata_to_user_data_dict(meta, user_data)
        self.assertFalse(modified)
        self.assertEqual(user_data["personal_trigger_words"], ["original_trigger"])
        self.assertEqual(user_data["notes"], "original notes")

    def test_save_metadata_idempotent(self):
        # Test 6: Refreshing metadata multiple times produces identical results (idempotent behavior).
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_file = tmp_path / "my_model.safetensors"
            model_file.touch()

            # First save
            meta = {
                "id": 12345,
                "name": "My Model",
                "activation text": "act1",
                "custom_field": "val1"
            }
            res1 = save_metadata_if_missing(model_file, meta)
            self.assertEqual(res1["status"], "saved")

            sidecar_path = tmp_path / "my_model.json"
            self.assertTrue(sidecar_path.exists())
            
            with sidecar_path.open("r", encoding="utf-8") as f:
                data1 = json.load(f)

            # Second save (refresh with same or updated metadata)
            res2 = save_metadata_if_missing(model_file, meta)
            self.assertEqual(res2["status"], "saved")
            
            with sidecar_path.open("r", encoding="utf-8") as f:
                data2 = json.load(f)

            # Assert identical output
            self.assertEqual(data1, data2)

    def test_is_complete_metadata(self):
        # Case A: Minimal Forge/A1111 metadata (incomplete)
        minimal_forge = {
            "activation text": " Rin, Fate",
            "preferred weight": 0.85,
            "negative text": "bad hands",
            "notes": "rin test note"
        }
        self.assertFalse(is_complete_metadata(minimal_forge))

        # Case B: Complete CivitAI metadata (complete)
        complete_civitai = {
            "modelId": 12345,
            "id": 67890,
            "baseModel": "SD 1.5",
            "trainedWords": ["rin", "fate"]
        }
        self.assertTrue(is_complete_metadata(complete_civitai))

        # Case C: Wrapped complete metadata
        wrapped_civitai = {
            "ok": True,
            "data": {
                "modelId": 12345,
                "id": 67890,
                "baseModel": "SD 1.5",
                "trainedWords": ["rin", "fate"]
            }
        }
        self.assertTrue(is_complete_metadata(wrapped_civitai))

    def test_load_json_lenient_malformed(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            malformed_json_file = tmp_path / "malformed.json"
            
            # Write a malformed JSON string (missing comma on line 3)
            malformed_content = """{
  "id": 2871944,
  "modelId": 2376313
  "sd version": "Unknown",
  "activation text": "pcrgrace, black hair, blue hair",
  "preferred weight": 0.9,
  "negative text": "",
  "notes": ""
}"""
            malformed_json_file.write_text(malformed_content, encoding="utf-8")
            
            # Load using load_json_lenient
            data = load_json_lenient(malformed_json_file)
            
            self.assertEqual(data.get("id"), 2871944)
            self.assertEqual(data.get("modelId"), 2376313)
            self.assertEqual(data.get("sd version"), "Unknown")
            self.assertEqual(data.get("activation text"), "pcrgrace, black hair, blue hair")
            self.assertEqual(data.get("preferred weight"), 0.9)


if __name__ == "__main__":
    unittest.main()
