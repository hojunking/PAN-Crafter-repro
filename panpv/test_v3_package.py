"""Synthetic, bounded ZIP intake checks; never touch private dataset artifacts."""
import hashlib
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile

from panpv.common import file_sha
from panpv.v3_package import safe_name, unpack


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="panpv_zip_test_")
        self.base = Path(self.temporary.name)
        self.archive = self.base / "input.zip"
        self.destination = self.base / "extracted"
        self.receipt = self.base / "receipt.json"
        self.payload = {
            "README.md": b"Synthetic private-input fixture.\n",
            "scenes/tiny.h5": b"Not real HDF5: transfer verification only.\x00\xff",
            # Import/execution is forbidden, irrespective of package provenance.
            "code/untrusted.py": b"raise RuntimeError('Never execute provider code')\n",
        }

    def tearDown(self):
        self.temporary.cleanup()

    def make_archive(self, *, replacements=None, omit=(), extras=(),
                     declared_extra=None, sums_override=None, manifest_schema=None,
                     duplicate_manifest=False):
        declared = [dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                    for name, data in self.payload.items()]
        if declared_extra:
            declared.append(declared_extra)
        if duplicate_manifest:
            declared.append(dict(declared[0]))
        manifest = dict(schema=manifest_schema or "PANPV_PORTABLE_PACKAGE_v1", files=declared)
        sums = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in declared)
        data = dict(self.payload)
        data.update({"manifest.json": json.dumps(manifest).encode(),
                     "SHA256SUMS": (sums if sums_override is None else sums_override).encode()})
        data.update(replacements or {})
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("fixture/", b"")
            for name, content in data.items():
                if name not in omit:
                    archive.writestr("fixture/" + name, content)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for name, content in extras:
                    archive.writestr(name, content)
        return self.archive

    def run_unpack(self, **kwargs):
        # Avoid coupling tests to host capacity, but exercise the explicit gate.
        with patch("panpv.v3_package.shutil.disk_usage",
                   return_value=SimpleNamespace(free=100 * 1024**3)):
            return unpack(self.archive, self.destination, self.receipt, **kwargs)

    def rejected(self, expression, *, pre_extraction=False):
        with self.assertRaisesRegex(ValueError, expression):
            self.run_unpack()
        self.assertFalse(self.receipt.exists())
        if pre_extraction:
            self.assertFalse(self.destination.exists())

    def test_valid_package_checks_crc_hashes_size_and_never_executes_code(self):
        self.make_archive()
        result = self.run_unpack(expected_bytes=self.archive.stat().st_size)
        self.assertEqual(result["state"], "TRANSFER_VERIFIED_NOT_TRAINING_ADMISSION")
        self.assertTrue(result["crc_verified"])
        self.assertTrue(result["manifest_verified"])
        self.assertFalse(result["executed_provider_code"])
        self.assertFalse(result["training_started"])
        self.assertEqual(result["archive_sha256"], file_sha(self.archive))
        self.assertEqual(result["archive_bytes"], self.archive.stat().st_size)
        self.assertEqual(json.loads(self.receipt.read_text()), result)
        for name, content in self.payload.items():
            self.assertEqual((self.destination / "fixture" / name).read_bytes(), content)
            self.assertEqual(result["files"][name]["bytes"], len(content))
            self.assertEqual(result["files"][name]["sha256"], hashlib.sha256(content).hexdigest())

    def test_unsafe_names_are_rejected(self):
        for name in ("", "/absolute", "../outside", "fixture/../outside", "fixture/./file",
                     "fixture//file", "fixture\\outside", "//server/share"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe"):
                safe_name(name)

    def test_archive_traversal_is_rejected_before_extracting(self):
        self.make_archive(extras=[("fixture/../../outside.txt", b"unsafe")])
        self.rejected("Unsafe", pre_extraction=True)
        self.assertFalse((self.base / "outside.txt").exists())

    def test_zip_symlink_is_rejected_before_extracting(self):
        link = zipfile.ZipInfo("fixture/shortcut")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.make_archive(extras=[(link, b"../../outside")])
        self.rejected("Links and special", pre_extraction=True)

    def test_zip_special_file_is_rejected_before_extracting(self):
        fifo = zipfile.ZipInfo("fixture/fifo")
        fifo.create_system = 3
        fifo.external_attr = (stat.S_IFIFO | 0o600) << 16
        self.make_archive(extras=[(fifo, b"")])
        self.rejected("Links and special", pre_extraction=True)

    def test_duplicate_archive_member_is_rejected(self):
        self.make_archive(extras=[("fixture/README.md", b"replacement")])
        self.rejected("Duplicate package member", pre_extraction=True)

    def test_directory_file_alias_is_rejected(self):
        self.make_archive(extras=[("fixture", b"alias to top-level directory")])
        self.rejected("Duplicate package member", pre_extraction=True)

    def test_multiple_archive_roots_are_rejected(self):
        self.make_archive(extras=[("other/file.txt", b"extra")])
        self.rejected("Unexpected archive root", pre_extraction=True)

    def test_unlisted_extra_file_is_rejected_without_admission(self):
        self.make_archive(extras=[("fixture/not-listed.txt", b"extra")])
        self.rejected("Unlisted or missing")
        self.assertTrue(self.destination.exists())  # visibly incomplete, not silently reused

    def test_missing_declared_file_is_rejected(self):
        self.make_archive(omit=("README.md",))
        self.rejected("Unlisted or missing")

    def test_sha_mismatch_is_rejected_even_when_size_matches(self):
        original = self.payload["README.md"]
        self.make_archive(replacements={"README.md": b"X" * len(original)})
        self.rejected("Package size/SHA mismatch")

    def test_size_mismatch_is_rejected(self):
        self.make_archive(replacements={"README.md": self.payload["README.md"] + b"extra"})
        self.rejected("Package size/SHA mismatch")

    def test_sha256sums_must_agree_with_manifest(self):
        self.make_archive(sums_override="0" * 64 + "  README.md\n")
        self.rejected("SHA256SUMS and package manifest disagree")

    def test_invalid_checksum_is_rejected(self):
        self.make_archive(sums_override="g" * 64 + "  README.md\n")
        self.rejected("Invalid checksum manifest")

    def test_duplicate_manifest_path_is_rejected(self):
        self.make_archive(duplicate_manifest=True)
        self.rejected("Duplicate manifest member")

    def test_manifest_traversal_is_rejected(self):
        self.make_archive(declared_extra=dict(path="../outside", bytes=0, sha256="0" * 64))
        self.rejected("Unsafe")

    def test_unknown_manifest_schema_is_rejected(self):
        self.make_archive(manifest_schema="NOT_THIS_PACKAGE")
        self.rejected("Unsupported package manifest")

    def test_wrong_archive_length_is_rejected_before_extraction(self):
        self.make_archive()
        with self.assertRaisesRegex(ValueError, "length differs"):
            self.run_unpack(expected_bytes=self.archive.stat().st_size + 1)
        self.assertFalse(self.destination.exists())

    def test_archive_symlink_is_rejected(self):
        source = self.make_archive()
        link = self.base / "symlink.zip"
        link.symlink_to(source)
        self.archive = link
        self.rejected("regular local file", pre_extraction=True)

    def test_existing_destination_is_preserved(self):
        self.make_archive()
        self.destination.mkdir()
        marker = self.destination / "prior-input"
        marker.write_bytes(b"preserve original")
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            self.run_unpack()
        self.assertEqual(marker.read_bytes(), b"preserve original")
        self.assertEqual(list(self.destination.iterdir()), [marker])

    def test_existing_receipt_is_preserved(self):
        self.make_archive()
        self.receipt.write_bytes(b"preserve receipt")
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            self.run_unpack()
        self.assertEqual(self.receipt.read_bytes(), b"preserve receipt")
        self.assertFalse(self.destination.exists())

    def test_dangling_destination_symlink_is_preserved(self):
        self.make_archive()
        self.destination.symlink_to(self.base / "not-existing-target", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            self.run_unpack()
        self.assertTrue(self.destination.is_symlink())

    def test_dangling_receipt_symlink_is_preserved(self):
        self.make_archive()
        self.receipt.symlink_to(self.base / "not-existing-receipt-target")
        with self.assertRaises(ValueError):
            self.run_unpack()
        self.assertTrue(self.receipt.is_symlink())
        self.assertFalse(self.destination.exists())

    def test_receipt_cannot_overwrite_newly_extracted_source(self):
        self.make_archive()
        self.receipt = self.destination / "fixture" / "README.md"
        with self.assertRaises(ValueError):
            self.run_unpack()
        self.assertFalse(self.destination.exists())

    def test_insufficient_reserved_capacity_rejects_before_extraction(self):
        self.make_archive()
        with zipfile.ZipFile(self.archive) as source:
            size = sum(item.file_size for item in source.infolist())
        with patch("panpv.v3_package.shutil.disk_usage",
                   return_value=SimpleNamespace(free=size + 10 * 1024**3 - 1)):
            with self.assertRaisesRegex(ValueError, "Insufficient space"):
                unpack(self.archive, self.destination, self.receipt)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.receipt.exists())


if __name__ == "__main__":
    unittest.main()
