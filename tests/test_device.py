"""Device naming: the slug in state file names, and which device a server gets."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isolation  # noqa: E402,F401

from bench import device, state


class ShortTest(unittest.TestCase):
    def test_slugs(self):
        cases = {"MacBook Pro M2 Max 64GB": "MBP-M2Max-64GB", "MacBook Air M4 32GB": "MBA-M4-32GB",
                 "Linux Mint 22.3 32GB NVIDIA GeForce GTX 1080 8192MiB": "LinuxMint22.3-32GB-GTX1080-8192MiB",
                 "Mac Studio M3 Ultra 512GB": "MacStudio-M3Ultra-512GB"}
        for name, want in cases.items():
            self.assertEqual(device.short(name), want)

    def test_empty(self):
        self.assertIsNone(device.short(""))


class ServerTest(unittest.TestCase):
    def tearDown(self):
        device.configure(None)

    def test_flag_wins(self):
        device.configure("MacBook Air M4 32GB")
        self.assertEqual(device.server("http://far:1/v1", {"details": {"chip": "M1"}}),
                         {"name": "MacBook Air M4 32GB", "source": "device-flag"})

    def test_server_reported_chip(self):
        self.assertEqual(device.server("http://far:1/v1", {"details": {"chip": "Apple M3 Max"}})["source"], "server")

    def test_local_endpoint_gets_this_machine(self):
        with mock.patch.object(device, "local", return_value="Test Box 8GB"):
            self.assertEqual(device.server("http://127.0.0.1:8080/v1"), {"name": "Test Box 8GB", "source": "local"})
            self.assertEqual(device.server("http://localhost:8080/v1")["name"], "Test Box 8GB")

    def test_remote_unknown(self):
        with mock.patch.object(device, "_host_names", return_value={"me"}):
            self.assertIsNone(device.server("http://other-host.local:8133/v1"))
            self.assertTrue(device.is_local("http://me.local:8133/v1"))

    def test_file_name_has_device(self):
        p = state.default_path("/nonexistent", {"kind": "splash", "version": "4b2dee69"}, "org/M", "conv-3",
                               dev={"name": "MacBook Pro M2 Max 64GB", "source": "local"})
        self.assertRegex(os.path.basename(p), r"^splash-4b2dee69_MBP-M2Max-64GB_org-M_conv-3_\d{8}-\d{6}\.jsonl$")
        p = state.default_path("/nonexistent", {"kind": "splash"}, "M", "smoke")
        self.assertRegex(os.path.basename(p), r"^splash_M_smoke_")


if __name__ == "__main__":
    unittest.main()
