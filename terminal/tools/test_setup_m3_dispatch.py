"""Tests for who setup.py sends to the MiniMax-M3.1-Flash path (m3_wanted, no GPU, no network, nothing installed):
the family says it; a size name says it only while no other family table claims that name. UD-Q2_K_XL and
UD-Q3_K_XL are Unsloth's names that the qwen MODELS table and Inkling-2-Mini use too, so those names must not
hijack a run into compiling strata-m3.

    python -m unittest tools.test_setup_m3_dispatch
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

M3_SIZES = setup.M3_FAMILIES["minimax-m3"]["models"]


def ask(family=None, model=None):
    """The two fields setup.main() reads at the strata-m3 branch."""
    return SimpleNamespace(family=family, model=model)


class M3Dispatch(unittest.TestCase):
    def test_the_family_names_it(self):
        self.assertTrue(setup.m3_wanted(ask("minimax-m3")))
        self.assertTrue(setup.m3_wanted(ask("minimax-m3", "UD-Q2_K_XL")))

    def test_a_size_only_minimax_m3_has_is_enough_on_its_own(self):
        self.assertNotIn("AD-Q2_K", setup.MODELS)
        self.assertTrue(setup.m3_wanted(ask(None, "AD-Q2_K")))

    def test_a_shared_size_does_not_hijack_another_family(self):
        for family in ("qwen", "inkling-2-mini", "hy4-flash", "qwen36"):
            for size in ("UD-Q2_K_XL", "UD-Q3_K_XL"):
                self.assertFalse(setup.m3_wanted(ask(family, size)), f"--family {family} --model {size} is not M3")

    def test_a_shared_size_with_no_family_stays_on_the_normal_path(self):
        self.assertIn("UD-Q2_K_XL", setup.MODELS)              # the name belongs to qwen: no M3
        self.assertFalse(setup.m3_wanted(ask(None, "UD-Q2_K_XL")))

    def test_other_families_and_nothing_are_not_m3(self):
        self.assertFalse(setup.m3_wanted(ask("qwen")))
        self.assertFalse(setup.m3_wanted(ask("qwen", "Q4_K_M")))
        self.assertFalse(setup.m3_wanted(ask(None, "Q4_K_M")))
        self.assertFalse(setup.m3_wanted(ask(None, None)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
