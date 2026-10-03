"""The docs must list exactly the commands and keys the code has.

This exists because the reference once documented `/icons` and `/pretty` long
after both were removed, and was missing `/ask` and `/repeats` long after both
were added. A reference that is wrong in checkable ways is worse than a short
one, and "checkable" is the operative word: check it.
"""

import os
import re
import unittest

from _support import ROOT, load


class DocsMatchCode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.C = load("console")
        with open(os.path.join(ROOT, "docs", "reference.md")) as f:
            cls.doc = f.read()
        with open(os.path.join(ROOT, "llamatap")) as f:
            cls.cli = f.read()

    def test_every_command_is_documented(self):
        documented = set(re.findall(r"`/([a-z]+)", self.doc))
        for name, _args, _desc in self.C.COMMANDS:
            with self.subTest(command=name):
                self.assertIn(name, documented,
                              f"/{name} exists but is not in docs/reference.md")

    def test_no_command_is_documented_that_does_not_exist(self):
        real = {n for n, _, _ in self.C.COMMANDS}
        # Only look inside the palette table, so HTTP paths like /running and
        # file paths like /home/... are not mistaken for commands.
        table = self.doc[self.doc.index("## The command palette"):]
        table = table[:table.index("\n---")]
        for name in set(re.findall(r"`/([a-z]+)", table)):
            with self.subTest(command=name):
                self.assertIn(name, real,
                              f"/{name} is documented but does not exist")

    def test_every_key_is_documented(self):
        block = self.doc[self.doc.index("### Keys"):]
        block = block[:block.index("###", 10)]
        for line in self.C.HELP:
            token = line.split()[0]
            with self.subTest(key=token):
                self.assertIn(token, block,
                              f"the {token!r} key is not in the reference")

    def test_every_subcommand_is_documented(self):
        subs = set(re.findall(r"^\s+([a-z]+)\)", self.cli, re.M))
        subs -= {"settings"}        # an alias for config
        for sub in subs:
            with self.subTest(subcommand=sub):
                self.assertIn(f"llamatap {sub}", self.doc,
                              f"`llamatap {sub}` exists but is not documented")


if __name__ == "__main__":
    unittest.main()
