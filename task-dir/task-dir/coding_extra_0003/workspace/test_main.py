import subprocess
import sys
import unittest

class TestMain(unittest.TestCase):
    def test_output(self):
        out = subprocess.check_output([sys.executable, "main.py"], text=True).strip()
        self.assertEqual(out, "10")

if __name__ == "__main__":
    unittest.main()
