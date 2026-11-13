import unittest
from looplib import collect_even

class TestLoop(unittest.TestCase):
    def test_includes_last_item(self):
        self.assertEqual(collect_even([1, 2, 4]), [2, 4])

if __name__ == "__main__":
    unittest.main()
