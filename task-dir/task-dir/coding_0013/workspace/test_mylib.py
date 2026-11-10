import unittest
from mylib import *

class T(unittest.TestCase):
    def test_dedup(self):
        self.assertEqual(remove_duplicates([3, 1, 3, 2, 1]), [3, 1, 2])
        self.assertEqual(remove_duplicates([]), [])
        self.assertEqual(remove_duplicates([5]), [5])


if __name__ == "__main__":
    unittest.main()
