import unittest
from mylib import *

class T(unittest.TestCase):
    def test_mi(self):
        self.assertEqual(merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]), [(1, 6), (8, 10), (15, 18)])
        self.assertEqual(merge_intervals([(1, 4), (4, 5)]), [(1, 5)])
        self.assertEqual(merge_intervals([]), [])


if __name__ == "__main__":
    unittest.main()
