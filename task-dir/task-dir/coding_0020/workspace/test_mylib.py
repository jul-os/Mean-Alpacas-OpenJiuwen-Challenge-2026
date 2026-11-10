import unittest
from mylib import *

class T(unittest.TestCase):
    def test_ways(self):
        self.assertEqual(ways(1), 1)
        self.assertEqual(ways(5), 8)
        self.assertEqual(ways(200), 453973694165307953197296969697410619233826)


if __name__ == "__main__":
    unittest.main()
