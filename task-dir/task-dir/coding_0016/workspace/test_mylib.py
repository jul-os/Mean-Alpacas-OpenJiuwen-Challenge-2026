import unittest
from mylib import *

class T(unittest.TestCase):
    def test_gcd(self):
        self.assertEqual(gcd(12, 18), 6)
        self.assertEqual(gcd(7, 13), 1)
        self.assertEqual(gcd(462, 1071), 21)
    def test_lcm(self):
        self.assertEqual(lcm(4, 6), 12)
        self.assertEqual(lcm(21, 6), 42)


if __name__ == "__main__":
    unittest.main()
