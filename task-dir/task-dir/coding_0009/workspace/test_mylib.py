import unittest
from mylib import *

class T(unittest.TestCase):
    def test_even(self):
        self.assertTrue(is_even(4))
        self.assertTrue(is_even(0))
        self.assertFalse(is_even(7))
        self.assertFalse(is_even(-3))


if __name__ == "__main__":
    unittest.main()
