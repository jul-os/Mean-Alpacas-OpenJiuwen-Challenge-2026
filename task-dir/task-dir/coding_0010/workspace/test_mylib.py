import unittest
from mylib import *

class T(unittest.TestCase):
    def test_fact(self):
        self.assertEqual(factorial(0), 1)
        self.assertEqual(factorial(1), 1)
        self.assertEqual(factorial(5), 120)


if __name__ == "__main__":
    unittest.main()
