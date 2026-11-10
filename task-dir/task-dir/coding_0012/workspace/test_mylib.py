import unittest
from mylib import *

class T(unittest.TestCase):
    def test_fib(self):
        self.assertEqual(fib(10), 55)
        self.assertEqual(fib(90), 2880067194370816120)


if __name__ == "__main__":
    unittest.main()
