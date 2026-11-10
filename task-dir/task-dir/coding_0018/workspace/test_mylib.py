import unittest
from mylib import *

class T(unittest.TestCase):
    def test_bs(self):
        self.assertEqual(binary_search([1, 3, 5, 7, 9], 1), 0)
        self.assertEqual(binary_search([1, 3, 5, 7, 9], 9), 4)
        self.assertEqual(binary_search([1, 3, 5, 7, 9], 5), 2)
        self.assertEqual(binary_search([1, 3, 5, 7, 9], 4), -1)
        self.assertEqual(binary_search([], 1), -1)


if __name__ == "__main__":
    unittest.main()
