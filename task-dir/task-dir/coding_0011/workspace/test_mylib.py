import unittest
from mylib import *

class T(unittest.TestCase):
    def test_rev(self):
        self.assertEqual(reverse_words("a b c"), "c b a")
        self.assertEqual(reverse_words("hello"), "hello")
        self.assertEqual(reverse_words("one  two"), "two one")


if __name__ == "__main__":
    unittest.main()
