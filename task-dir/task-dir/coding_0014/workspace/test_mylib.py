import unittest
from mylib import *

class T(unittest.TestCase):
    def test_wc(self):
        self.assertEqual(word_count("Hello, hello world!"), {"hello": 2, "world": 1})
        self.assertEqual(word_count("a\nb a"), {"a": 2, "b": 1})


if __name__ == "__main__":
    unittest.main()
