import unittest
from broken import greet

class TestBroken(unittest.TestCase):
    def test_greet(self):
        self.assertEqual(greet("Ada"), "Hello, Ada")

if __name__ == "__main__":
    unittest.main()
