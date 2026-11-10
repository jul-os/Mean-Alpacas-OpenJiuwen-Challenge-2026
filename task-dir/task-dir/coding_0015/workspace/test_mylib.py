import unittest
from mylib import *

class T(unittest.TestCase):
    def test_counter(self):
        c = Counter(10)
        c.increment()
        c.increment()
        self.assertEqual(c.get(), 12)


if __name__ == "__main__":
    unittest.main()
