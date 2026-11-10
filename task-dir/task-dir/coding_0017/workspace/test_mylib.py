import unittest
from mylib import *

class T(unittest.TestCase):
    def test_csv(self):
        self.assertEqual(parse_csv_line('a,b,c'), ["a", "b", "c"])
        self.assertEqual(parse_csv_line('a,"b,x",c'), ["a", "b,x", "c"])
        self.assertEqual(parse_csv_line('""'), [""])


if __name__ == "__main__":
    unittest.main()
