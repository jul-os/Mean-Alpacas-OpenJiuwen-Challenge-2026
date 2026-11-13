import unittest
from calc import total

class TestCalc(unittest.TestCase):
    def test_total(self):
        self.assertEqual(total([{"price": 10, "qty": 2}, {"price": 3, "qty": 1}]), 23)

if __name__ == "__main__":
    unittest.main()
