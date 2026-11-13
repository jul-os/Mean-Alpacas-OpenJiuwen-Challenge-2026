import unittest
from cart import total

class TestCart(unittest.TestCase):
    def test_discounted_total(self):
        self.assertEqual(total([(100, 2), (50, 1)], 10), 225)

if __name__ == "__main__":
    unittest.main()
