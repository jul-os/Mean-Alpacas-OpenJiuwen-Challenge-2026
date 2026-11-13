import unittest
from slow import wait_until_zero

class TestTimeout(unittest.TestCase):
    def test_finishes(self):
        self.assertEqual(wait_until_zero(1), 0)

if __name__ == "__main__":
    unittest.main()
