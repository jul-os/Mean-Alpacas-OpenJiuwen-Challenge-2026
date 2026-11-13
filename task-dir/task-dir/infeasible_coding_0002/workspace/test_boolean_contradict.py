import unittest
import mylib

class TestContradict(unittest.TestCase):
    def test_same_call_is_true_and_false(self):
        self.assertTrue(mylib.enabled(1))
        self.assertFalse(mylib.enabled(1))

if __name__ == "__main__":
    unittest.main()
