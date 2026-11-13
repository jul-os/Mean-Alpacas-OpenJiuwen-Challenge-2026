import unittest
import mylib

class TestContradict(unittest.TestCase):
    def test_same_call_has_two_answers(self):
        self.assertEqual(mylib.answer(1), 1)
        self.assertEqual(mylib.answer(1), 2)

if __name__ == "__main__":
    unittest.main()
