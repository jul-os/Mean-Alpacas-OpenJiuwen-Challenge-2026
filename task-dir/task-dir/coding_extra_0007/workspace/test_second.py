import unittest
from projectlib import answer
class TestSecond(unittest.TestCase):
    def test_answer_is_int(self):
        self.assertIsInstance(answer(), int)
if __name__ == "__main__":
    unittest.main()
