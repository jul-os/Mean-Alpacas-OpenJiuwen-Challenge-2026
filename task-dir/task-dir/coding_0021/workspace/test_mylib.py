import unittest
from mylib import *

class T(unittest.TestCase):
    def test_eval(self):
        self.assertEqual(eval_expr("2+3*4"), 14)
        self.assertEqual(eval_expr("(2+3)*4"), 20)
        self.assertEqual(eval_expr("10-3-2"), 5)
        self.assertEqual(eval_expr("2*(3+(4-1))"), 12)


if __name__ == "__main__":
    unittest.main()
