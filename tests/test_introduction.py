import sys
import os
import unittest
from tsnn.introduction import introduction

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


class TestUtils(unittest.TestCase):

    def test_introduction(self):
        result = introduction()
        self.assertEqual(result, 10)


if __name__ == '__main__':
    unittest.main()
