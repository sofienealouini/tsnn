# -*- coding: UTF-8 -*-

import sys
import os
import unittest
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from tsnn.introduction import introduction

class TestUtils(unittest.TestCase):

    # @unittest.skip('')
    def test_introduction(self):
        result = introduction()
        self.assertEqual(result, 10)


if __name__ == '__main__':
    unittest.main()
