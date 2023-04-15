import sys
sys.path.append('../')
import unittest
from test_all_castle import TestCastleAll


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(TestCastleAll('test_ncpol'))
