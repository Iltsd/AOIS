import unittest
import math
from floating_point import FloatingPoint


class TestFloatingPoint(unittest.TestCase):
    def setUp(self):
        self.fp = FloatingPoint()


    def test_convert_positive(self):
        test_num = 12.375
        binary = self.fp.convert_float_to_bin(test_num)
        result = self.fp.convert_bin_to_float(binary)
        self.assertAlmostEqual(test_num, result, places=6)

    def test_convert_negative(self):
        test_num = -3.75
        binary = self.fp.convert_float_to_bin(test_num)
        result = self.fp.convert_bin_to_float(binary)
        self.assertAlmostEqual(test_num, result, places=6)


    def test_addition_simple(self):
        result = self.fp.float_summa(1.5, 2.5)
        self.assertAlmostEqual(result, 4.0, places=6)

    def test_addition_with_negative(self):
        result = self.fp.float_summa(5.5, -2.5)
        self.assertAlmostEqual(result, 3.0, places=6)


    def test_max_value(self):
        max_float = 3.4028235e38
        binary = self.fp.convert_float_to_bin(max_float)
        result = self.fp.convert_bin_to_float(binary)
        self.assertTrue(math.isclose(result, max_float, rel_tol=1e-6))

    def test_min_positive(self):
        min_float = 1.17549435e-38
        binary = self.fp.convert_float_to_bin(min_float)
        result = self.fp.convert_bin_to_float(binary)
        self.assertAlmostEqual(result, min_float, places=6)


    def test_binary_structure(self):
        test_num = 10.5
        binary = self.fp.convert_float_to_bin(test_num)


        self.assertEqual(len(binary), 32)


        sign = binary[0]
        exponent = binary[1:9]
        mantissa = binary[9:]

        self.assertEqual(sign, '0')
        self.assertEqual(exponent, '10000010')
        self.assertEqual(mantissa, '01010000000000000000000')


if __name__ == '__main__':
    unittest.main()