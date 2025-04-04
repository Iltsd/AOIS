import unittest
from fixed_point import Number
from floating_tests import TestFloatingPoint


class TestNumberOperations(unittest.TestCase):
    def setUp(self):
        self.num = Number()

    # Тесты для binary_number()
    def test_binary_number_positive(self):
        self.assertEqual(self.num.binary_number(5), '00000000000000000000000000000101')

    def test_binary_number_negative(self):
        self.assertEqual(self.num.binary_number(-5), '10000000000000000000000000000101')

    # Тесты для transfered_num_for_additional()
    def test_transfered_num_positive(self):
        self.assertEqual(self.num.transfered_num_for_additional('0101'), '0110')

    def test_transfered_num_negative(self):
        self.assertEqual(self.num.transfered_num_for_additional('1010'), '1011')

    # Тесты для convert_to_10()
    def test_convert_to_10_positive(self):
        self.assertEqual(self.num.convert_to_10('0101'), 5)

    def test_convert_to_10_negative(self):
        self.assertEqual(self.num.convert_to_10('1011'), -5)

    # Тесты для direct_sum()
    def test_direct_sum_positive(self):
        self.assertEqual(self.num.direct_sum(5, 3), '00000000000000000000000000001000')

    # Тесты для convert_to_reverse_binary()
    def test_reverse_code_positive(self):
        self.assertEqual(self.num.convert_to_reverse_binary(5), '00000000000000000000000000000101')

    def test_reverse_code_negative(self):
        self.assertEqual(self.num.convert_to_reverse_binary(-5), '11111111111111111111111111111010')

    # Тесты для additional_sum()
    def test_additional_sum_positive(self):
        self.assertEqual(self.num.additional_sum(5, 3), '00000000000000000000000000001000')

    def test_additional_sum_negative(self):
        self.assertEqual(self.num.additional_sum(-5, -3), '11111111111111111111111111111000')

    # Тесты для additional_subtract()
    def test_additional_subtract(self):
        self.assertEqual(self.num.additional_subtract(10, 3), '00000000000000000000000000000111')

    # Тесты для direct_code_multiplication()
    def test_multiplication_positive(self):
        result = self.num.direct_code_multiplication(5, 3)
        self.assertIn("Десятичный результат: 15", result)

    def test_multiplication_negative(self):
        result = self.num.direct_code_multiplication(-5, 3)
        self.assertIn("Десятичный результат: -15", result)

    # Тесты для divide_bin()
    def test_division_normal(self):
        result = self.num.divide_bin(10, 3)
        self.assertIn("Десятичный результат: 3.33325", result)

    def test_division_by_zero(self):
        with self.assertRaises(ValueError):
            self.num.divide_bin(10, 0)

    def test_division_negative(self):
        result = self.num.divide_bin(-10, 3)
        self.assertIn("Десятичный результат: -3.33325", result)


if __name__ == '__main__':
    unittest.main()