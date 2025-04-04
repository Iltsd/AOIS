import unittest
from truth_table import TruthTable

class TestLogicExpression(unittest.TestCase):

    def test_init_valid_expression(self):

        expr = TruthTable('(((a&b)|(a<->f))&m)')
        self.assertEqual(expr.infix, '(((a&b)|(a<->f))&m)')
        self.assertEqual(expr.variables, ['a', 'b', 'f', 'm'])
        self.assertIsNotNone(expr.postfix)
        self.assertIsNotNone(expr.table)

    def test_init_invalid_parentheses(self):

        with self.assertRaises(ValueError):
            TruthTable('(a&b')
        with self.assertRaises(ValueError):
            TruthTable('((a&b|c)')

    def test_check_strict_parentheses(self):

        expr1 = TruthTable('(((a&b)|(a<->f))&m)')
        self.assertTrue(expr1.check_strict_parentheses())

        expr2 = TruthTable('(!a)')
        self.assertTrue(expr2.check_strict_parentheses())


        temp = TruthTable('(((a&b)|(a<->f))&m)')
        temp.infix = '(a&b)'
        self.assertTrue(temp.check_strict_parentheses())

    def test_infix_to_postfix(self):

        expr = TruthTable('(((a&b)|(a<->f))&m)')
        self.assertEqual(expr.infix_to_postfix(), ['a', 'b', '&', 'a', 'f', '<->', '|', 'm', '&'])

        expr2 = TruthTable('(!a)')
        self.assertEqual(expr2.infix_to_postfix(), ['a', '!'])

    def test_evaluate_postfix_with_intermediates(self):

        expr = TruthTable('(((a&b)|(a<->f))&m)')
        values = {'a': True, 'b': False, 'f': True, 'm': True}
        result, intermediates = expr.evaluate_postfix_with_intermediates(values)
        self.assertEqual(result, True)
        self.assertEqual(intermediates[-1][1], True)
        self.assertEqual(len(intermediates), 9)

    def test_generate_truth_table(self):

        expr = TruthTable('((a&b))')
        table = expr.generate_truth_table()
        self.assertEqual(len(table), 4)
        self.assertEqual(table[0][2], False)
        self.assertEqual(table[3][2], True)

    def test_build_dnf(self):

        expr = TruthTable('((a&b))')
        dnf = expr.build_dnf()
        self.assertEqual(dnf, 'a & b')

        expr2 = TruthTable('((a|b))')
        dnf2 = expr2.build_dnf()
        self.assertEqual(dnf2, '!a & b | a & !b | a & b')

    def test_build_cnf(self):

        expr = TruthTable('((a&b))')
        cnf = expr.build_cnf()
        self.assertEqual(cnf, '(a | b) & (a | !b) & (!a | b)')

        expr2 = TruthTable('((a|b))')
        cnf2 = expr2.build_cnf()
        self.assertEqual(cnf2, '(a | b)')

    def test_get_numeric_forms(self):

        expr = TruthTable('((a&b))')
        dnf_nums, cnf_nums = expr.get_numeric_forms()
        self.assertEqual(dnf_nums, [3])
        self.assertEqual(cnf_nums, [0, 1, 2])

    def test_get_index_form(self):

        expr = TruthTable('((a&b))')
        index = expr.get_index_form()
        self.assertEqual(index, 1)

        expr2 = TruthTable('((a|b))')
        index2 = expr2.get_index_form()
        self.assertEqual(index2, 7)  

if __name__ == "__main__":
    unittest.main()