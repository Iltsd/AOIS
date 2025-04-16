import unittest
import io
import sys
from unittest.mock import patch
from logic import LogicExpression


class TestLogicMinimization(unittest.TestCase):
    def normalize_expression(self, expr):
        """Нормализует логическое выражение для сравнения, сортируя термы."""
        if expr in ['0', '1']:
            return expr
        terms = expr.replace('(', '').replace(')', '').split(' | ')
        terms = [' & '.join(sorted(t.split(' & '))) for t in terms]
        return ' | '.join(sorted(terms))

    def normalize_cnf_expression(self, expr):
        """Нормализует СКНФ, сортируя дизъюнкции и их литералы."""
        if expr in ['0', '1']:
            return expr
        terms = expr.split(' & ')
        terms = [t.strip('()') for t in terms]
        terms = [' | '.join(sorted(t.split(' | '))) for t in terms]
        return ' & '.join([f'({t})' for t in sorted(terms)])

    def test_three_variables(self):
        """Тестирует выражение ((a&b)|(!c)) с 3 переменными."""
        expr = LogicExpression('((a&b)|(!c))')

        # Проверка таблицы истинности
        expected_table = [((0, 0, 0),
  [('a', 0),
   ('b', 0),
   ('(a&b)', 0),
   ('c', 0),
   ('(!c)', True),
   ('(c|(!c))', True)],
  True),
 ((0, 0, 1),
  [('a', 0),
   ('b', 0),
   ('(a&b)', 0),
   ('c', 1),
   ('(!c)', False),
   ('(c|(!c))', False)],
  False),
 ((0, 1, 0),
  [('a', 0),
   ('b', 1),
   ('(a&b)', 0),
   ('c', 0),
   ('(!c)', True),
   ('(c|(!c))', True)],
  True),
 ((0, 1, 1),
  [('a', 0),
   ('b', 1),
   ('(a&b)', 0),
   ('c', 1),
   ('(!c)', False),
   ('(c|(!c))', False)],
  False),
 ((1, 0, 0),
  [('a', 1),
   ('b', 0),
   ('(a&b)', 0),
   ('c', 0),
   ('(!c)', True),
   ('(c|(!c))', True)],
  True),
 ((1, 0, 1),
  [('a', 1),
   ('b', 0),
   ('(a&b)', 0),
   ('c', 1),
   ('(!c)', False),
   ('(c|(!c))', False)],
  False),
 ((1, 1, 0),
  [('a', 1), ('b', 1), ('(a&b)', 1), ('c', 0), ('(!c)', True), ('(c|(!c))', 1)],
  1),
 ((1, 1, 1),
  [('a', 1),
   ('b', 1),
   ('(a&b)', 1),
   ('c', 1),
   ('(!c)', False),
   ('(c|(!c))', 1)],
  1)]
        self.assertEqual(expr.table, expected_table)

        # Проверка СДНФ
        expected_dnf = '!a & !b & !c | !a & b & !c | a & !b & !c | a & b & !c | a & b & c'
        self.assertEqual(self.normalize_expression(expr.build_dnf()), self.normalize_expression(expected_dnf))

        # Проверка СКНФ
        expected_cnf = '(!a | !c | b) & (!b | !c | a) & (!c | a | b)'
        self.assertEqual(self.normalize_cnf_expression(expr.build_cnf()), self.normalize_cnf_expression(expected_cnf))

        # Проверка числовых форм
        expected_dnf_nums = [0, 2, 4, 6, 7]
        expected_cnf_nums = [1, 3, 5]
        self.assertEqual(expr.get_numeric_forms(), (expected_dnf_nums, expected_cnf_nums))

        # Проверка индексной формы
        self.assertEqual(expr.get_index_form(), 171)  # Бинарно: 11010101

        # Проверка минимизации СДНФ
        min_dnf, _ = expr.minimize_sdnf_calculation()
        expected_min_dnf = '!c | a & b'
        self.assertEqual(self.normalize_expression(min_dnf), self.normalize_expression(expected_min_dnf))

        # Проверка минимизации СКНФ
        min_cnf, _ = expr.minimize_sknf_calculation()
        expected_min_cnf = '(a | !c) & (b | !c)'
        self.assertEqual(self.normalize_cnf_expression(min_cnf), self.normalize_cnf_expression(expected_min_cnf))

        # Проверка карты Карно
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            expr.print_karnaugh_map()
            output = fake_out.getvalue()
            self.assertIn('Карта Карно для СДНФ (3 переменные):', output)
            self.assertIn(' a\\bc | 00 | 01 | 11 | 10', output)
            self.assertIn(' 0    |  1 |  0 |  0 |  1 |', output)
            self.assertIn(' 1    |  1 |  0 |  1 |  1 |', output)

    def test_four_variables(self):
        """Тестирует выражение ((a&b)|(c&d)) с 4 переменными."""
        expr = LogicExpression('((a&b)|(c&d))')

        # Проверка таблицы истинности (частично)
        self.assertEqual(len(expr.table), 16)
        self.assertEqual(expr.table[0][2], False)  # a=0, b=0, c=0, d=0
        self.assertEqual(expr.table[15][2], True)  # a=1, b=1, c=1, d=1

        # Проверка СДНФ
        expected_dnf = '!a & !b & c & d | !a & b & c & d | !b & a & c & d | !c & !d & a & b | !c & a & b & d | !d & a & b & c | a & b & c & d'
        self.assertEqual(self.normalize_expression(expr.build_dnf()), self.normalize_expression(expected_dnf))

        # Проверка числовых форм
        dnf_nums, _ = expr.get_numeric_forms()
        self.assertEqual(sorted(dnf_nums), [3, 7, 11, 12, 13, 14, 15])

        # Проверка минимизации СДНФ
        min_dnf, _ = expr.minimize_sdnf_calculation()
        expected_min_dnf = 'a & b | c & d'
        self.assertEqual(self.normalize_expression(min_dnf), self.normalize_expression(expected_min_dnf))

        # Проверка карты Карно
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            expr.print_karnaugh_map()
            output = fake_out.getvalue()
            self.assertIn('Карта Карно для СДНФ (4 переменные):', output)
            self.assertIn(' ab\\cd | 00 | 01 | 11 | 10', output)
            self.assertIn(' 11   |  1 |  1 |  1 |  1 |', output)

    def test_five_variables(self):
        """Тестирует выражение (((a&b)|(c<->d))&e) с 5 переменными."""
        expr = LogicExpression('(((a&b)|(c<->d))&e)')

        # Проверка таблицы истинности (частично)
        self.assertEqual(len(expr.table), 32)
        self.assertEqual(expr.table[0][2], False)  # a=0, b=0, c=0, d=0, e=0
        self.assertEqual(expr.table[31][2], True)  # a=1, b=1, c=1, d=1, e=1

        # Проверка СДНФ
        dnf = expr.build_dnf()
        self.assertIn('!a & !b & !c', dnf)
        self.assertIn('c & d & e', dnf)

        # Проверка числовых форм
        dnf_nums, _ = expr.get_numeric_forms()
        self.assertIn(31, dnf_nums)  # a=1, b=1, c=1, d=1, e=1

        # Проверка минимизации СДНФ
        min_dnf, _ = expr.minimize_sdnf_calculation()
        self.assertIn('a & b & e', self.normalize_expression(min_dnf))

        # Проверка карты Карно
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            expr.print_karnaugh_map()
            output = fake_out.getvalue()
            self.assertIn('Карта Карно для СДНФ (5 переменных):', output)
            self.assertIn(' ab\\cde | 000 | 001 | 011 | 010 | 110 | 111 | 101 | 100', output)
            self.assertIn('11    |  0  |  1  |  1  |  0  |  0  |  1  |  1  |  0  |', output)


    def test_invalid_parentheses(self):
        """Тестирует выражение с несбалансированными скобками."""
        with self.assertRaises(ValueError):
            LogicExpression('((a&b)|c')  # Несбалансированные скобки




if __name__ == '__main__':
    unittest.main()