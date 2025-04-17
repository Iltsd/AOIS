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
        self.assertEqual(expr.get_index_form(), 171)  # Бинарно: 10101011

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
            self.assertIn('Карта Карно (3 переменные):', output)
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
            self.assertIn('Карта Карно (4 переменные):', output)
            self.assertIn(' ab\\cd | 00 | 01 | 11 | 10', output)
            self.assertIn(' 11   |  1 |  1 |  1 |  1 |', output)
            self.assertIn('11   |  1 |  1 |  1 |  1 |', output)

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
        self.assertEqual(sorted(dnf_nums), [1, 7, 9, 15, 17, 23, 25, 27, 29, 31])

        # Проверка минимизации СДНФ (расчётный метод)
        min_dnf, _ = expr.minimize_sdnf_calculation()
        expected_min_dnf = '!c & !d & e | c & d & e | a & b & e'
        self.assertEqual(self.normalize_expression(min_dnf), self.normalize_expression(expected_min_dnf))

        # Проверка минимизации СКНФ (расчётный метод)
        min_cnf, _ = expr.minimize_sknf_calculation()
        expected_min_cnf = '(!c | a | d) & (!c | b | d) & (!d | a | c) & (!d | b | c) & (e)'
        self.assertEqual(self.normalize_cnf_expression(min_cnf), self.normalize_cnf_expression(expected_min_cnf))

        # Проверка карты Карно
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            expr.print_karnaugh_map()
            output = fake_out.getvalue()
            self.assertIn('Карта Карно (5 переменных):', output)
            self.assertIn(' ab\\cde | 000 | 001 | 011 | 010 | 110 | 111 | 101 | 100', output)
            self.assertIn(' 00    |  0  |  1  |  0  |  0  |  0  |  1  |  0  |  0  |', output)
            self.assertIn(' 01    |  0  |  1  |  0  |  0  |  0  |  1  |  0  |  0  |', output)
            self.assertIn(' 11    |  0  |  1  |  1  |  0  |  0  |  1  |  1  |  0  |', output)
            self.assertIn(' 10    |  0  |  1  |  0  |  0  |  0  |  1  |  0  |  0  |', output)

    def test_invalid_parentheses(self):
        """Тестирует выражение с несбалансированными скобками."""
        with self.assertRaises(ValueError):
            LogicExpression('((a&b)|c')  # Несбалансированные скобки

    def test_equivalence_operation(self):
        """Тестирует выражение с эквивалентностью (a<->b) с 2 переменными."""
        expr = LogicExpression('(a<->b)')

        # Проверка таблицы истинности
        expected_table = [
            ((0, 0), [('a', 0), ('b', 0), ('(a<->b)', True)], True),
            ((0, 1), [('a', 0), ('b', 1), ('(a<->b)', False)], False),
            ((1, 0), [('a', 1), ('b', 0), ('(a<->b)', False)], False),
            ((1, 1), [('a', 1), ('b', 1), ('(a<->b)', True)], True)
        ]
        self.assertEqual(expr.table, expected_table)

        # Проверка СДНФ
        expected_dnf = '!a & !b | a & b'
        self.assertEqual(self.normalize_expression(expr.build_dnf()), self.normalize_expression(expected_dnf))

        # Проверка СКНФ
        expected_cnf = '(!a | b) & (a | !b)'
        self.assertEqual(self.normalize_cnf_expression(expr.build_cnf()), self.normalize_cnf_expression(expected_cnf))

        # Проверка числовых форм
        dnf_nums, cnf_nums = expr.get_numeric_forms()
        self.assertEqual(dnf_nums, [0, 3])
        self.assertEqual(cnf_nums, [1, 2])

        # Проверка минимизации СДНФ
        min_dnf, _ = expr.minimize_sdnf_calculation()
        self.assertEqual(self.normalize_expression(min_dnf), self.normalize_expression(expected_dnf))

    def test_implication_operation(self):
        """Тестирует выражение с импликацией (a->b) с 2 переменными."""
        expr = LogicExpression('(a->b)')

        # Проверка таблицы истинности
        expected_table = [
            ((0, 0), [('a', 0), ('b', 0), ('(a->b)', True)], True),
            ((0, 1), [('a', 0), ('b', 1), ('(a->b)', True)], True),
            ((1, 0), [('a', 1), ('b', 0), ('(a->b)', False)], False),
            ((1, 1), [('a', 1), ('b', 1), ('(a->b)', True)], True)
        ]
        self.assertEqual(expr.table, expected_table)

        # Проверка СДНФ
        expected_dnf = '!a & !b | !a & b | a & b'
        self.assertEqual(self.normalize_expression(expr.build_dnf()), self.normalize_expression(expected_dnf))

        # Проверка СКНФ
        expected_cnf = '(!a | b)'
        self.assertEqual(self.normalize_cnf_expression(expr.build_cnf()), self.normalize_cnf_expression(expected_cnf))

        # Проверка числовых форм
        dnf_nums, cnf_nums = expr.get_numeric_forms()
        self.assertEqual(dnf_nums, [0, 1, 3])
        self.assertEqual(cnf_nums, [2])

        # Проверка минимизации СДНФ
        min_dnf, _ = expr.minimize_sdnf_calculation()
        expected_min_dnf = '!a | b'
        self.assertEqual(self.normalize_expression(min_dnf), self.normalize_expression(expected_min_dnf))

    def test_minimize_sknf_table_four_variables(self):
        """Тестирует минимизацию СКНФ методом таблицы для ((a&b)|(c&d)) с 4 переменными."""
        expr = LogicExpression('((a&b)|(c&d))')

        # Проверка числовых форм для подтверждения макстермов
        _, cnf_nums = expr.get_numeric_forms()
        expected_cnf_nums = [0, 1, 2, 4, 5, 6, 8, 9, 10]
        self.assertEqual(sorted(cnf_nums), sorted(expected_cnf_nums))

        # Проверка минимизации СКНФ методом таблицы
        min_cnf, _ = expr.minimize_sknf_table()
        expected_min_cnf = '(a | c) & (a | d) & (b | c) & (b | d)'
        self.assertEqual(self.normalize_cnf_expression(min_cnf), self.normalize_cnf_expression(expected_min_cnf))

    def test_minimize_sdnf_table_three_variables(self):
        """Тестирует минимизацию СДНФ методом таблицы для ((a&b)|(!c)) с 3 переменными."""
        expr = LogicExpression('((a&b)|(!c))')

        # Проверка числовых форм для подтверждения минтермов
        dnf_nums, _ = expr.get_numeric_forms()
        expected_dnf_nums = [0, 2, 4, 6, 7]
        self.assertEqual(dnf_nums, expected_dnf_nums)

        # Проверка минимизации СДНФ методом таблицы
        min_dnf, _ = expr.minimize_sdnf_table()
        expected_min_dnf = '!c | a & b'
        self.assertEqual(self.normalize_expression(min_dnf), self.normalize_expression(expected_min_dnf))

    def test_xor_operation(self):
        """Тестирует выражение с XOR (a^b) с 2 переменными."""
        expr = LogicExpression('(a^b)')

        # Проверка таблицы истинности
        expected_table = [((0, 0), [('a', 0), ('b', 0)], 0),
 ((0, 1), [('a', 0), ('b', 1)], 0),
 ((1, 0), [('a', 1), ('b', 0)], 1),
 ((1, 1), [('a', 1), ('b', 1)], 1)]
        self.assertEqual(expr.table, expected_table)

        # Проверка СДНФ
        expected_dnf = '!b & a | a & b'
        self.assertEqual(self.normalize_expression(expr.build_dnf()), self.normalize_expression(expected_dnf))

        # Проверка СКНФ
        expected_cnf = '(!b | a) & (a | b)'
        self.assertEqual(self.normalize_cnf_expression(expr.build_cnf()), self.normalize_cnf_expression(expected_cnf))

        # Проверка числовых форм
        dnf_nums, cnf_nums = expr.get_numeric_forms()
        self.assertEqual(dnf_nums, [2, 3])
        self.assertEqual(cnf_nums, [0, 1])

        # Проверка минимизации СДНФ
        min_dnf, _ = expr.minimize_sdnf_calculation()
        self.assertEqual(self.normalize_expression(min_dnf), 'a')





    def test_single_variable(self):
        """Тестирует выражение с одной переменной (a)."""
        expr = LogicExpression('a')

        # Проверка таблицы истинности
        expected_table = [
            ((0,), [('a', 0)], False),
            ((1,), [('a', 1)], True)
        ]
        self.assertEqual(expr.table, expected_table)

        # Проверка СДНФ
        expected_dnf = 'a'
        self.assertEqual(expr.build_dnf(), expected_dnf)

        # Проверка СКНФ
        expected_cnf = '(a)'
        self.assertEqual(expr.build_cnf(), expected_cnf)

        # Проверка числовых форм
        dnf_nums, cnf_nums = expr.get_numeric_forms()
        self.assertEqual(dnf_nums, [1])
        self.assertEqual(cnf_nums, [0])

        # Проверка минимизации
        min_dnf, _ = expr.minimize_sdnf_calculation()
        self.assertEqual(min_dnf, '1')
        min_cnf, _ = expr.minimize_sknf_calculation()
        self.assertEqual(min_cnf, '(0)')

    def test_complex_five_variables(self):
        """Тестирует сложное выражение ((a->b)&(c^d)&e) с 5 переменными."""
        expr = LogicExpression('((a->b)&(c^d)&e)')

        # Проверка таблицы истинности (частично)
        self.assertEqual(len(expr.table), 32)
        self.assertEqual(expr.table[0][2], True)  # a=0, b=0, c=0, d=0, e=0
        self.assertEqual(expr.table[27][2], True)  # a=1, b=1, c=0, d=1, e=1

        # Проверка числовых форм
        dnf_nums, _ = expr.get_numeric_forms()
        expected_dnf_nums = [0,
 1,
 2,
 3,
 4,
 5,
 6,
 7,
 8,
 9,
 10,
 11,
 12,
 13,
 14,
 15,
 24,
 25,
 26,
 27,
 28,
 29,
 30,
 31]  # Минтермы, где (a->b)=1, (c^d)=1, e=1
        self.assertEqual(sorted(dnf_nums), sorted(expected_dnf_nums))

        # Проверка минимизации СДНФ
        min_dnf, _ = expr.minimize_sdnf_calculation()
        expected_min_dnf = '!a | b'
        self.assertEqual(self.normalize_expression(min_dnf), self.normalize_expression(expected_min_dnf))



if __name__ == '__main__':
    unittest.main()