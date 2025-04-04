import itertools
import re

class TruthTable:

    operators = {
        '!': (3, 'R'),
        '&': (2, 'L'),
        '|': (2, 'L'),
        '->': (1, 'R'),
        '<->': (1, 'L')
    }

    def __init__(self, infix):

        self.infix = infix.strip().replace(" ", "")
        self.variables = sorted(set(re.findall(r'[a-z]', self.infix)))
        self.postfix = None
        self.table = None
        self._validate_and_process()

    def _validate_and_process(self):

        if not self.check_strict_parentheses():
            raise ValueError("Ошибка: каждая операция должна быть строго заключена в скобки.")
        self.postfix = self.infix_to_postfix()
        self.table = self.generate_truth_table()

    def check_strict_parentheses(self):

        if self.infix.count('(') != self.infix.count(')'):
            return False
        unary_ops = self.infix.count('!')
        binary_ops = len(re.findall(r'\&|\||->|<->', self.infix))
        total_ops = unary_ops + binary_ops
        parens_pairs = self.infix.count('(')
        return parens_pairs >= total_ops

    def infix_to_postfix(self):

        infix = re.findall(r'[a-z]+|[\&\|\!\(\)]|->|<->', self.infix)
        stack = []
        postfix = []
        for token in infix:
            if token.isalpha() and token.islower():
                postfix.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()
            elif token in self.operators:
                while stack and stack[-1] in self.operators and (
                        (self.operators[stack[-1]][1] == 'L' and self.operators[stack[-1]][0] >= self.operators[token][0]) or
                        (self.operators[stack[-1]][1] == 'R' and self.operators[stack[-1]][0] > self.operators[token][0])
                ):
                    postfix.append(stack.pop())
                stack.append(token)
        while stack:
            postfix.append(stack.pop())
        return postfix

    def evaluate_postfix_with_intermediates(self, values):

        stack = []
        intermediates = []
        for token in self.postfix:
            if token.isalpha() and token.islower():
                stack.append(values[token])
                intermediates.append((token, values[token]))
            elif token == '!':
                a = stack.pop()
                result = not a
                stack.append(result)
                intermediates.append((f'(!{intermediates[-1][0]})', result))
            elif token in ['&', '|', '->', '<->']:
                b = stack.pop()
                a = stack.pop()
                if token == '&':
                    result = a and b
                elif token == '|':
                    result = a or b
                elif token == '->':
                    result = not a or b
                elif token == '<->':
                    result = a == b
                stack.append(result)
                op_a = intermediates[-2][0] if len(intermediates) >= 2 else '?'
                op_b = intermediates[-1][0] if len(intermediates) >= 1 else '?'
                intermediates.append((f'({op_a}{token}{op_b})', result))
        return stack[0], intermediates

    def generate_truth_table(self):

        n = len(self.variables)
        table = []
        for vals in itertools.product([0, 1], repeat=n):
            values = dict(zip(self.variables, vals))
            result, intermediates = self.evaluate_postfix_with_intermediates(values)
            table.append((vals, intermediates, result))
        return table

    def build_dnf(self):

        dnf = []
        for i, (vals, _, result) in enumerate(self.table):
            if result:
                term = ' & '.join([f'{"!" if not vals[j] else ""}{self.variables[j]}' for j in range(len(self.variables))])
                dnf.append(term)
        return ' | '.join(dnf) if dnf else '0'

    def build_cnf(self):

        cnf = []
        for i, (vals, _, result) in enumerate(self.table):
            if not result:
                term = ' | '.join([f'{"!" if vals[j] else ""}{self.variables[j]}' for j in range(len(self.variables))])
                cnf.append(f'({term})')
        return ' & '.join(cnf) if cnf else '1'

    def get_numeric_forms(self):

        dnf_nums = [i for i, (_, _, result) in enumerate(self.table) if result]
        cnf_nums = [i for i, (_, _, result) in enumerate(self.table) if not result]
        return dnf_nums, cnf_nums

    def get_index_form(self):

        outputs = [int(result) for _, _, result in self.table]
        return sum(outputs[i] * (1 << (len(self.table) - i - 1)) for i in range(len(self.table)))

    def print_truth_table(self):

        print("\nТаблица истинности:")
        header = ' '.join(self.variables) + ' | ' + ' | '.join([f'{i[0]}' for i in self.table[0][1]]) + ' | F'
        print(header)
        print('-' * len(header))
        for vals, intermediates, result in self.table:
            row = ' '.join(map(str, vals)) + ' | ' + ' | '.join([str((len(i[0])-1)*' ') + str(int(i[1])) for i in intermediates]) + ' | ' + str(int(result))
            print(row)

    def print_results(self):

        self.print_truth_table()
        print("\nСДНФ:", self.build_dnf())
        print("СКНФ:", self.build_cnf())
        dnf_nums, cnf_nums = self.get_numeric_forms()
        print("\nЧисловая форма СДНФ:", dnf_nums)
        print("Числовая форма СКНФ:", cnf_nums)
        print("Индексная форма:", self.get_index_form())

def main():
    infix = input("Введите логическую функцию (например, '(((a&b)|(a<->f))&m)'): ")
    try:
        logic_expr = TruthTable(infix)
        logic_expr.print_results()
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()