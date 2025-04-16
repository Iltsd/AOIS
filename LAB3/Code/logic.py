import itertools
import re


class LogicExpression:
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
                        (self.operators[stack[-1]][1] == 'L' and self.operators[stack[-1]][0] >= self.operators[token][
                            0]) or
                        (self.operators[stack[-1]][1] == 'R' and self.operators[stack[-1]][0] > self.operators[token][
                            0])
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
                term = ' & '.join(
                    [f'{"!" if not vals[j] else ""}{self.variables[j]}' for j in range(len(self.variables))])
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
            row = ' '.join(map(str, vals)) + ' | ' + ' | '.join(
                [str((len(i[0]) - 1) * ' ') + str(int(i[1])) for i in intermediates]) + ' | ' + str(int(result))
            print(row)

    def can_merge(self, t1, t2):
        differ_count = 0
        for j in range(len(t1)):
            if t1[j] == t2[j]:
                continue
            elif (t1[j], t2[j]) in [('0', '1'), ('1', '0')]:
                differ_count += 1
            else:
                return False
        return differ_count == 1

    def merge(self, t1, t2):
        merged = []
        for j in range(len(t1)):
            if t1[j] == t2[j]:
                merged.append(t1[j])
            else:
                merged.append('-')
        return ''.join(merged)

    def term_to_expression(self, term, is_dnf=True):
        if term == '0' or term == '1':
            return term
        result = []
        for i, bit in enumerate(term):
            if bit == '0':
                literal = f'!{self.variables[i]}' if is_dnf else self.variables[i]
                result.append(literal)
            elif bit == '1':
                literal = self.variables[i] if is_dnf else f'!{self.variables[i]}'
                result.append(literal)
        join_op = ' & ' if is_dnf else ' | '
        return join_op.join(result) if result else ('1' if is_dnf else '0')

    def minimize_sdnf_calculation(self):
        n = len(self.variables)
        minterms_indices = [i for i, (_, _, result) in enumerate(self.table) if result]
        if not minterms_indices:
            return "0", []

        terms = [(bin(i)[2:].zfill(n), frozenset([i]), False) for i in minterms_indices]
        gluing_steps = []
        prime_implicants = []

        while True:
            new_terms = []
            used = set()
            merged_found = False

            for i, (t1_str, t1_mins, _) in enumerate(terms):
                for j, (t2_str, t2_mins, _) in enumerate(terms[i + 1:], i + 1):
                    if self.can_merge(t1_str, t2_str):
                        merged_str = self.merge(t1_str, t2_str)
                        merged_mins = t1_mins | t2_mins
                        new_terms.append((merged_str, merged_mins, False))
                        used.add(i)
                        used.add(j)
                        merged_found = True
                        gluing_steps.append(
                            f"Склеивание: {self.term_to_expression(t1_str)} + "
                            f"{self.term_to_expression(t2_str)} = {self.term_to_expression(merged_str)}"
                        )

            for i, (t_str, t_mins, _) in enumerate(terms):
                if i not in used:
                    prime_implicants.append((t_str, t_mins))

            if not merged_found:
                break

            terms = new_terms

        minterms_set = set(minterms_indices)
        essential = []
        covered_minterms = set()

        for pi in prime_implicants:
            unique_minterms = pi[1] - covered_minterms
            if unique_minterms and any(m in minterms_set for m in unique_minterms):
                essential.append(pi)
                covered_minterms |= pi[1]

        remaining_minterms = minterms_set - covered_minterms
        if remaining_minterms:
            for pi in sorted(prime_implicants, key=lambda x: len(x[1]), reverse=True):
                if pi not in essential and pi[1] & remaining_minterms:
                    essential.append(pi)
                    covered_minterms |= pi[1]
                    remaining_minterms -= pi[1]
                    if not remaining_minterms:
                        break

        minimized = ' | '.join(self.term_to_expression(pi[0]) for pi in essential) if essential else '0'
        return minimized, gluing_steps

    def minimize_sknf_calculation(self):
        n = len(self.variables)
        maxterms_indices = [i for i, (_, _, result) in enumerate(self.table) if not result]
        if not maxterms_indices:
            return "1", []

        terms = [(bin(i)[2:].zfill(n), frozenset([i]), False) for i in maxterms_indices]
        gluing_steps = []
        prime_implicants = []

        while True:
            new_terms = []
            used = set()
            merged_found = False

            for i, (t1_str, t1_maxs, _) in enumerate(terms):
                for j, (t2_str, t2_maxs, _) in enumerate(terms[i + 1:], i + 1):
                    if self.can_merge(t1_str, t2_str):
                        merged_str = self.merge(t1_str, t2_str)
                        merged_maxs = t1_maxs | t2_maxs
                        new_terms.append((merged_str, merged_maxs, False))
                        used.add(i)
                        used.add(j)
                        merged_found = True
                        gluing_steps.append(
                            f"Склеивание: ({self.term_to_expression(t1_str, is_dnf=False)}) * "
                            f"({self.term_to_expression(t2_str, is_dnf=False)}) = ({self.term_to_expression(merged_str, is_dnf=False)})"
                        )

            for i, (t_str, t_maxs, _) in enumerate(terms):
                if i not in used:
                    prime_implicants.append((t_str, t_maxs))

            if not merged_found:
                break

            terms = new_terms

        maxterms_set = set(maxterms_indices)
        essential = []
        covered_maxterms = set()

        for pi in prime_implicants:
            unique_maxterms = pi[1] - covered_maxterms
            if unique_maxterms and any(m in maxterms_set for m in unique_maxterms):
                essential.append(pi)
                covered_maxterms |= pi[1]

        remaining_maxterms = maxterms_set - covered_maxterms
        if remaining_maxterms:
            for pi in sorted(prime_implicants, key=lambda x: len(x[1]), reverse=True):
                if pi not in essential and pi[1] & remaining_maxterms:
                    essential.append(pi)
                    covered_maxterms |= pi[1]
                    remaining_maxterms -= pi[1]
                    if not remaining_maxterms:
                        break

        minimized = ' & '.join(
            f'({self.term_to_expression(pi[0], is_dnf=False)})' for pi in essential) if essential else '1'
        return minimized, gluing_steps

    def print_karnaugh_map(self):
        n = len(self.variables)
        if n > 5:
            print("Карта Карно для более чем 5 переменных не поддерживается в этой реализации.")
            return

        outputs = [int(result) for _, _, result in self.table]
        gray_code_2 = ['00', '01', '11', '10']
        gray_code_3 = ['000', '001', '011', '010', '110', '111', '101', '100']

        if n == 3:
            print("\nКарта Карно для СДНФ (3 переменные):")
            print(" a\\bc | 00 | 01 | 11 | 10")
            print("-------+----+----+----+----")
            for a in [0, 1]:
                row = f" {a}    |"
                for bc in gray_code_2:
                    b, c = int(bc[0]), int(bc[1])
                    idx = a * 4 + b * 2 + c
                    row += f"  {outputs[idx]} |"
                print(row)
            # Упрощённая минимизация
            minterms = [(bin(i)[2:].zfill(n), i) for i, r in enumerate(outputs) if r]
            groups = []
            for size in [4, 2, 1]:
                for i, (t1, m1) in enumerate(minterms):
                    for t2, m2 in minterms[i + 1:]:
                        if size == 1 and t1 == t2:
                            groups.append(t1)
                        elif self.can_merge(t1, t2):
                            merged = self.merge(t1, t2)
                            if merged.count('-') == n - int(size).bit_length() + 1:
                                groups.append(merged)
            minimized = ' | '.join(self.term_to_expression(g) for g in set(groups)) if groups else '0'
            print("\nМинимизированная СДНФ (Карта Карно):", minimized)

        elif n == 4:
            print("\nКарта Карно для СДНФ (4 переменные):")
            print(" ab\\cd | 00 | 01 | 11 | 10")
            print("--------+----+----+----+----")
            for ab in gray_code_2:
                a, b = int(ab[0]), int(ab[1])
                row = f" {ab}   |"
                for cd in gray_code_2:
                    c, d = int(cd[0]), int(cd[1])
                    idx = a * 8 + b * 4 + c * 2 + d
                    row += f"  {outputs[idx]} |"
                print(row)
            # Упрощённая минимизация
            minterms = [(bin(i)[2:].zfill(n), i) for i, r in enumerate(outputs) if r]
            groups = []
            for size in [8, 4, 2, 1]:
                for i, (t1, m1) in enumerate(minterms):
                    for t2, m2 in minterms[i + 1:]:
                        if size == 1 and t1 == t2:
                            groups.append(t1)
                        elif self.can_merge(t1, t2):
                            merged = self.merge(t1, t2)
                            if merged.count('-') == n - int(size).bit_length() + 1:
                                groups.append(merged)
            minimized = ' | '.join(self.term_to_expression(g) for g in set(groups)) if groups else '0'
            print("\nМинимизированная СДНФ (Карта Карно):", minimized)

        elif n == 5:
            print("\nКарта Карно для СДНФ (5 переменных):")
            print(" ab\\cde | 000 | 001 | 011 | 010 | 110 | 111 | 101 | 100")
            print("---------+-----+-----+-----+-----+-----+-----+-----+-----")
            for ab in gray_code_2:
                a, b = int(ab[0]), int(ab[1])
                row = f" {ab}    |"
                for cde in gray_code_3:
                    c, d, e = int(cde[0]), int(cde[1]), int(cde[2])
                    idx = a * 16 + b * 8 + c * 4 + d * 2 + e
                    row += f"  {outputs[idx]}  |"
                print(row)
            # Упрощённая минимизация
            minterms = [(bin(i)[2:].zfill(n), i) for i, r in enumerate(outputs) if r]
            groups = []
            for size in [16, 8, 4, 2, 1]:
                for i, (t1, m1) in enumerate(minterms):
                    for t2, m2 in minterms[i + 1:]:
                        if size == 1 and t1 == t2:
                            groups.append(t1)
                        elif self.can_merge(t1, t2):
                            merged = self.merge(t1, t2)
                            if merged.count('-') == n - int(size).bit_length() + 1:
                                groups.append(merged)
            minimized = ' | '.join(self.term_to_expression(g) for g in set(groups)) if groups else '0'
            print("\nМинимизированная СДНФ (Карта Карно):", minimized)

    def print_results(self):
        self.print_truth_table()
        print("\nСДНФ:", self.build_dnf())
        print("СКНФ:", self.build_cnf())
        dnf_nums, cnf_nums = self.get_numeric_forms()
        print("\nЧисловая форма СДНФ:", dnf_nums)
        print("Числовая форма СКНФ:", cnf_nums)
        print("Индексная форма:", self.get_index_form())
        print("\nМинимизация СДНФ (расчетный метод):")
        min_sdnf, sdnf_gluing = self.minimize_sdnf_calculation()
        for step in sdnf_gluing:
            print(step)
        print("Результат:", min_sdnf)
        print("\nМинимизация СКНФ (расчетный метод):")
        min_sknf, sknf_gluing = self.minimize_sknf_calculation()
        for step in sknf_gluing:
            print(step)
        print("Результат:", min_sknf)
        self.print_karnaugh_map()


def main():
    infix = input("Введите логическую функцию (например, '(((a&b)|(c<->d))&e)'): ")
    try:
        logic_expr = LogicExpression(infix)
        logic_expr.print_results()
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()