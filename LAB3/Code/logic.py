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

    def _initialize_terms(self, indices, n):
        """Инициализирует термы для минимизации."""
        return [(bin(i)[2:].zfill(n), frozenset([i]), False) for i in indices]

    def _perform_gluing_step(self, terms, is_dnf=True):
        """Выполняет один этап склеивания термов."""
        new_terms = []
        used = set()
        merged_found = False
        gluing_steps = []
        for i, (t1_str, t1_terms, _) in enumerate(terms):
            for j, (t2_str, t2_terms, _) in enumerate(terms[i + 1:], i + 1):
                if self.can_merge(t1_str, t2_str):
                    merged_str = self.merge(t1_str, t2_str)
                    merged_terms = t1_terms | t2_terms
                    new_terms.append((merged_str, merged_terms, False))
                    used.add(i)
                    used.add(j)
                    merged_found = True
                    gluing_steps.append((t1_str, t1_terms, t2_str, t2_terms, merged_str, merged_terms))
        return new_terms, used, merged_found, gluing_steps

    def _log_gluing_step(self, t1_str, t1_terms, t2_str, t2_terms, merged_str, merged_terms, is_dnf=True):
        """Форматирует и выводит лог склеивания для одной пары термов."""
        term1 = self.term_to_expression(t1_str, is_dnf)
        term2 = self.term_to_expression(t2_str, is_dnf)
        merged = self.term_to_expression(merged_str, is_dnf)
        log = (
            f"Склеивание: {term1} ({'минтермы' if is_dnf else 'макстермы'} {sorted(t1_terms)}) + "
            f"{term2} ({'минтермы' if is_dnf else 'макстермы'} {sorted(t2_terms)}) = "
            f"{merged} ({'минтермы' if is_dnf else 'макстермы'} {sorted(merged_terms)})"
            if is_dnf else
            f"Склеивание: ({term1}) ({'макстермы'} {sorted(t1_terms)}) * "
            f"({term2}) ({'макстермы'} {sorted(t2_terms)}) = "
            f"({merged}) ({'макстермы'} {sorted(merged_terms)})"
        )
        print(log)
        return log

    def _collect_prime_implicants(self, terms, used):
        """Собирает главные импликанты из неиспользованных термов."""
        return [(t_str, t_terms) for i, (t_str, t_terms, _) in enumerate(terms) if i not in used]

    def _glue_terms(self, terms, is_dnf=True):
        """Выполняет этапы склеивания термов."""
        gluing_steps = []
        prime_implicants = []
        stage = 0
        print(f"\nЭтапы склеивания {'СДНФ' if is_dnf else 'СКНФ'} (расчетный метод):")
        while True:
            stage += 1
            print(f"\nЭтап {stage}:")
            new_terms, used, merged_found, step_gluings = self._perform_gluing_step(terms, is_dnf)
            for glue_data in step_gluings:
                log = self._log_gluing_step(*glue_data, is_dnf)
                gluing_steps.append(log)
            prime_implicants.extend(self._collect_prime_implicants(terms, used))
            if not merged_found:
                break
            terms = new_terms
        return prime_implicants, gluing_steps

    def _print_prime_implicants(self, prime_implicants, is_dnf=True):
        """Выводит главные импликанты."""
        print("\nГлавные импликанты:")
        seen = []
        for pi_str, pi_terms in prime_implicants:
            if pi_str in seen:
                continue
            seen.append(pi_str)
            expr = self.term_to_expression(pi_str, is_dnf)
            print(f"{expr if is_dnf else f'({expr})'} ({'минтермы' if is_dnf else 'макстермы'} {sorted(pi_terms)})")

    def _select_essential_implicants(self, prime_implicants, terms_set):
        """Выбирает эссенциальные импликанты для минимального покрытия."""
        essential = []
        covered_terms = set()
        for pi in prime_implicants:
            unique_terms = pi[1] - covered_terms
            if unique_terms and any(m in terms_set for m in unique_terms):
                essential.append(pi)
                covered_terms |= pi[1]
        remaining_terms = terms_set - covered_terms
        if remaining_terms:
            for pi in sorted(prime_implicants, key=lambda x: len(x[1]), reverse=True):
                if pi not in essential and pi[1] & remaining_terms:
                    essential.append(pi)
                    covered_terms |= pi[1]
                    remaining_terms -= pi[1]
                    if not remaining_terms:
                        break
        return essential

    def _format_minimized_expression(self, essential, is_dnf=True):
        """Форматирует минимальное выражение (СДНФ или СКНФ)."""
        if not essential:
            return "0" if is_dnf else "1"
        exprs = [self.term_to_expression(pi[0], is_dnf) for pi in essential]
        join_op = ' | ' if is_dnf else ' & '
        if is_dnf:
            return join_op.join(exprs)
        return join_op.join(f'({e})' for e in exprs)

    def minimize_sdnf_calculation(self):
        n = len(self.variables)
        minterms_indices = [i for i, (_, _, result) in enumerate(self.table) if result]
        if not minterms_indices:
            return "0", []
        terms = self._initialize_terms(minterms_indices, n)
        prime_implicants, gluing_steps = self._glue_terms(terms, is_dnf=True)
        self._print_prime_implicants(prime_implicants, is_dnf=True)
        essential = self._select_essential_implicants(prime_implicants, set(minterms_indices))
        minimized = self._format_minimized_expression(essential, is_dnf=True)
        return minimized, gluing_steps

    def minimize_sknf_calculation(self):
        n = len(self.variables)
        maxterms_indices = [i for i, (_, _, result) in enumerate(self.table) if not result]
        if not maxterms_indices:
            return "1", []
        terms = self._initialize_terms(maxterms_indices, n)
        prime_implicants, gluing_steps = self._glue_terms(terms, is_dnf=False)
        self._print_prime_implicants(prime_implicants, is_dnf=False)
        essential = self._select_essential_implicants(prime_implicants, set(maxterms_indices))
        minimized = self._format_minimized_expression(essential, is_dnf=False)
        return minimized, gluing_steps

    def _print_coverage_table(self, prime_implicants, terms_set, is_dnf=True):
        """Выводит таблицу покрытия."""
        print("\nТаблица покрытия:")
        header = "Импликант | " + " | ".join([str(m) for m in sorted(terms_set)])
        print(header)
        print("-" * len(header))
        coverage = []
        seen = []
        for pi_str, pi_terms in prime_implicants:
            if pi_str in seen:
                continue
            seen.append(pi_str)
            expr = self.term_to_expression(pi_str, is_dnf)
            row = f"{expr if is_dnf else f'({expr})'}".ljust(15 if is_dnf else 20) + "|"
            for m in sorted(terms_set):
                row += " X |" if m in pi_terms else "   |"
            print(row)
            coverage.append((pi_str, pi_terms))
        return coverage

    def minimize_sdnf_table(self):
        n = len(self.variables)
        minterms_indices = [i for i, (_, _, result) in enumerate(self.table) if result]
        if not minterms_indices:
            return "0", []
        terms = self._initialize_terms(minterms_indices, n)
        prime_implicants, gluing_steps = self._glue_terms(terms, is_dnf=True)
        self._print_prime_implicants(prime_implicants, is_dnf=True)
        coverage = self._print_coverage_table(prime_implicants, set(minterms_indices), is_dnf=True)
        essential = self._select_essential_implicants(coverage, set(minterms_indices))
        minimized = self._format_minimized_expression(essential, is_dnf=True)
        return minimized, gluing_steps

    def minimize_sknf_table(self):
        n = len(self.variables)
        maxterms_indices = [i for i, (_, _, result) in enumerate(self.table) if not result]
        if not maxterms_indices:
            return "1", []
        terms = self._initialize_terms(maxterms_indices, n)
        prime_implicants, gluing_steps = self._glue_terms(terms, is_dnf=False)
        self._print_prime_implicants(prime_implicants, is_dnf=False)
        coverage = self._print_coverage_table(prime_implicants, set(maxterms_indices), is_dnf=False)
        essential = self._select_essential_implicants(coverage, set(maxterms_indices))
        minimized = self._format_minimized_expression(essential, is_dnf=False)
        return minimized, gluing_steps

    def _get_karnaugh_indices(self, n, row, col):
        """Возвращает индексы в таблице истинности для ячейки карты Карно."""
        gray_code_2 = ['00', '01', '11', '10']
        gray_code_3 = ['000', '001', '011', '010', '110', '111', '101', '100']
        if n == 3:
            a = row
            bc = gray_code_2[col]
            b, c = int(bc[0]), int(bc[1])
            return [a * 4 + b * 2 + c]
        elif n == 4:
            ab = gray_code_2[row]
            cd = gray_code_2[col]
            a, b = int(ab[0]), int(ab[1])
            c, d = int(cd[0]), int(cd[1])
            return [a * 8 + b * 4 + c * 2 + d]
        elif n == 5:
            ab = gray_code_2[row]
            cde = gray_code_3[col]
            a, b = int(ab[0]), int(ab[1])
            c, d, e = int(cde[0]), int(cde[1]), int(cde[2])
            return [a * 16 + b * 8 + c * 4 + d * 2 + e]
        return []

    def _group_to_term(self, group, n):
        """Преобразует группу ячеек в терм (бинарную строку)."""
        bits = ['-' for _ in range(n)]
        fixed_positions = {}
        for idx in group:
            bin_str = bin(idx)[2:].zfill(n)
            for i, bit in enumerate(bin_str):
                if i not in fixed_positions:
                    fixed_positions[i] = bit
                elif fixed_positions[i] != bit:
                    fixed_positions[i] = '-'
        for i, bit in fixed_positions.items():
            bits[i] = bit
        return ''.join(bits)

    def _generate_group_indices(self, n, r_start, c_start, height, width, rows, cols):
        """Генерирует индексы ячеек для группы в карте Карно."""
        group_indices = []
        for dr in range(height):
            for dc in range(width):
                r = (r_start + dr) % rows
                c = (c_start + dc) % cols
                idxs = self._get_karnaugh_indices(n, r, c)
                group_indices.extend(idxs)
        return group_indices

    def _validate_group(self, group_indices, outputs, target_value):
        """Проверяет, что все ячейки в группе имеют целевое значение."""
        return all(outputs[idx] == target_value for idx in group_indices)

    def _add_single_cells(self, outputs, target_value, covered_indices):
        """Добавляет одиночные ячейки, не покрытые группами."""
        single_groups = []
        for i, val in enumerate(outputs):
            if val == target_value and i not in covered_indices:
                term = bin(i)[2:].zfill(len(self.variables))
                single_groups.append((term, frozenset([i])))
                covered_indices.add(i)
        return single_groups

    def _select_essential_groups(self, groups, outputs, target_value):
        """Выбирает эссенциальные группы для покрытия всех минтермов/макстермов."""
        essential_groups = []
        minterms_set = set(i for i, val in enumerate(outputs) if val == target_value)
        covered = set()
        seen_terms = set()
        for term, indices in sorted(groups, key=lambda x: len(x[1]), reverse=True):
            if term not in seen_terms:
                unique = indices - covered
                if unique and any(i in minterms_set for i in unique):
                    essential_groups.append((term, indices))
                    covered.update(indices)
                    seen_terms.add(term)
        return essential_groups

    def _find_groups(self, target_value, n, sizes):
        """Находит группы ячеек с заданным значением (1 для СДНФ, 0 для СКНФ)."""
        rows = 2 if n == 3 else 4
        cols = 4 if n < 5 else 8
        groups = []
        covered_indices = set()
        outputs = [int(result) for _, _, result in self.table]

        for size in sizes:
            for height in [1, 2, 4][:min(rows, int(size).bit_length())]:
                if size % height != 0:
                    continue
                width = size // height
                if width > cols:
                    continue
                for r_start in range(rows):
                    for c_start in range(cols):
                        group_indices = self._generate_group_indices(n, r_start, c_start, height, width, rows, cols)
                        if group_indices and self._validate_group(group_indices, outputs, target_value):
                            term = self._group_to_term(group_indices, n)
                            groups.append((term, frozenset(group_indices)))

        groups.extend(self._add_single_cells(outputs, target_value, covered_indices))
        return self._select_essential_groups(groups, outputs, target_value)

    def _print_karnaugh_map_header(self, n):
        """Выводит заголовок карты Карно."""
        gray_code_2 = ['00', '01', '11', '10']
        gray_code_3 = ['000', '001', '011', '010', '110', '111', '101', '100']
        if n == 3:
            print("\nКарта Карно (3 переменные):")
            print(" a\\bc | 00 | 01 | 11 | 10")
            print("-------+----+----+----+----")
        elif n == 4:
            print("\nКарта Карно (4 переменные):")
            print(" ab\\cd | 00 | 01 | 11 | 10")
            print("--------+----+----+----+----")
        elif n == 5:
            print("\nКарта Карно (5 переменных):")
            print(" ab\\cde | 000 | 001 | 011 | 010 | 110 | 111 | 101 | 100")
            print("---------+-----+-----+-----+-----+-----+-----+-----+-----")

    def _print_karnaugh_map_rows(self, n):
        """Выводит строки карты Карно."""
        gray_code_2 = ['00', '01', '11', '10']
        gray_code_3 = ['000', '001', '011', '010', '110', '111', '101', '100']
        outputs = [int(result) for _, _, result in self.table]
        if n == 3:
            for a in [0, 1]:
                row = f" {a}    |"
                for bc in gray_code_2:
                    b, c = int(bc[0]), int(bc[1])
                    idx = a * 4 + b * 2 + c
                    row += f"  {outputs[idx]} |"
                print(row)
        elif n == 4:
            for ab in gray_code_2:
                a, b = int(ab[0]), int(ab[1])
                row = f" {ab}   |"
                for cd in gray_code_2:
                    c, d = int(cd[0]), int(cd[1])
                    idx = a * 8 + b * 4 + c * 2 + d
                    row += f"  {outputs[idx]} |"
                print(row)
        elif n == 5:
            for ab in gray_code_2:
                a, b = int(ab[0]), int(ab[1])
                row = f" {ab}    |"
                for cde in gray_code_3:
                    c, d, e = int(cde[0]), int(cde[1]), int(cde[2])
                    idx = a * 16 + b * 8 + c * 4 + d * 2 + e
                    row += f"  {outputs[idx]}  |"
                print(row)

    def _minimize_by_karnaugh(self, n):
        """Выполняет минимизацию СДНФ и СКНФ по карте Карно."""
        sizes = [4, 2, 1] if n == 3 else [8, 4, 2, 1] if n == 4 else [16, 8, 4, 2, 1]
        dnf_groups = self._find_groups(1, n, sizes)
        minimized_dnf = ' | '.join(
            self.term_to_expression(g[0], is_dnf=True) for g in dnf_groups) if dnf_groups else '0'
        cnf_groups = self._find_groups(0, n, sizes)
        minimized_cnf = ' & '.join(
            f'({self.term_to_expression(g[0], is_dnf=False)})' for g in cnf_groups) if cnf_groups else '1'
        return minimized_dnf, minimized_cnf

    def print_karnaugh_map(self):
        n = len(self.variables)
        if n > 5:
            print("Карта Карно для более чем 5 переменных не поддерживается.")
            return
        self._print_karnaugh_map_header(n)
        self._print_karnaugh_map_rows(n)
        return self._minimize_by_karnaugh(n)

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
        print("Результат:", min_sdnf)

        print("\nМинимизация СКНФ (расчетный метод):")
        min_sknf, sknf_gluing = self.minimize_sknf_calculation()
        print("Результат:", min_sknf)

        print("\nМинимизация СДНФ (расчетно-табличный метод):")
        min_sdnf_table, sdnf_table_gluing = self.minimize_sdnf_table()
        print("Результат:", min_sdnf_table)

        print("\nМинимизация СКНФ (расчетно-табличный метод):")
        min_sknf_table, sknf_table_gluing = self.minimize_sknf_table()
        print("Результат:", min_sknf_table)

        print("\nМинимизация СДНФ (Карта Карно):")
        min_sdnf, _ = self.print_karnaugh_map()
        print("Результат:", min_sdnf)

        print("\nМинимизация СКНФ (Карта Карно):")
        _, min_sknf = self.print_karnaugh_map()
        print("Результат:", min_sknf)


def main():
    infix = input("Введите логическую функцию (например, '((a&b)|(!c))' или '(((a&b)|(c<->d))&e)'): ")
    try:
        logic_expr = LogicExpression(infix)
        logic_expr.print_results()
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()