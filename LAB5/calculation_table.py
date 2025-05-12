from logical_operations import LogicalOperations

class CalculationTable(LogicalOperations):
    def __init__(self, truth_table=None, variable_names=None, expression=None, max_vars=5):
        super().__init__()
        self.truth_table = truth_table if truth_table is not None else []
        self.variable_names = variable_names if variable_names is not None else []
        self.expression = expression if expression is not None else ""
        self.max_vars = max_vars
        self.stage = 1
        self.column_width = 1
        self.num_vars = 0

    def calculate_table_sdnf(self, verbose=True):
        table = self.truth_table
        names = self.variable_names
        num_vars = len(names)

        minterms = []
        for i in range(len(table)):
            if table[i][1] == 1:
                minterms.append(i)

        if not minterms:
            return "Нет СДНФ"

        terms = {}
        for m in minterms:
            binary = self.num_to_bin(m, num_vars)
            terms[tuple(binary)] = [m]

        if verbose:
            print("\nМинимизация СДНФ расчетно-табличным методом:")
            print("Таблица покрытия:")
            print("Минтермы:", minterms)
            print("Начальные термы:", list(terms.keys()))

        self.stage = 1
        while True:
            new_terms = {}
            used = set()
            term_list = list(terms.keys())
            has_new = False

            for i in range(len(term_list)):
                for j in range(i + 1, len(term_list)):
                    t1 = term_list[i]
                    t2 = term_list[j]
                    if self.is_combinated(t1, t2):
                        result = self.merge(t1, t2)
                        t1_str = ""
                        for bit in t1:
                            if bit == '0':
                                t1_str += '0'
                            elif bit == '1':
                                t1_str += '1'
                            else:
                                t1_str += 'X'
                        t2_str = ""
                        for bit in t2:
                            if bit == '0':
                                t2_str += '0'
                            elif bit == '1':
                                t2_str += '1'
                            else:
                                t2_str += 'X'
                        result_str = ""
                        for bit in result:
                            if bit == '0':
                                result_str += '0'
                            elif bit == '1':
                                result_str += '1'
                            else:
                                result_str += 'X'
                        if verbose:
                            print(f"({t1_str}) v ({t2_str}) -> ({result_str})")
                        new_terms[tuple(result)] = terms[t1] + terms[t2]
                        used.add(t1)
                        used.add(t2)
                        has_new = True

            for t in terms:
                if t not in used:
                    new_terms[t] = terms[t]

            display_terms = []
            for t in new_terms.keys():
                term_str = ""
                for bit in t:
                    if bit == '0':
                        term_str += '0'
                    elif bit == '1':
                        term_str += '1'
                    else:
                        term_str += 'X'
                display_terms.append(f"({term_str})")
            if verbose:
                print(display_terms)

            if not has_new:
                break
            terms = new_terms
            self.stage = self.stage + 1

        coverage_table = {}
        for i in range(len(terms)):
            term = list(terms.keys())[i]
            coverage_table[i] = []
            for j in range(len(minterms)):
                coverage_table[i].append(False)

        for i in range(len(terms)):
            term = list(terms.keys())[i]
            for j in range(len(minterms)):
                minterm = minterms[j]
                term_binary = term
                minterm_binary = self.num_to_bin(minterm, num_vars)
                match = True
                for k in range(len(term_binary)):
                    t_bit = term_binary[k]
                    m_bit = minterm_binary[k]
                    if t_bit != '-' and t_bit != m_bit:
                        match = False
                        break
                if match:
                    coverage_table[i][j] = True

        if verbose:
            print("\nконституэнты")
        minterm_names = []
        for m in range(len(minterms)):
            minterm = minterms[m]
            minterm_binary = self.num_to_bin(minterm, num_vars)
            minterm_name_parts = []
            for i in range(len(minterm_binary)):
                bit = minterm_binary[i]
                if bit == '1':
                    minterm_name_parts.append(names[i])
                else:
                    minterm_name_parts.append('!' + names[i])
            minterm_name = '(' + ''.join(minterm_name_parts) + ')'
            minterm_names.append(minterm_name)

        term_names = []
        for t in range(len(terms)):
            term = list(terms.keys())[t]
            term_name_parts = []
            for i in range(len(term)):
                bit = term[i]
                if bit == '1':
                    term_name_parts.append(names[i])
                elif bit == '0':
                    term_name_parts.append('!' + names[i])
            term_name = '(' + ''.join(term_name_parts) + ')'
            term_names.append(term_name)

        max_term_length = 0
        for name in term_names:
            if len(name) > max_term_length:
                max_term_length = len(name)

        max_minterm_length = 0
        for name in minterm_names:
            if len(name) > max_minterm_length:
                max_minterm_length = len(name)

        self.column_width = max_minterm_length
        if self.column_width < 1:
            self.column_width = 1

        if verbose:
            header = " " * max_term_length + " | "
            for name in minterm_names:
                spaces_before = (self.column_width - len(name)) // 2
                spaces_after = self.column_width - len(name) - spaces_before
                header += " " * spaces_before + name + " " * spaces_after + " | "
            print(header)

            for i in range(len(term_names)):
                term_name = term_names[i]
                row = term_name + " " * (max_term_length - len(term_name)) + " | "
                for j in range(len(minterms)):
                    if coverage_table[i][j]:
                        mark = "X"
                    else:
                        mark = " "
                    spaces_before = (self.column_width - len(mark)) // 2
                    spaces_after = self.column_width - len(mark) - spaces_before
                    row += " " * spaces_before + mark + " " * spaces_after + " | "
                print(row)

        essential = []
        covered = set()
        for col in range(len(minterms)):
            covering = []
            for i in range(len(coverage_table)):
                if coverage_table[i][col]:
                    covering.append(i)
            if len(covering) == 1:
                imp_idx = covering[0]
                if imp_idx not in essential:
                    essential.append(imp_idx)
                    for j in range(len(coverage_table[imp_idx])):
                        if coverage_table[imp_idx][j]:
                            covered.add(j)

        remaining = set()
        for i in range(len(minterms)):
            if i not in covered:
                remaining.add(i)

        while remaining:
            best_imp_idx = None
            best_count = 0
            for i in range(len(coverage_table)):
                if i in essential:
                    continue
                count = 0
                for j in remaining:
                    if coverage_table[i][j]:
                        count = count + 1
                if count > best_count:
                    best_count = count
                    best_imp_idx = i
            if best_imp_idx is None or best_count == 0:
                break
            essential.append(best_imp_idx)
            for j in range(len(coverage_table[best_imp_idx])):
                if coverage_table[best_imp_idx][j]:
                    covered.add(j)
            remaining = set()
            for i in range(len(minterms)):
                if i not in covered:
                    remaining.add(i)

        result = []
        for idx in essential:
            term = list(terms.keys())[idx]
            term_parts = []
            for i in range(len(term)):
                bit = term[i]
                if bit == '1':
                    term_parts.append(names[i])
                elif bit == '0':
                    term_parts.append('!' + names[i])
            if term_parts:
                if len(term_parts) == 1:
                    result.append(term_parts[0])
                else:
                    result.append('(' + ' ∧ '.join(term_parts) + ')')

        final_expression = ' v '.join(result)
        if not result:
            final_expression = "0"
        if verbose:
            print("убираем лишние импликанты и получаем:")
            print(final_expression)
        return final_expression

    def calculate_table_sknf(self, verbose=True):
        table = self.truth_table
        names = self.variable_names
        num_vars = len(names)
        if num_vars > self.max_vars:
            raise ValueError(f"Метод поддерживает до {self.max_vars} переменных.")

        maxterms = []
        for i in range(len(table)):
            if table[i][1] == 0:
                maxterms.append(i)

        if not maxterms:
            return "Нет СКНФ"

        terms = {}
        for m in maxterms:
            binary = self.num_to_bin(m, num_vars)
            terms[tuple(binary)] = [m]

        if verbose:
            print("\nМинимизация СКНФ расчетно-табличным методом:")
            print("Таблица покрытия:")
            print("Макстермы:", maxterms)
            print("Начальные термы:", list(terms.keys()))

        self.stage = 1
        while True:
            new_terms = {}
            used = set()
            term_list = list(terms.keys())
            has_new = False

            for i in range(len(term_list)):
                for j in range(i + 1, len(term_list)):
                    t1 = term_list[i]
                    t2 = term_list[j]
                    if self.is_combinated(t1, t2):
                        result = self.merge(t1, t2)
                        t1_str = ""
                        for bit in t1:
                            if bit == '0':
                                t1_str += '0'
                            elif bit == '1':
                                t1_str += '1'
                            else:
                                t1_str += 'X'
                        t2_str = ""
                        for bit in t2:
                            if bit == '0':
                                t2_str += '0'
                            elif bit == '1':
                                t2_str += '1'
                            else:
                                t2_str += 'X'
                        result_str = ""
                        for bit in result:
                            if bit == '0':
                                result_str += '0'
                            elif bit == '1':
                                result_str += '1'
                            else:
                                result_str += 'X'
                        if verbose:
                            print(f"({t1_str}) v ({t2_str}) -> ({result_str})")
                        new_terms[tuple(result)] = terms[t1] + terms[t2]
                        used.add(t1)
                        used.add(t2)
                        has_new = True

            for t in terms:
                if t not in used:
                    new_terms[t] = terms[t]

            display_terms = []
            for t in new_terms.keys():
                term_str = ""
                for bit in t:
                    if bit == '0':
                        term_str += '0'
                    elif bit == '1':
                        term_str += '1'
                    else:
                        term_str += 'X'
                display_terms.append(f"({term_str})")
            if verbose:
                print(f"Стадия {self.stage}:", display_terms)

            if not has_new:
                break
            terms = new_terms
            self.stage = self.stage + 1

        coverage_table = {}
        for i in range(len(terms)):
            term = list(terms.keys())[i]
            coverage_table[i] = []
            for j in range(len(maxterms)):
                coverage_table[i].append(False)

        for i in range(len(terms)):
            term = list(terms.keys())[i]
            for j in range(len(maxterms)):
                maxterm = maxterms[j]
                term_binary = term
                maxterm_binary = self.num_to_bin(maxterm, num_vars)
                match = True
                for k in range(len(term_binary)):
                    t_bit = term_binary[k]
                    m_bit = maxterm_binary[k]
                    if t_bit != '-' and t_bit != m_bit:
                        match = False
                        break
                if match:
                    coverage_table[i][j] = True

        if verbose:
            print("\nКОНСТИТУЭНТЫ")
        maxterm_names = []
        for m in range(len(maxterms)):
            maxterm = maxterms[m]
            maxterm_binary = self.num_to_bin(maxterm, num_vars)
            maxterm_name_parts = []
            for i in range(len(maxterm_binary)):
                bit = maxterm_binary[i]
                if bit == '0':
                    maxterm_name_parts.append('!' + names[i])
                else:
                    maxterm_name_parts.append(names[i])
            maxterm_name = '(' + ''.join(maxterm_name_parts) + ')'
            maxterm_names.append(maxterm_name)

        term_names = []
        for t in range(len(terms)):
            term = list(terms.keys())[t]
            term_name_parts = []
            for i in range(len(term)):
                bit = term[i]
                if bit == '0':
                    term_name_parts.append('!' + names[i])
                elif bit == '1':
                    term_name_parts.append(names[i])
            term_name = '(' + ''.join(term_name_parts) + ')'
            term_names.append(term_name)

        max_term_length = 0
        for name in term_names:
            if len(name) > max_term_length:
                max_term_length = len(name)

        max_maxterm_length = 0
        for name in maxterm_names:
            if len(name) > max_maxterm_length:
                max_maxterm_length = len(name)

        self.column_width = max_maxterm_length
        if self.column_width < 1:
            self.column_width = 1

        if verbose:
            header = " " * max_term_length + " | "
            for name in maxterm_names:
                spaces_before = (self.column_width - len(name)) // 2
                spaces_after = self.column_width - len(name) - spaces_before
                header += " " * spaces_before + name + " " * spaces_after + " | "
            print(header)

            for i in range(len(term_names)):
                term_name = term_names[i]
                row = term_name + " " * (max_term_length - len(term_name)) + " | "
                for j in range(len(maxterms)):
                    if coverage_table[i][j]:
                        mark = "X"
                    else:
                        mark = " "
                    spaces_before = (self.column_width - len(mark)) // 2
                    spaces_after = self.column_width - len(mark) - spaces_before
                    row += " " * spaces_before + mark + " " * spaces_after + " | "
                print(row)

        essential = []
        covered = set()
        for col in range(len(maxterms)):
            covering = []
            for i in range(len(coverage_table)):
                if coverage_table[i][col]:
                    covering.append(i)
            if len(covering) == 1:
                imp_idx = covering[0]
                if imp_idx not in essential:
                    essential.append(imp_idx)
                    for j in range(len(coverage_table[imp_idx])):
                        if coverage_table[imp_idx][j]:
                            covered.add(j)

        remaining = set()
        for i in range(len(maxterms)):
            if i not in covered:
                remaining.add(i)

        while remaining:
            best_imp_idx = None
            best_count = 0
            for i in range(len(coverage_table)):
                if i in essential:
                    continue
                count = 0
                for j in remaining:
                    if coverage_table[i][j]:
                        count = count + 1
                if count > best_count:
                    best_count = count
                    best_imp_idx = i
            if best_imp_idx is None or best_count == 0:
                break
            essential.append(best_imp_idx)
            for j in range(len(coverage_table[best_imp_idx])):
                if coverage_table[best_imp_idx][j]:
                    covered.add(j)
            remaining = set()
            for i in range(len(maxterms)):
                if i not in covered:
                    remaining.add(i)

        result = []
        for idx in essential:
            term = list(terms.keys())[idx]
            term_parts = []
            for i in range(len(term)):
                bit = term[i]
                if bit == '0':
                    term_parts.append(names[i])
                elif bit == '1':
                    term_parts.append('!' + names[i])
            if term_parts:
                if len(term_parts) == 1:
                    result.append(term_parts[0])
                else:
                    result.append('(' + ' ∨ '.join(term_parts) + ')')

        final_expression = ' ∧ '.join(result)
        if not result:
            final_expression = "1"
        if verbose:
            print("убираем лишние импликанты и получаем:")
            print(final_expression)
        return final_expression