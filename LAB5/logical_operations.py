class LogicalOperations:
    @classmethod
    def conjunction(cls, elem1, elem2):
        if elem1 == 1 and elem2 == 1:
            return 1
        return 0

    @classmethod
    def disjunction(cls, elem1, elem2):
        if elem1 == 1 or elem2 == 1:
            return 1
        return 0

    @classmethod
    def negation(cls, elem1):
        if elem1 == 1:
            return 0
        return 1

    @classmethod
    def implication(cls, elem1, elem2):
        if elem1 == 1 and elem2 == 0:
            return 0
        return 1

    @classmethod
    def equivalence(cls, elem1, elem2):
        if elem1 == elem2:
            return 1
        return 0

    @classmethod
    def priority(cls, op):
        if op == '!':
            return 5
        if op == '&':
            return 4
        if op == '|':
            return 3
        if op == '>':
            return 2
        if op == '~':
            return 1
        return 0

    @classmethod
    def evaluate_expression(cls, parts, values):
        rpn_result = []
        temp_ops = []
        calc_stack = []

        for part in parts:
            if part.isalpha():
                rpn_result.append(values[part])
            elif part == '(':
                temp_ops.append(part)
            elif part == ')':
                while len(temp_ops) > 0 and temp_ops[-1] != '(':
                    rpn_result.append(temp_ops.pop())
                if len(temp_ops) > 0:
                    temp_ops.pop()
            elif part in ('!', '&', '|', '~', '>'):
                while len(temp_ops) > 0 and temp_ops[-1] != '(' and cls.priority(temp_ops[-1]) >= cls.priority(part):
                    rpn_result.append(temp_ops.pop())
                temp_ops.append(part)

        while temp_ops:
            rpn_result.append(temp_ops.pop())

        for thing in rpn_result:
            if isinstance(thing, int):
                calc_stack.append(thing)
            elif thing == '!':
                if not calc_stack:
                    raise ValueError("Недостаточно операндов для операции '!'")
                one = calc_stack.pop()
                calc_stack.append(cls.negation(one))
            else:
                if len(calc_stack) < 2:
                    raise ValueError(f"Недостаточно операндов для операции '{thing}'")
                second = calc_stack.pop()
                first = calc_stack.pop()
                if thing == '&':
                    calc_stack.append(cls.conjunction(first, second))
                elif thing == '|':
                    calc_stack.append(cls.disjunction(first, second))
                elif thing == '>':
                    calc_stack.append(cls.implication(first, second))
                elif thing == '~':
                    calc_stack.append(cls.equivalence(first, second))

        if not calc_stack:
            raise ValueError("Выражение не содержит результата")
        if len(calc_stack) > 1:
            raise ValueError("Выражение содержит лишние операнды")

        return calc_stack[0]

    @classmethod
    def bin_to_dec(cls, binary: str) -> int:
        return cls.binary_to_dec_num(binary)

    @classmethod
    def binary_to_dec_num(cls, binary: str) -> int:
        if not binary:
            return 0
        decimal = 0
        for i, bit in enumerate(reversed(binary)):
            if bit == '1':
                decimal += 2 ** i
        return decimal

    @classmethod
    def results_to_string(cls, table):
        result_str = ''
        for row in table:
            result = row[1]
            result_str += str(int(result))
        return result_str

    @classmethod
    def unique_elements(cls, expression):
        elements = []
        for char in expression:
            if ('A' <= char <= 'Z') or ('a' <= char <= 'z'):
                if char not in elements:
                    elements.append(char)
        elements.sort()
        return elements

    @classmethod
    def numerical_form(cls, table):
        sdnf_nums = []
        sknf_nums = []
        for index, row in enumerate(table):
            values = row[0]
            result = row[1]
            if result == 1:
                sdnf_nums.append(index)
            else:
                sknf_nums.append(index)
        sdnf_str = f"({', '.join(str(i) for i in sdnf_nums)})∨"
        sknf_str = f"({', '.join(str(i) for i in sknf_nums)})∧"
        return sdnf_str, sknf_str

    @classmethod
    def num_to_bin(cls, digit: int, size: int) -> str:
        binary = ""
        current_num = digit
        for _ in range(size):
            remainder = current_num % 2
            binary = str(remainder) + binary
            current_num = current_num // 2
        return binary

    @classmethod
    def position_replacement(cls, pos1: str, pos2: str) -> int:
        differences = 0
        for i in range(len(pos1)):
            char1 = pos1[i]
            char2 = pos2[i]
            if char1 != char2:
                differences = differences + 1
        return differences

    @classmethod
    def is_combinated(cls, num1: str, num2: str) -> bool:
        return cls.position_replacement(num1, num2) == 1

    @classmethod
    def merge(cls, str1: str, str2: str) -> str:
        result = ""
        for i in range(len(str1)):
            char1 = str1[i]
            char2 = str2[i]
            if char1 == char2:
                result = result + char1
            else:
                result = result + '-'
        return result