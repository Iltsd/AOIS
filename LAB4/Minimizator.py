from typing import List, Set, Optional
from constants import LITERALS, BINARY_OPERANDS


class Minimizator:
    @staticmethod
    def check_if_neighbours(first: str, second: str) -> Optional[Set[str]]:
        literals_f, literals_s = Minimizator.find_literals(first), Minimizator.find_literals(second)
        diff = literals_f.symmetric_difference(literals_s)

        if len(diff) != 2:
            return None

        a, b = diff
        if (a.startswith("!") and a[1:] == b) or (b.startswith("!") and b[1:] == a):
            return literals_f.intersection(literals_s)
        return None

    @staticmethod
    def check_form(FORM: List[str]) -> Optional[str]:
        if not FORM:
            return None
        for term in FORM:
            for char in term:
                if char in BINARY_OPERANDS:
                    return char
        return None

    @staticmethod
    def scleivanie(FORM: List[str], operand: str) -> List[str]:
        if not FORM:
            return []

        length = len(FORM)
        with_neighbours = [0] * length
        new_FORM = []

        for i in range(length):
            for j in range(i + 1, length):
                is_neighbour = Minimizator.check_if_neighbours(FORM[i], FORM[j])
                if is_neighbour is not None:
                    with_neighbours[i] = 1
                    with_neighbours[j] = 1
                    neighbor_excluded = "(" + operand.join(sorted(is_neighbour)) + ")"
                    new_FORM.append(neighbor_excluded)

        for i in range(length):
            if with_neighbours[i] == 0:
                new_FORM.append(FORM[i])

        return new_FORM

    @staticmethod
    def scleivanie_till_end_FORM(FORM: List[str]) -> List[Set[str]]:
        steps = []
        operand = Minimizator.check_form(FORM)
        if operand is None:
            return [set(FORM)]

        new_FORM = Minimizator.scleivanie(FORM, operand)
        steps.append(set(FORM))

        while new_FORM != FORM:
            FORM = new_FORM.copy()
            steps.append(set(FORM))
            new_FORM = Minimizator.scleivanie(FORM, operand)

        return steps

    @staticmethod
    def find_literals(form: str) -> Set[str]:
        literals = set()
        i = 0
        while i < len(form):
            if form[i] in LITERALS:
                if i > 0 and form[i - 1] == "!":
                    literals.add(f"!{form[i]}")
                else:
                    literals.add(form[i])
            i += 1
        return literals