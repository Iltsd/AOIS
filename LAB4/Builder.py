from typing import List
from constants import LITERALS

class Builder:
    @staticmethod
    def build_SDNF(literals: List[str], bits: List[int]):
        res = []
        for i in range(len(literals)):
            if bits[i]:
                res.append(str(literals[i]))
            else:
                res.append("!" + str(literals[i]))
        return "(" + "&".join(res) + ")"

    @staticmethod
    def build_SKNF(literals: List[str], bits: List[int]):
        res = []
        for i in range(len(literals)):
            if bits[i] == 0:
                res.append(str(literals[i]))
            else:
                res.append("!" + str(literals[i]))
        return "(" + "|".join(res) + ")"