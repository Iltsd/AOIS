from typing import List
import re
from constants import LITERALS
from Binary_helper import Binary_helper

class Parser:
    @staticmethod
    def get_arguments(s: str):
        letters = []
        for i in range(len(s)):
            if s[i] in LITERALS and s[i] not in letters:
                letters.append(s[i])
        return letters

    @staticmethod
    def work_w_str(s: str, letters: List[str]):
        bits = [0] * len(letters)
        one = [0] * (len(letters) - 1) + [1]
        while True:
            new_s = s
            for i in range(len(letters)):
                new_s = new_s.replace(letters[i], str(bits[i]))
            new_s = new_s.replace("->", ">")
            new_s = new_s.replace("\/", "|")
            yield new_s, bits
            bits = Binary_helper.sum_b(bits, one)
            if sum(bits) == 0:
                break

    @staticmethod
    def delete_brackets(s: str):
        new_s = s
        for i in re.findall("\([0,1]\)", new_s):
            new_s = new_s.replace(i, i[1])
        return new_s

    @staticmethod
    def delete_tabulation(s: str):
        new_s = s
        for i in re.findall(r"\s", new_s):
            new_s = new_s.replace(i, "")
        return new_s