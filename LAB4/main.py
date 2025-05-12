from Builder import Builder
from Binary_helper import Binary_helper
from constants import LITERALS, LEN_OF_TETRADA


def get_adder():
    values = [0] * 3  # A, B, C
    one = [0] * 2 + [1]
    literals = list(LITERALS[:3])  # A, B, C
    SDNF_S = []
    SDNF_Cout = []

    for i in range(2 ** 3):
        S = values[0] ^ values[1] ^ values[2]  # Сумма: A ⊕ B ⊕ C
        Cout = int((values[0] and values[1]) or (values[0] and values[2]) or (
                    values[1] and values[2]))  # Перенос: AB | AC | BC
        if S == 1:
            SDNF_S.append(Builder.build_SDNF(literals, values))
        if Cout == 1:
            SDNF_Cout.append(Builder.build_SDNF(literals, values))
        values = Binary_helper.sum_b(values, one)

    return "|".join(SDNF_S), "|".join(SDNF_Cout)


def get_D8421_6():
    values = [0] * LEN_OF_TETRADA
    one = [0] * (LEN_OF_TETRADA - 1) + [1]

    print('D8421\t\t\tD8421+6')
    for i in range(16):
        decimal = Binary_helper.calculate(values)
        if decimal > 9:
            y3, y2, y1, y0 = "X", "X", "X", "X"
        else:
            dec_plus6 = decimal + 6
            y3 = (dec_plus6 // 8) % 2
            y2 = (dec_plus6 // 4) % 2
            y1 = (dec_plus6 // 2) % 2
            y0 = dec_plus6 % 2
        print(f"{values[0]} {values[1]} {values[2]} {values[3]}\t\t\t{y3} {y2} {y1} {y0}")
        values = Binary_helper.sum_b(values, one)

    # Заданные минимизированные выражения для Dec=0..9
    min_Y = {
        "Y1": "(!B&!C)|(B&D)",  # Y1 = !X2&!X1 | X2&X0
        "Y2": "(!A&!C)|(A&!B&!C)",  # Y2 = !X3&!X1 | X3&!X2&!X1
        "Y3": "(A)|(!A&C)",  # Y3 = X3 | !X3&X1
        "Y0": "(D)"  # Y0 = X0
    }

    return min_Y


def replace(form):
    form = form.replace("A", "X3")
    form = form.replace("B", "X2")
    form = form.replace("C", "X1")
    form = form.replace("D", "X0")
    return form


def replace_back(form):
    form = form.replace("X3", "A")
    form = form.replace("X2", "B")
    form = form.replace("X1", "C")
    form = form.replace("X0", "D")
    return form


if __name__ == "__main__":
    SDNF_S, SDNF_Cout = get_adder()
    print("СДНФ для S: " + SDNF_S, "СДНФ для C_out: " + SDNF_Cout, sep='\n')
    print()

    min_Y = get_D8421_6()
    print(f"Y1 = {replace(min_Y['Y1'])}")
    print(f"Y2 = {replace(min_Y['Y2'])}")
    print(f"Y3 = {replace(min_Y['Y3'])}")
    print(f"Y0 = {replace(min_Y['Y0'])}")