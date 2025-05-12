from calculation_table import CalculationTable
from karno_sdnf_maps import KarnoSDNF

LITERALS = ['q3_star', 'q2_star', 'q1_star', 'V']

def build_transition_table():
    table = []
    for q3_star in range(2):
        for q2_star in range(2):
            for q1_star in range(2):
                for v in range(2):
                    state = [q3_star, q2_star, q1_star]
                    decimal = q3_star * 4 + q2_star * 2 + q1_star * 1
                    if v == 0:
                        next_decimal = decimal
                    else:
                        next_decimal = (decimal - 1) % 8
                    q3 = (next_decimal // 4) % 2
                    q2 = (next_decimal // 2) % 2
                    q1 = next_decimal % 2
                    h3 = 1 if q3_star != q3 else 0
                    h2 = 1 if q2_star != q2 else 0
                    h1 = 1 if q1_star != q1 else 0
                    table.append(([q3_star, q2_star, q1_star, v], [q3, q2, q1], [h3, h2, h1]))
    return table

def minimize_excitation_functions():
    table = build_transition_table()
    print("Таблица переходов и возбуждения:")
    print("q3* q2* q1*  V  | q3 q2 q1  | h3 h2 h1")
    for row in table:
        inputs, outputs, excitations = row
        print(f"{inputs[0]}  {inputs[1]}  {inputs[2]}  {inputs[3]}  | "
              f"{outputs[0]} {outputs[1]} {outputs[2]}  | "
              f"{excitations[0]} {excitations[1]} {excitations[2]}")

    table_h3 = [(row[0], row[2][0]) for row in table]
    karno_h3 = KarnoSDNF(table_h3, LITERALS)
    SDNF_h3 = karno_h3.karno_1_to_4_sdnf()

    table_h2 = [(row[0], row[2][1]) for row in table]
    karno_h2 = KarnoSDNF(table_h2, LITERALS)
    SDNF_h2 = karno_h2.karno_1_to_4_sdnf()

    table_h1 = [(row[0], row[2][2]) for row in table]
    karno_h1 = KarnoSDNF(table_h1, LITERALS)
    SDNF_h1 = karno_h1.karno_1_to_4_sdnf()

    print("\nМинимизированные функции возбуждения:")
    print(f"h3 = {SDNF_h3}")
    print(f"h2 = {SDNF_h2}")
    print(f"h1 = {SDNF_h1}")

if __name__ == "__main__":
    minimize_excitation_functions()