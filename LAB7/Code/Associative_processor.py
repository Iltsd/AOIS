class AssociativeProcessor:
    def __init__(self):
        # Инициализация матрицы 16x16
        self.matrix = [
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
            [1,1,0,1,1,0,0,0,0,1,1,1,1,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ]

    # Чтение слова S_k
    def read_word(self, k):
        word = 0
        for m in range(16):
            i = (m + k) % 16
            bit = self.matrix[i][k]
            word |= (bit << m)
        return word

    # Запись слова S_k
    def write_word(self, k, val):
        for m in range(16):
            i = (m + k) % 16
            bit = (val >> m) & 1
            self.matrix[i][k] = bit

    # Чтение битового столбца m
    def read_bit_column(self, m):
        return [self.matrix[(m + k) % 16][k] for k in range(16)]

    # Логические функции
    def f1(self, bit_column):
        # Возвращает 1, если все биты равны 1 (логическое И)
        return int(all(bit_column))

    def f3(self, bit_column):
        # Возвращает 1, если хотя бы один бит равен 1 (логическое ИЛИ)
        return int(any(bit_column))

    def f12(self, bit_column):
        # Возвращает 1, если чётное количество битов равны 1 (чётность)
        return int(sum(bit_column) % 2 == 0)

    def f14(self, bit_column):
        # Возвращает 1, если нечётное количество битов равны 1 (нечётность)
        return int(sum(bit_column) % 2 != 0)

    # Поиск ближайшего значения выше V
    def nearest_above(self, V):
        min_above = None
        for k in range(16):
            S_k = self.read_word(k)
            if S_k > V:
                if min_above is None or S_k < min_above:
                    min_above = S_k
        return min_above

    # Поиск ближайшего значения ниже V
    def nearest_below(self, V):
        max_below = None
        for k in range(16):
            S_k = self.read_word(k)
            if S_k < V:
                if max_below is None or S_k > max_below:
                    max_below = S_k
        return max_below

    # Арифметическая операция
    def arithmetic_operation(self, V):
        for k in range(16):
            S_k = self.read_word(k)
            Vj = (S_k >> 13) & 7  # Биты 15-13
            if Vj == V:
                Aj = (S_k >> 9) & 15  # Биты 12-9
                Bj = (S_k >> 5) & 15  # Биты 8-5
                sum_ab = (Aj + Bj) % 32  # Сумма по модулю 32
                mask = ~((1 << 5) - 1) & ((1 << 16) - 1)  # Очистка битов 4-0
                S_k_new = (S_k & mask) | (sum_ab & ((1 << 5) - 1))
                self.write_word(k, S_k_new)

# Пример использования (можно закомментировать при импорте)
if __name__ == "__main__":
    ap = AssociativeProcessor()

    # Вывод всех слов
    print("Исходные слова:")
    for k in range(16):
        print(f"Слово {k}: {bin(ap.read_word(k))[2:].zfill(16)}")

    # Выполнение арифметической операции для V=1
    print("\nВыполняем арифметическую операцию для V=1")
    ap.arithmetic_operation(1)

    # Вывод обновлённого слова 0
    print(f"Обновлённое слово 0: {bin(ap.read_word(0))[2:].zfill(16)}")

    # Пример работы логических функций
    bit_col = ap.read_bit_column(0)
    print("\nПример битового столбца для m=0:", bit_col)
    print("f1 (все биты 1):", ap.f1(bit_col))
    print("f3 (хотя бы один бит 1):", ap.f3(bit_col))
    print("f12 (чётное количество 1):", ap.f12(bit_col))
    print("f14 (нечётное количество 1):", ap.f14(bit_col))

    # Пример поиска ближайших значений
    V_example = 10000
    above = ap.nearest_above(V_example)
    below = ap.nearest_below(V_example)
    print(f"\nБлижайшее значение выше {V_example}: {above}")
    print(f"Ближайшее значение ниже {V_example}: {below}")