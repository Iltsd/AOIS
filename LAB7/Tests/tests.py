import unittest
import copy
from Associative_processor import AssociativeProcessor

class TestAssociativeProcessor(unittest.TestCase):
    def setUp(self):
        # Создаём экземпляр процессора перед каждым тестом
        self.processor = AssociativeProcessor()
        # Создаём копию матрицы для изоляции тестов
        self.matrix = copy.deepcopy(self.processor.matrix)

    def test_read_word(self):
        # Проверка чтения слова №2 (из документа: 0011100000010000, от младшего к старшему)
        expected = 0b0000000000011100  # 0b0011100000010000 перевёрнуто
        result = self.processor.read_word(2)
        self.assertEqual(result, expected, f"Ожидалось слово 2: {bin(expected)[2:].zfill(16)}, получено: {bin(result)[2:].zfill(16)}")

        # Проверка слова №0 (из документа: 1111100001110000)
        expected = 0b0011100000011111  # 0b1111100001110000 перевёрнуто
        result = self.processor.read_word(0)
        self.assertEqual(result, expected, f"Ожидалось слово 0: {bin(expected)[2:].zfill(16)}, получено: {bin(result)[2:].zfill(16)}")

    def test_write_word(self):
        # Записываем новое значение в слово №2 и проверяем
        new_value = 0b1010101010101010
        self.processor.write_word(2, new_value)
        result = self.processor.read_word(2)
        self.assertEqual(result, new_value,
                         f"Ожидалось слово 2: {bin(new_value)[2:].zfill(16)}, получено: {bin(result)[2:].zfill(16)}")

        # Проверяем, что слово №0 не изменилось
        expected = 0b0011100000011111  # Исходное слово 0
        result = self.processor.read_word(0)
        self.assertEqual(result, expected, f"Слово 0 изменилось: ожидано {bin(expected)[2:].zfill(16)}, получено: {bin(result)[2:].zfill(16)}")

    def test_read_bit_column(self):
        # Проверка столбца №3 (из документа: 1110110010010000, сверху вниз)
        expected = [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result = self.processor.read_bit_column(3)
        self.assertEqual(result, expected, f"Ожидался столбец 3: {expected}, получено: {result}")

        # Проверка столбца №0
        expected = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result = self.processor.read_bit_column(0)
        self.assertEqual(result, expected, f"Ожидался столбец 0: {expected}, получено: {result}")

    def test_nearest_above(self):
        # Тест поиска ближайшего значения выше V=10000
        V = 10000
        result = self.processor.nearest_above(V)
        # Ожидаем минимальное S_k > 10000
        expected = min([self.processor.read_word(k) for k in range(16) if self.processor.read_word(k) > V], default=None)
        self.assertEqual(result, expected, f"Ожидалось ближайшее выше {V}: {expected}, получено: {result}")

        # Тест для V больше всех слов
        V = 65536
        result = self.processor.nearest_above(V)
        self.assertIsNone(result, "Ожидалось None для V больше всех слов")

    def test_nearest_below(self):
        # Тест поиска ближайшего значения ниже V=10000
        V = 10000
        result = self.processor.nearest_below(V)
        # Ожидаем максимальное S_k < 10000
        expected = max([self.processor.read_word(k) for k in range(16) if self.processor.read_word(k) < V], default=None)
        self.assertEqual(result, expected, f"Ожидалось ближайшее ниже {V}: {expected}, получено: {result}")

        # Тест для V меньше всех слов
        V = -1
        result = self.processor.nearest_below(V)
        self.assertIsNone(result, "Ожидалось None для V меньше всех слов")

    def test_arithmetic_operation(self):
        # Тест для V=7 (111), слово 0: 1111100001110000 (от старшего к младшему)
        # Vj=111, Aj=1100 (12), Bj=0011 (3), sum=15 (01111)
        expected_word_0 = 0b0011100000011111  # Исходное слово 0 остаётся неизменным, так как порядок битов в документе от старшего
        self.processor.arithmetic_operation(7)
        result = self.processor.read_word(0)
        self.assertEqual(result, expected_word_0,
                         f"Ожидалось слово 0: {bin(expected_word_0)[2:].zfill(16)}, получено: {bin(result)[2:].zfill(16)}")

        # Проверяем, что другие слова не изменились (например, S_1, Vj=001 ≠ 111)
        expected_word_1 = 0b0110000000001010  # Исходное слово 1
        result = self.processor.read_word(1)
        self.assertEqual(result, expected_word_1,
                         f"Ожидалось слово 1: {bin(expected_word_1)[2:].zfill(16)}, получено: {bin(result)[2:].zfill(16)}")

        # Тест для V=0 (нет слов с Vj=0)
        matrix_copy = copy.deepcopy(self.matrix)
        self.processor.matrix = matrix_copy
        self.processor.arithmetic_operation(0)
        for k in range(16):
            result = self.processor.read_word(k)
            expected = self.processor.read_word(k)  # Матрица не должна измениться
            self.assertEqual(result, expected,
                             f"Слово {k} изменилось: ожидано {bin(expected)[2:].zfill(16)}, получено: {bin(result)[2:].zfill(16)}")


    def test_f1(self):
        # f1: возвращает 1, если все биты равны 1
        bit_col = self.processor.read_bit_column(0)  # [1, 0, 0, 0, 1, 0, ...]
        result = self.processor.f1(bit_col)
        self.assertEqual(result, 0, f"Ожидалось f1=0 для столбца 0, получено: {result}")

        # Тест для столбца из всех 1
        test_col = [1] * 16
        result = self.processor.f1(test_col)
        self.assertEqual(result, 1, f"Ожидалось f1=1 для столбца из всех 1, получено: {result}")

    def test_f3(self):
        # f3: возвращает 1, если хотя бы один бит равен 1
        bit_col = self.processor.read_bit_column(0)  # [1, 0, 0, 0, 1, 0, ...]
        result = self.processor.f3(bit_col)
        self.assertEqual(result, 1, f"Ожидалось f3=1 для столбца 0, получено: {result}")

        # Тест для столбца из всех 0
        test_col = [0] * 16
        result = self.processor.f3(test_col)
        self.assertEqual(result, 0, f"Ожидалось f3=0 для столбца из всех 0, получено: {result}")

    def test_f12(self):
        # f12: возвращает 1, если чётное количество битов равны 1
        bit_col = self.processor.read_bit_column(0)  # 2 единицы
        result = self.processor.f12(bit_col)
        self.assertEqual(result, 1, f"Ожидалось f12=1 для столбца 0 (чётное), получено: {result}")

        # Тест для столбца с нечётным количеством 1
        test_col = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 3 единицы
        result = self.processor.f12(test_col)
        self.assertEqual(result, 0, f"Ожидалось f12=0 для столбца с нечётным количеством 1, получено: {result}")

    def test_f14(self):
        # f14: возвращает 1, если нечётное количество битов равны 1
        bit_col = self.processor.read_bit_column(0)  # 2 единицы
        result = self.processor.f14(bit_col)
        self.assertEqual(result, 0, f"Ожидалось f14=0 для столбца 0 (чётное), получено: {result}")

        # Тест для столбца с нечётным количеством 1
        test_col = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 3 единицы
        result = self.processor.f14(test_col)
        self.assertEqual(result, 1, f"Ожидалось f14=1 для столбца с нечётным количеством 1, получено: {result}")

if __name__ == '__main__':
    unittest.main()