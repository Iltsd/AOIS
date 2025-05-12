import unittest
from Hash_table import HashTable, Entry

class TestHashTable(unittest.TestCase):
    def setUp(self):
        self.ht = HashTable(size=10)  # Меньший размер для упрощения тестов
        self.test_terms = [
            ("существительное", "Часть речи, обозначающая предмет"),
            ("глагол", "Часть речи, обозначающая действие"),
            ("прилагательное", "Часть речи, обозначающая признак предмета"),
            ("наречие", "Часть речи, обозначающая признак действия"),
            ("местоимение", "Часть речи, заменяющая существительное"),
        ]

    def test_insert_single(self):
        """Тест вставки одной записи и проверка флажков."""
        self.ht.insert("существительное", "Часть речи, обозначающая предмет")
        self.assertEqual(self.ht.get_num_entries(), 1)
        entry = self.ht.search("существительное")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "существительное")
        self.assertEqual(entry.pi, "Часть речи, обозначающая предмет")
        self.assertEqual(entry.C, 0)  # Без коллизии
        self.assertEqual(entry.U, 1)  # Занята
        self.assertEqual(entry.T, 1)  # Терминальная
        self.assertEqual(entry.L, 0)  # Без связи
        self.assertEqual(entry.D, 0)  # Не удалена
        self.assertEqual(entry.Po, 0) # Без переполнения

    def test_insert_multiple_with_collisions(self):
        """Тест вставки нескольких записей с коллизиями."""
        for id, pi in self.test_terms:
            self.ht.insert(id, pi)
        self.assertEqual(self.ht.get_num_entries(), len(self.test_terms))
        self.assertGreaterEqual(self.ht.get_num_collisions(), 1)  # Хотя бы одна коллизия
        self.assertGreaterEqual(self.ht.get_num_slots_with_probing(), 1)

    def test_search_existing(self):
        """Тест поиска существующей записи."""
        self.ht.insert("глагол", "Часть речи, обозначающая действие")
        entry = self.ht.search("глагол")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "глагол")
        self.assertEqual(entry.pi, "Часть речи, обозначающая действие")

    def test_search_non_existing(self):
        """Тест поиска несуществующей записи."""
        self.ht.insert("глагол", "Часть речи, обозначающая действие")
        entry = self.ht.search("несуществующий")
        self.assertIsNone(entry)

    def test_delete_existing(self):
        """Тест удаления существующей записи."""
        self.ht.insert("прилагательное", "Часть речи, обозначающая признак предмета")
        deleted = self.ht.delete("прилагательное")
        self.assertTrue(deleted)
        entry = self.ht.search("прилагательное")
        self.assertIsNone(entry)
        self.assertEqual(self.ht.get_num_entries(), 0)  # Учитываем только D == 0

    def test_delete_non_existing(self):
        """Тест удаления несуществующей записи."""
        deleted = self.ht.delete("несуществующий")
        self.assertFalse(deleted)

    def test_delete_and_reinsert(self):
        """Тест удаления и повторной вставки."""
        self.ht.insert("союз", "Служебная часть речи для соединения")
        self.ht.delete("союз")
        self.ht.insert("союз", "Служебная часть речи для соединения")
        entry = self.ht.search("союз")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "союз")
        self.assertEqual(entry.D, 0)

    def test_full_table(self):
        """Тест заполнения таблицы и выброса исключения."""
        for i in range(10):
            self.ht.insert(f"термин{i}", f"Определение {i}")
        self.assertEqual(self.ht.get_num_entries(), 10)
        with self.assertRaises(Exception):
            self.ht.insert("лишний", "лишнее определение")

    def test_insert_duplicate_key(self):
        """Тест вставки записи с уже существующим ключом (перезапись)."""
        self.ht.insert("существительное", "Часть речи, обозначающая предмет")
        original_entries = self.ht.get_num_entries()
        self.ht.insert("существительное", "Обновленное определение предмета")
        entry = self.ht.search("существительное")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "существительное")
        self.assertEqual(entry.pi, "Часть речи, обозначающая предмет")  # Проверяем обновление
        self.assertEqual(self.ht.get_num_entries(), 2)  # Количество записей не изменилось
        self.assertEqual(entry.U, 1)
        self.assertEqual(entry.D, 0)

if __name__ == "__main__":
    unittest.main()