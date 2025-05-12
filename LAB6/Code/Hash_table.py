class Entry:
    def __init__(self, id, pi):
        self.id = id
        self.pi = pi
        self.C = 0  # Флажок коллизии
        self.U = 1  # Флажок занятости
        self.T = 1  # Терминальный флажок
        self.L = 0  # Флажок связи
        self.D = 0  # Флажок удаления
        self.Po = 0 # Указатель области переполнения

    def __repr__(self):
        return (f"Entry(id='{self.id}', pi='{self.pi}', "
                f"C={self.C}, U={self.U}, T={self.T}, L={self.L}, D={self.D}, Po={self.Po})")

class HashTable:
    def __init__(self, size=20):
        self.size = size
        self.table = [None] * size
        self.collisions = 0  # Счетчик коллизий
        self.probing_steps = {}  # Хранит количество шагов пробинга для каждого слота

    def hash1(self, key):
        # h1(k) = sum(ord(c)) mod size
        return sum(ord(c) for c in key) % self.size

    def hash2(self, key):
        # h2(k) = 1 + (prod(ord(c)) mod (size-1))
        step = 1
        for c in key:
            step = (step * ord(c)) % (self.size - 1)
        return 1 + step

    def insert(self, id, pi):
        index = self.hash1(id)
        step = self.hash2(id)
        attempts = 0
        original_index = index

        # Проверяем, занята ли ячейка и не помечена ли как удаленная
        while self.table[index] is not None and self.table[index].D == 0:
            self.collisions += 1
            attempts += 1
            if attempts >= self.size:
                raise Exception(f"Невозможно вставить элемент '{id}': таблица заполнена или зацикливание")
            index = (self.hash1(id) + attempts * step) % self.size

        entry = Entry(id, pi)
        if attempts > 0:
            entry.C = 1  # Устанавливаем флажок коллизии, если был пробинг
        self.table[index] = entry
        self.probing_steps[index] = attempts

    def search(self, id):
        """Поиск записи по ключу id. Возвращает Entry, если найдено и D == 0, иначе None."""
        index = self.hash1(id)
        step = self.hash2(id)
        attempts = 0

        while attempts < self.size:
            if self.table[index] is None:
                return None
            if self.table[index].id == id and self.table[index].D == 0:
                return self.table[index]
            attempts += 1
            index = (self.hash1(id) + attempts * step) % self.size
        return None

    def delete(self, id):
        """Удаление записи по ключу id. Устанавливает D = 1, U = 0. Возвращает True, если удалено."""
        index = self.hash1(id)
        step = self.hash2(id)
        attempts = 0

        while attempts < self.size:
            if self.table[index] is None:
                return False
            if self.table[index].id == id and self.table[index].D == 0:
                self.table[index].D = 1
                self.table[index].U = 0
                return True
            attempts += 1
            index = (self.hash1(id) + attempts * step) % self.size
        return False

    def get_table_size(self):
        return self.size

    def get_num_entries(self):
        # Считаем только записи с D == 0 (не удаленные)
        return sum(1 for slot in self.table if slot is not None and slot.D == 0)

    def get_num_collisions(self):
        return self.collisions

    def get_num_slots_with_probing(self):
        # Считаем слоты, где потребовалось более 0 шагов пробинга
        return sum(1 for steps in self.probing_steps.values() if steps > 0)

    def get_load_factor(self):
        return self.get_num_entries() / self.size

    def display(self):
        for i, entry in enumerate(self.table):
            if entry and entry.D == 0:  # Показываем только неудаленные записи
                steps = self.probing_steps.get(i, 0)
                print(f"Index {i}: {entry.id} (пробинг: {steps} шагов, C={entry.C}, U={entry.U}, D={entry.D})")

# Данные: 20 грамматических терминов
terms = [
    ("существительное", "Часть речи, обозначающая предмет"),
    ("глагол", "Часть речи, обозначающая действие"),
    ("прилагательное", "Часть речи, обозначающая признак предмета"),
    ("наречие", "Часть речи, обозначающая признак действия"),
    ("местоимение", "Часть речи, заменяющая существительное"),
    ("предлог", "Служебная часть речи для отношений"),
    ("союз", "Служебная часть речи для соединения"),
    ("частица", "Служебная часть речи для отношения говорящего"),
    ("инфинитив", "Неличная форма глагола"),
    ("деепричастие", "Форма глагола как признак действия"),
    ("причастие", "Форма глагола как признак предмета"),
    ("морфология", "Раздел грамматики о словоизменении"),
    ("синтаксис", "Раздел грамматики о предложениях"),
    ("фонетика", "Наука о звуках языка"),
    ("лексика", "Словарный состав языка"),
    ("ортография", "Правила правописания"),
    ("пунктуация", "Правила знаков препинания"),
    ("этимология", "Изучение происхождения слов"),
    ("семантика", "Учение о значении слов"),
    ("прагматика", "Учение о языке в действии"),
]

# Основная программа
if __name__ == "__main__":
    ht = HashTable(size=20)
    for id, pi in terms:
        try:
            ht.insert(id, pi)
        except Exception as e:
            print(e)
            break
    else:
        print(f"Размер таблицы: {ht.get_table_size()}")
        print(f"Количество записей: {ht.get_num_entries()}")
        print(f"Количество коллизий: {ht.get_num_collisions()}")
        print(f"Количество слотов с пробингом: {ht.get_num_slots_with_probing()}")
        print(f"Коэффициент заполнения: {ht.get_load_factor():.2f}")
        print("\nСодержимое таблицы:")
        ht.display()