from logical_operations import *

class KarnoSKNF(LogicalOperations):
    def __init__(self, inputs, labels):
        super().__init__()
        self.inputs = inputs
        self.labels = labels
        self.var_count = len(labels)
        self.order = [0, 1, 3, 2]

    def make_map_5(self):
        grid = [[[0] * 4 for _ in range(4)] for _ in range(2)]
        zeros = []
        for row in self.inputs:
            q, r, s, t, p = row[0]
            block = p
            row_i = self.order.index((q << 1) | r)
            col_i = self.order.index((s << 1) | t)
            grid[block][row_i][col_i] = row[1]
            if row[1] == 0:
                zeros.append((q, r, s, t, p))
        return grid, zeros

    def make_map_6(self):
        grid = [[[[0] * 4 for _ in range(4)] for _ in range(4)] for _ in range(4)]
        zeros = []
        for row in self.inputs:
            q, r, s, t, o, p = row[0]
            block_i = self.order.index((o << 1) | p)
            row_i = self.order.index((q << 1) | r)
            col_i = self.order.index((s << 1) | t)
            grid[block_i][row_i][col_i] = row[1]
            if row[1] == 0:
                zeros.append((q, r, s, t, o, p))
        return grid, zeros

    def show_map_5(self, grid):
        print("Карта Карно:")
        print("ab\\cde   000  001  011  010   110  111  101  100")
        for r in range(4):
            label = format(self.order[r], '02b')
            line = f"{label}       "
            part1 = [str(grid[0][r][c]) for c in range(4)]
            part2 = [str(grid[1][r][c]) for c in range(4)]
            line += "  ".join(f"{v:>3}" for v in part1) + "  " + "  ".join(f"{v:>3}" for v in part2)
            print(line)

    def show_map_6(self, grid):
        print("Карта Карно:")
        print("ef ab\\cd   0000  0001  0011  0010   0100  0101  0111  0110   1100  1101  1111  1110   1000  1001  1011  1010")
        for r in range(4):
            label = format(self.order[r], '02b')
            line = f"  {label}       "
            for b in range(4):
                vals = [str(grid[b][r][c]) for c in range(4)]
                line += "  ".join(f"{v:>3}" for v in vals) + "  "
            print(line)

    def get_cells_5_6(self, grid, b_start, b_size, r_start, r_size, c_start, c_size, b_max, r_max, c_max):
        cells = set()
        all_zeros = True
        for db in range(b_size):
            b = (b_start + db) % b_max
            for dr in range(r_size):
                r = (r_start + dr) % r_max
                for dc in range(c_size):
                    c = (c_start + dc) % c_max
                    cells.add((b, r, c))
                    if grid[b][r][c] == 1:
                        all_zeros = False
                        break
                if not all_zeros:
                    break
            if not all_zeros:
                break
        return cells, all_zeros

    def make_pattern_5_6(self, cells):
        pattern = ['-'] * self.var_count
        values = []
        for b, r, c in cells:
            qr = self.order[r]
            st = self.order[c]
            q = (qr >> 1) & 1
            r_val = qr & 1
            s = (st >> 1) & 1
            t = st & 1
            if self.var_count == 5:
                p = b
                values.append([q, r_val, s, t, p])
            else:
                op = self.order[b]
                o = (op >> 1) & 1
                p = op & 1
                values.append([q, r_val, s, t, o, p])
        for i in range(self.var_count):
            unique_vals = {val[i] for val in values}
            if len(unique_vals) == 1:
                pattern[i] = str(unique_vals.pop())
        return pattern

    def find_groups_5_6(self, grid, b_max):
        groups = []
        for b_size in [1, 2, 4][:b_max + 1]:
            for b_start in range(b_max):
                if b_size > 1 and b_start != 0:
                    continue
                for r_size in [1, 2, 4]:
                    for c_size in [1, 2, 4]:
                        for r_start in range(4):
                            for c_start in range(4):
                                cells, all_zeros = self.get_cells_5_6(grid, b_start, b_size, r_start, r_size, c_start, c_size, b_max, 4, 4)
                                if all_zeros and cells:
                                    pattern = self.make_pattern_5_6(cells)
                                    groups.append((cells, pattern))
        return groups

    def get_biggest_groups(self, groups):
        biggest = []
        for i, (cells_i, pat_i) in enumerate(groups):
            is_biggest = True
            for j, (cells_j, pat_j) in enumerate(groups):
                if i != j and cells_i <= cells_j and cells_i != cells_j:
                    is_biggest = False
                    break
            if is_biggest:
                biggest.append((cells_i, pat_i))
        return biggest

    def get_zeros_pos_5_6(self, zeros):
        positions = []
        for p in zeros:
            if self.var_count == 5:
                q, r, s, t, p = p
                positions.append((p, self.order.index((q << 1) | r), self.order.index((s << 1) | t)))
            else:
                q, r, s, t, o, p = p
                positions.append((self.order.index((o << 1) | p), self.order.index((q << 1) | r), self.order.index((s << 1) | t)))
        return positions

    def choose_groups(self, big_groups, zeros_pos, verbose):
        chosen = []
        to_cover = set(range(len(zeros_pos)))
        for i, pos in enumerate(zeros_pos):
            covers = [idx for idx, (cells, _) in enumerate(big_groups) if pos in cells]
            if len(covers) == 1:
                idx = covers[0]
                if idx not in chosen:
                    chosen.append(idx)
                    for j, other_pos in enumerate(zeros_pos):
                        if other_pos in big_groups[idx][0]:
                            to_cover.discard(j)
        while to_cover:
            best_idx = None
            best_covered = set()
            for idx, (cells, _) in enumerate(big_groups):
                if idx in chosen:
                    continue
                covered = {j for j in to_cover if zeros_pos[j] in cells}
                if len(covered) > len(best_covered):
                    best_covered = covered
                    best_idx = idx
            if best_idx is None:
                if verbose:
                    print("Не удалось покрыть все нули!")
                break
            chosen.append(best_idx)
            to_cover -= best_covered
        return chosen

    def make_result(self, chosen, big_groups):
        terms = set()
        for idx in chosen:
            pattern = big_groups[idx][1]
            term = []
            for i, val in enumerate(pattern):
                if val == '0':
                    term.append(self.labels[i])
                elif val == '1':
                    term.append(f'!{self.labels[i]}')
            if term:
                term_str = term[0] if len(term) == 1 else '(' + ' ∨ '.join(sorted(term)) + ')'
                terms.add(term_str)
        terms = sorted(terms)
        result = "1" if not terms else " ∧ ".join(terms)
        return result

    def karno_sknf(self, verbose=True):
        if self.var_count not in [5, 6]:
            return "Этот метод не поддерживает такое количество переменных."
        if verbose:
            print(f"\nСокращаем СКНФ с помощью карты Карно ({self.var_count} переменных):")
        if self.var_count == 5:
            grid, zeros = self.make_map_5()
            if verbose:
                self.show_map_5(grid)
            b_max = 2
        else:
            grid, zeros = self.make_map_6()
            if verbose:
                self.show_map_6(grid)
            b_max = 4
        if not zeros:
            return "1"
        groups = self.find_groups_5_6(grid, b_max)
        big_groups = self.get_biggest_groups(groups)
        zeros_pos = self.get_zeros_pos_5_6(zeros)
        chosen = self.choose_groups(big_groups, zeros_pos, verbose)
        return self.make_result(chosen, big_groups)

    def make_map_1_to_4(self):
        zeros = []
        if self.var_count == 1:
            grid = [0] * 2
            for row in self.inputs:
                x, = row[0]
                grid[x] = row[1]
                if row[1] == 0:
                    zeros.append((x,))
        elif self.var_count == 2:
            grid = [[0] * 2 for _ in range(2)]
            for row in self.inputs:
                x, y = row[0]
                grid[x][y] = row[1]
                if row[1] == 0:
                    zeros.append((x, y))
        elif self.var_count == 3:
            grid = [[0] * 4 for _ in range(2)]
            for row in self.inputs:
                x, y, z = row[0]
                grid[x][self.order.index((y << 1) | z)] = row[1]
                if row[1] == 0:
                    zeros.append((x, y, z))
        else:
            grid = [[0] * 4 for _ in range(4)]
            for row in self.inputs:
                x, y, z, w = row[0]
                grid[self.order.index((x << 1) | y)][self.order.index((z << 1) | w)] = row[1]
                if row[1] == 0:
                    zeros.append((x, y, z, w))
        return grid, zeros

    def show_map_1_to_4(self, grid):
        print("Карта Карно:")
        if self.var_count == 1:
            print("a    0  1")
            print("     " + "  ".join(str(grid[i]) for i in range(2)))
        elif self.var_count == 2:
            print("a\\b   0  1")
            for r in range(2):
                print(f"{r}     " + "  ".join(str(grid[r][c]) for c in range(2)))
        elif self.var_count == 3:
            print("a\\bc   00  01  11  10")
            for r in range(2):
                print(f"{r}     " + "  ".join(f"{grid[r][c]:>3}" for c in range(4)))
        else:
            print("ab\\cd   00  01  11  10")
            for r in range(4):
                label = format(self.order[r], '02b')
                print(f"{label}     " + "  ".join(f"{grid[r][c]:>3}" for c in range(4)))

    def get_cells_1_to_4(self, grid, r_start, r_size, c_start, c_size, r_max, c_max):
        cells = set()
        all_zeros = True
        for dr in range(r_size):
            r = (r_start + dr) % r_max
            for dc in range(c_size):
                c = (c_start + dc) % c_max
                cells.add((r, c))
                val = grid[c] if self.var_count == 1 else grid[r][c]
                if val == 1:
                    all_zeros = False
                    break
            if not all_zeros:
                break
        return cells, all_zeros

    def make_pattern_1_to_4(self, cells):
        pattern = ['-'] * self.var_count
        values = []
        for r, c in cells:
            if self.var_count == 1:
                values.append([c])
            elif self.var_count == 2:
                values.append([r, c])
            elif self.var_count == 3:
                yz = self.order[c]
                values.append([r, (yz >> 1) & 1, yz & 1])
            else:
                xy = self.order[r]
                zw = self.order[c]
                values.append([(xy >> 1) & 1, xy & 1, (zw >> 1) & 1, zw & 1])
        for i in range(self.var_count):
            unique_vals = {val[i] for val in values}
            if len(unique_vals) == 1:
                pattern[i] = str(unique_vals.pop())
        return pattern

    def find_groups_1_to_4(self, grid, r_max, c_max):
        groups = []
        row_sizes = [1, 2] if r_max <= 2 else [1, 2, 4]
        col_sizes = [1, 2] if c_max == 2 else [1, 2, 4]
        for r_size in row_sizes:
            for c_size in col_sizes:
                for r_start in range(r_max):
                    for c_start in range(c_max):
                        cells, all_zeros = self.get_cells_1_to_4(grid, r_start, r_size, c_start, c_size, r_max, c_max)
                        if all_zeros and cells:
                            pattern = self.make_pattern_1_to_4(cells)
                            groups.append((cells, pattern))
        return groups

    def get_zeros_pos_1_to_4(self, zeros):
        positions = []
        for p in zeros:
            if self.var_count == 1:
                positions.append((0, p[0]))
            elif self.var_count == 2:
                positions.append((p[0], p[1]))
            elif self.var_count == 3:
                positions.append((p[0], self.order.index((p[1] << 1) | p[2])))
            else:
                positions.append((self.order.index((p[0] << 1) | p[1]), self.order.index((p[2] << 1) | p[3])))
        return positions

    def karno_1_to_4_sknf(self, verbose=True):
        if self.var_count not in [1, 2, 3, 4]:
            return "Этот метод работает только для 1–4 переменных."
        if verbose:
            print(f"\nУпрощаем СКНФ через карту Карно ({self.var_count} переменных):")
        grid, zeros = self.make_map_1_to_4()
        if verbose:
            self.show_map_1_to_4(grid)
        if not zeros:
            return "1"
        r_max = 1 if self.var_count == 1 else 2 if self.var_count in [2, 3] else 4
        c_max = 2 if self.var_count in [1, 2] else 4
        groups = self.find_groups_1_to_4(grid, r_max, c_max)
        big_groups = self.get_biggest_groups(groups)
        zeros_pos = self.get_zeros_pos_1_to_4(zeros)
        chosen = self.choose_groups(big_groups, zeros_pos, verbose)
        return self.make_result(chosen, big_groups)