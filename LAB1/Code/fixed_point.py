class FixedPoint:
    BIT_LENGTH = 32
    SIGN_BIT = 31

    @staticmethod
    def decimal_to_binary(decimal):

        if decimal == 0:
            return '0' * FixedPoint.BIT_LENGTH
        sign = '0' if decimal >= 0 else '1'
        binary = bin(abs(decimal))[2:]
        binary = binary.zfill(FixedPoint.BIT_LENGTH - 1)
        if len(binary) > FixedPoint.BIT_LENGTH - 1:
            raise OverflowError("Число слишком большое для представления")
        return sign + binary

    @staticmethod
    def get_direct_code(decimal):

        return FixedPoint.decimal_to_binary(decimal)

    @staticmethod
    def get_inverse_code(decimal):

        binary = FixedPoint.get_direct_code(decimal)
        if binary[0] == '0':
            return binary

        inverted = ''.join('1' if bit == '0' else '0' for bit in binary[1:])
        return binary[0] + inverted

    @staticmethod
    def get_complement_code(decimal):

        if decimal >= 0:
            return FixedPoint.get_direct_code(decimal)
        inverse = FixedPoint.get_inverse_code(decimal)

        complement = bin(int(inverse, 2) + 1)[2:].zfill(FixedPoint.BIT_LENGTH)
        return complement

    @staticmethod
    def add_complement(a, b):

        a_comp = FixedPoint.get_complement_code(a)
        b_comp = FixedPoint.get_complement_code(b)
        sum_binary = bin(int(a_comp, 2) + int(b_comp, 2))[2:].zfill(FixedPoint.BIT_LENGTH)
        if len(sum_binary) > FixedPoint.BIT_LENGTH:
            sum_binary = sum_binary[-FixedPoint.BIT_LENGTH:]
        return sum_binary

    @staticmethod
    def subtract_complement(a, b):

        return FixedPoint.add_complement(a, -b)

    @staticmethod
    def multiply_direct(a, b):

        sign = '0' if (a >= 0 and b >= 0) or (a < 0 and b < 0) else '1'
        a_bin = FixedPoint.get_direct_code(abs(a))[1:]
        b_bin = FixedPoint.get_direct_code(abs(b))[1:]
        product = int(a_bin, 2) * int(b_bin, 2)
        product_bin = bin(product)[2:].zfill(FixedPoint.BIT_LENGTH - 1)
        if len(product_bin) > FixedPoint.BIT_LENGTH - 1:
            raise OverflowError("Произведение слишком большое для представления")
        return sign + product_bin

    @staticmethod
    def divide_direct(a, b, precision=5):

        if b == 0:
            raise ValueError("Деление на ноль")
        sign = '0' if (a >= 0 and b >= 0) or (a < 0 and b < 0) else '1'
        a_abs = abs(a)
        b_abs = abs(b)
        quotient = a_abs // b_abs
        remainder = a_abs % b_abs
        quotient_bin = bin(quotient)[2:].zfill(FixedPoint.BIT_LENGTH - 1)
        fractional = ''
        for _ in range(precision):
            remainder *= 2
            bit = remainder // b_abs
            remainder = remainder % b_abs
            fractional += str(bit)
        return sign + quotient_bin + '.' + fractional

    @staticmethod
    def binary_to_decimal(binary):

        if binary[0] == '0':
            return int(binary[1:], 2)
        else:
            inverted = ''.join('1' if bit == '0' else '0' for bit in binary[1:])
            return -(int(inverted, 2) + 1)
