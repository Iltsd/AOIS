class FloatingPoint:
    SIGN_BIT = 31
    EXPONENT_BITS = 8
    MANTISSA_BITS = 23
    EXPONENT_BIAS = 127

    @staticmethod
    def decimal_to_binary(decimal):

        if decimal == 0:
            return '0' * 32
        sign = '0' if decimal >= 0 else '1'
        decimal = abs(decimal)
        integer_part = int(decimal)
        fractional_part = decimal - integer_part
        integer_bin = bin(integer_part)[2:]
        fractional_bin = ''

        while fractional_part > 0 and len(fractional_bin) < 32:
            fractional_part *= 2
            bit = int(fractional_part)
            fractional_bin += str(bit)
            fractional_part -= bit
        if integer_bin == '0':
            exponent = -1
            while fractional_bin and fractional_bin[0] == '0':
                fractional_bin = fractional_bin[1:]
                exponent -= 1
            mantissa = fractional_bin[1:24]
            exponent += 127
        else:
            exponent = len(integer_bin) - 1
            mantissa = (integer_bin[1:] + fractional_bin).ljust(23, '0')[:23]
            exponent += 127
        exponent_bin = bin(exponent)[2:].zfill(8)
        return sign + exponent_bin + mantissa

    @staticmethod
    def add(a, b):

        a_bin = FloatingPoint.decimal_to_binary(a)
        b_bin = FloatingPoint.decimal_to_binary(b)
        a_sign, a_exp, a_mant = a_bin[0], a_bin[1:9], '1' + a_bin[9:]
        b_sign, b_exp, b_mant = b_bin[0], b_bin[1:9], '1' + b_bin[9:]
        a_exp_int = int(a_exp, 2)
        b_exp_int = int(b_exp, 2)

        if a_exp_int > b_exp_int:
            diff = a_exp_int - b_exp_int
            b_mant = '0' * diff + b_mant[:-diff]
            exp = a_exp
        else:
            diff = b_exp_int - a_exp_int
            a_mant = '0' * diff + a_mant[:-diff]
            exp = b_exp

        a_mant_int = int(a_mant, 2)
        b_mant_int = int(b_mant, 2)
        sum_mant = a_mant_int + b_mant_int
        sum_mant_bin = bin(sum_mant)[2:]

        if len(sum_mant_bin) > 24:
            sum_mant_bin = sum_mant_bin[1:]
            exp = bin(int(exp, 2) + 1)[2:].zfill(8)
        else:
            sum_mant_bin = sum_mant_bin.zfill(24)
        mantissa = sum_mant_bin[1:24]
        return a_sign + exp + mantissa

    @staticmethod
    def binary_to_decimal(binary):

        sign = -1 if binary[0] == '1' else 1
        exp = int(binary[1:9], 2) - 127
        mantissa = 1 + sum(int(binary[9 + i]) * 2 ** -(i + 1) for i in range(23))
        return sign * mantissa * 2 ** exp
