#!/usr/bin/env/python3
# -*- coding: UTF-8 -*-
"""
This module contains the test suites for the whole Polynomial package
Use it by typing: python3 -m unittest test_polynesian.py
"""
import builtins
import unittest
import pathlib
import math
import os.path
import sys

import numpy as np

# sys.path.append('../src')

from polynomial_to_py import *
import polynomial_to_py as PolynomialToPython
from zero_to_py import *
import zero_to_py as ZerosToPython
from utils import *

file_path = pathlib.PurePath(__file__)


class TestPolynomialToPython(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zeros = file_path.with_name('zeros')
        cls.polynomial = file_path.with_name('polynomial')
        cls.python = file_path.with_name('python')
        cls.old_open = open
        cls.to_remove = []

        def wrapped(*args, **kwargs):
            if len(args) >= 2 and args[1] == 'w':
                cls.to_remove.append(args[0])
            return cls.old_open(*args, **kwargs)
        builtins.open = wrapped

    def tearDown(self):
        [os.remove(i) for i in self.to_remove]
        self.to_remove.clear()

    def test_module_only_input_file(self):
        path = PolynomialToPython.__file__
        PolynomialToPython.main([path, self.polynomial])

        new_file = self.polynomial.with_suffix('.py')
        self.assertTrue(os.path.isfile(new_file))

        with open(new_file) as fresh, open(self.python) as original:
            self.assertEqual(fresh.read(), original.read())

    def test_module_input_output(self):
        path = PolynomialToPython.__file__
        new_file = self.polynomial.with_name('xxx')
        PolynomialToPython.main([path, self.polynomial, new_file])

        self.assertTrue(os.path.isfile(new_file))

        with open(new_file) as fresh, open(self.python) as original:
            self.assertEqual(fresh.read(), original.read())

    def test_translate_to_python(self):
        # Hello, World!
        code = [(72+1j), 1j, (29+1j), 1j, (7+1j), 1j, 1j,
                (3+1j), 1j, (67+2j), 1j, (12+2j), 1j, (55+1j),
                1j, (24+1j), 1j, (3+1j), 1j, (6+2j), 1j,
                (8+2j), 1j, (67+2j), 1j]

        python_code = """ACC = 0
    ACC += 72
    print(chr(ACC), end="")
    ACC += 29
    print(chr(ACC), end="")
    ACC += 7
    print(chr(ACC), end="")
    print(chr(ACC), end="")
    ACC += 3
    print(chr(ACC), end="")
    ACC -= 67
    print(chr(ACC), end="")
    ACC -= 12
    print(chr(ACC), end="")
    ACC += 55
    print(chr(ACC), end="")
    ACC += 24
    print(chr(ACC), end="")
    ACC += 3
    print(chr(ACC), end="")
    ACC -= 6
    print(chr(ACC), end="")
    ACC -= 8
    print(chr(ACC), end="")
    ACC -= 67
    print(chr(ACC), end="")
    """
        python_code = '\n'.join(i.lstrip() for i in python_code.splitlines())
        assert translate_to_python(code) == python_code

    def test_get_actual_zeros(self):
        roots = [4826809, 243, 25j, 1+11j, 7j, 1+2j]
        commands = [1+1j, 5, 2j, 1j, 1+1j, 6]
        assert get_actual_zeros(roots) == commands

    def test_get_exponent(self):
        #assert math.isclose(get_exponent((-3) ** 5, 3), 5)
        assert math.isclose(get_exponent(2 ** 1, 2), 1)
        assert math.isclose(get_exponent(2 ** 4, 2), 4)
        assert math.isclose(get_exponent(2 ** 1.2, 2), 1.2)
        assert math.isclose(get_exponent(7 ** 11, 7), 11)

    def test_get_roots(self):
        # f(x) = (x - 1)(x + 7) = x2 + 7x - x - 7 = x2 + 6x - 7
        # roots: 1, -7
        np.testing.assert_array_almost_equal(
            np.sort(get_roots([1, 6, -7])),
            np.sort([-7, 1]))
        # f(x) = (x - 1)(x - i)(x + i)(x - 2) = x4 -3x3 +3x2 -3x +2
        # roots: 1, 2, -i, i
        np.testing.assert_array_almost_equal(
            np.sort(np.roots([1, -3, 3, -3, 2])),
            np.sort([1, 2, -1j, 1j]))

    def test_parse_polynomial(self):
        pol = 'f(x) = x^4 - 3x^2 + 2'
        res = [(1, 4), (-3, 2), (2, 0)]
        assert parse_polynomial(pol) == res

        pol = 'f(x) = + x^6 - 3x^2'
        res = [(1, 6), (-3, 2)]
        assert parse_polynomial(pol) == res

        pol = 'f(x) = - 171x^2 - 12x - 0'
        res = [(-171, 2), (-12, 1), (0, 0)]
        assert parse_polynomial(pol) == res

    def test_fix_coefficients(self):
        pairs = [(12, 4), (7, 2), (-2, 0)]
        assert get_coefficients(pairs) == [12, 0, 7, 0, -2]

        pairs = [(12, 100), (-2, 0)]
        assert get_coefficients(pairs) == [12] + [0 for _ in range(99)] + [-2]

        pairs = [(5, 3), (5, 2), (5, 1), (5, 0)]
        assert get_coefficients(pairs) == [5, 5, 5, 5]

        pairs = [(0, 1), (1, 0)]
        assert get_coefficients(pairs) == [0, 1]

    def test_parse_monomial(self):
        """
        Allowed forms:
            {sgn}{int}x^{power}: +13x^6
                 {int}x^{power}: 13x^6   then {sgn} == +
            {sgn}{int}x:         +13x    then {power} == 1
                 {int}x:         13x     then ({sgn} == +) & ({power} == 1)
            {sgn}{int}:          +13     then {power} == 0
                 {int}:          13      then ({sgn} == +) & ({power} == 0)
        """
        # fixme: handle no coefficient given
        assert parse_monomial('+12x^6') == (12, 6)
        assert parse_monomial('-12x^6') == (-12, 6)
        assert parse_monomial('12x^6') == (12, 6)
        # assert parse_monomial('x^6') == (1, 6)

        assert parse_monomial('+12x') == (12, 1)
        assert parse_monomial('-12x') == (-12, 1)
        assert parse_monomial('12x') == (12, 1)
        # assert parse_monomial('x') == (1, 1)

        assert parse_monomial('+12') == (12, 0)
        assert parse_monomial('-12') == (-12, 0)
        assert parse_monomial('12') == (12, 0)

        assert parse_monomial('+1x^1') == (1, 1)
        assert parse_monomial('-1x^0') == (-1, 0)


class TestZerosToPolynomial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zeros = file_path.with_name('zeros')
        cls.polynomial = file_path.with_name('polynomial')
        cls.python = file_path.with_name('python')
        cls.old_open = open
        cls.to_remove = []

        def wrapped(*args, **kwargs):
            if len(args) >= 2 and args[1] == 'w':
                cls.to_remove.append(args[0])
            return cls.old_open(*args, **kwargs)
        builtins.open = wrapped

    def tearDown(self):
        [os.remove(i) for i in self.to_remove]
        self.to_remove.clear()

    def test_module_only_input_file(self):
        path = ZerosToPython.__file__
        ZerosToPython.main([path, self.zeros])

        new_file = self.zeros.with_suffix('.py')
        self.assertTrue(os.path.isfile(new_file))

        with open(new_file) as fresh, open(self.python) as original:
            self.assertEqual(fresh.read(), original.read())

    def test_module_input_output(self):
        path = ZerosToPython.__file__
        new_file = self.zeros.with_name('xxx')
        ZerosToPython.main([path, self.zeros, new_file])

        self.assertTrue(os.path.isfile(new_file))
        with open(new_file) as fresh, open(self.python) as original:
            self.assertEqual(fresh.read(), original.read())

    def test_module_only_input_file_pol(self):
        path = ZerosToPython.__file__
        ZerosToPython.main([path, self.zeros, '-pol'])

        new_file = self.zeros.with_suffix('.py')
        self.assertTrue(os.path.isfile(new_file))
        with open(new_file) as fresh, open(self.python) as original:
            self.assertEqual(fresh.read(), original.read())

        pol_file = self.zeros.with_suffix('.pol')
        self.assertTrue(os.path.isfile(pol_file))
        with open(pol_file) as fresh, open(self.polynomial) as original:
            self.assertEqual(fresh.read(), original.read())

    def test_module_input_output_pol(self):
        new_file = self.zeros.with_name('xxx')
        path = ZerosToPython.__file__
        ZerosToPython.main([path, self.zeros, new_file, '-pol'])

        self.assertTrue(os.path.isfile(new_file))
        with open(new_file) as fresh, open(self.python) as original:
            self.assertEqual(fresh.read(), original.read())

        pol_file = new_file.with_suffix('.pol')
        self.assertTrue(os.path.isfile(pol_file))
        with open(pol_file) as fresh, open(self.polynomial) as original:
            self.assertEqual(fresh.read(), original.read())

    def test_add_primes(self):
        # 1, i, 2 -> 2^1, 3^1i, 5^2 -> 2^1, 3^1i, -3^1i, 5^2
        commands = list(map(complex, [1, 1j, 2]))
        result = list(map(complex, [2, 3j, 25]))
        assert add_primes(commands) == result

    def test_polynomial_to_string(self):
        polynomial = np.poly1d([1, 2, 3, -4])
        string = 'f(x) = x^3 + 2x^2 + 3x - 4'
        assert polynomial_to_string(polynomial.c) == string

    def test_parse_source(self):
        source = """
                # This module implements a CAT program
                # pseudo-code:
                # <ACC := 0>
                # ACC++             {1+1j}
                # while ACC > 0{    {5}
                #   ACC = getchar() {2j}
                #   putchar(ACC)    {1j}
                #   ACC++           {1+1j}
                # }                 {6}

                1+1j
                5
                2j
                1j
                1+1j
                6
                """
        assert parse_source(source) == [1+1j, 5, 2j, 1j, 1+1j, 6]

    def test_add_conjugates(self):
        # Cat program:
        commands = 1+1j, 5, 2j, 1j, 1+1j, 6
        result = 1+1j, 1-1j, 5, 2j, -2j, 1j, -1j, 1+1j, 1-1j, 6
        commands = list(map(complex, commands))
        result = list(map(complex, result))
        assert add_conjugates(commands) == result


class TestUtilities(unittest.TestCase):
    def test_generate_primes(self):
        prime_gen = generate_primes()
        primes = [next(prime_gen) for _ in range(20)]
        ok = [2,    3,  5,  7,  11, 13, 17, 19, 23, 29,
              31,   37, 41, 43, 47, 53, 59, 61, 67, 71]
        assert primes[:20] == ok

        # -20 cause 20 first elements were checked earlier and we
        # didn't change the generator, so it wouldn't work
        for _ in range(1000 - 20):
            prime = next(prime_gen)

        assert prime == 7919


if __name__ == '__main__':
    unittest.main()
