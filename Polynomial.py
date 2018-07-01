#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
This module implements an interpreter for the Polynomial esoteric language
Save your correct Polynomial code as a string variable and apply convert()
function on it
"""
from collections import namedtuple
from typing import List, Tuple, Generator, Union
import math

import numpy as np

Operation = namedtuple('Operation', ['operation', 'operand'])


def get_roots(coefficients: List[int], tol=1e-11) -> np.ndarray:
    """
    Take the polynomial to get roots of in the form of its coefficients.
    Use NumPy to get the roots, then round those which are almost-zero
    and discard those which imaginary part is negative (complex conjugates;
    it's NOP)

    :param coefficients: coefficients of the polynomial to get roots of
    :param tol: tolerance value when rounding to 0
    :return: cleared array of roots of the given polynomial
    """
    roots = np.roots(coefficients)
    roots.real[abs(roots.real) < tol] = 0
    if roots.imag.any():
        roots.imag[abs(roots.imag) < tol] = 0
    roots = roots[roots.imag >= 0]  # if b in (a+bi) < 0, it's a NOP for us
    return roots


def test_get_roots():
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


def parse_monomial(monomial: str) -> Tuple[int, int]:
    """
    Convert a single monomial as a string to (coefficient, order of x)

    Allowed forms:
        {sgn}{int}x^{power}: +13x^6
             {int}x^{power}: 13x^6
                  x^{power}: x^6
        {sgn}{int}x:         +13x
             {int}x:         13x
                  x:         x
        {sgn}{int}:          +13
             {int}:          13

        when sgn isn't given, it's assumed to be '+'
        when int isn't given, it's assumed to be 1
        when power isn't given, it's assumed to be:
                1 if x is present: 13x -> power == 1
                0 otherwise:       13  -> power == 0

    Example:
    '+x^10'         ->  (1,          10)
    '-4827056x^9'   ->  (-4827056,   9)
    """

    # +x^n -> +1x^n
    if 'x' in monomial:
        ind = monomial.index('x')
        if not monomial[ind-1].isdigit():
            monomial = monomial[0] + '1' + monomial[1:]

    if 'x' in monomial:
        coefficient_end = monomial.index('x')
        coefficient = int(monomial[:coefficient_end] or '1')

    if 'x^' in monomial:
        coefficient_end = monomial.index('x')
        power_beginning = coefficient_end + 2
        power = int(monomial[power_beginning:])
    elif 'x' in monomial and '^' not in monomial:
        coefficient_end = monomial.index('x')
        power = 1
    elif 'x' not in monomial and '^' not in monomial:
        coefficient_end = len(monomial)
        power = 0
    else:
        raise Exception('Wrong format')

    # if there's no coefficient given, this string it's '', so we make or '1'
    # cause default coefficient is 1
    coefficient = int(monomial[:coefficient_end] or '1')
    return coefficient, power


def test_parse_monomial():
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


def parse_polynomial(polynomial: str) -> List[Tuple[int, int]]:
    """
    Take the whole polynomial as a string and perform parsing on its
    each element. Return a sequence of [(coefficient, power of x)];
    powers will be in descending order, but some of them can be missing;

    Example:
    f(x) = x^4 - 3x^2 + 2
    -> [(1, 4), (-3, 2), (2, 0)]
    """
    # cut 'f(x)' and '='
    polynomial = polynomial.split()[2:]

    # first element can go without sign, if it's a +
    if polynomial[0] not in {'+', '-'}:
        polynomial = ['+'] + polynomial

    # merge monomials with their signs
    iterator = iter(polynomial)
    merged = [sgn + monomial for (sgn, monomial) in zip(iterator, iterator)]
    return [parse_monomial(i) for i in merged]


def test_parse_polynomial():
    pol = 'f(x) = x^4 - 3x^2 + 2'
    res = [(1, 4), (-3, 2), (2, 0)]
    assert parse_polynomial(pol) == res

    pol = 'f(x) = + x^6 - 3x^2'
    res = [(1, 6), (-3, 2)]
    assert parse_polynomial(pol) == res

    pol = 'f(x) = - 171x^2 - 12x - 0'
    res = [(-171, 2), (-12, 1), (0, 0)]
    assert parse_polynomial(pol) == res


def get_coefficients(pairs: List[Tuple[int, int]]) -> List[int]:
    """
    Take a sequence like [(coefficient, order)] and add missing elements
    Then return only coefficients - powers are ok, so we do not need them

    Example:
        when f(x) = 12x^4 + 7x^2 - 2, then:
        pairs = [(12, 4), (7, 2), (-2, 0)]
        return: [12, 0, 7, 0, -2]
    """
    coefficients, powers = zip(*pairs)
    order = powers[0]
    coefficients, powers = list(coefficients), set(powers)
    missing = set(range(order)) - powers
    [coefficients.insert(len(coefficients) - i, 0) for i in missing]
    return coefficients


def test_fix_coefficients():
    pairs = [(12, 4), (7, 2), (-2, 0)]
    assert get_coefficients(pairs) == [12, 0, 7, 0, -2]

    pairs = [(12, 100), (-2, 0)]
    assert get_coefficients(pairs) == [12] + [0 for _ in range(99)] + [-2]

    pairs = [(5, 3), (5, 2), (5, 1), (5, 0)]
    assert get_coefficients(pairs) == [5, 5, 5, 5]

    pairs = [(0, 1), (1, 0)]
    assert get_coefficients(pairs) == [0, 1]


def generate_primes() -> Generator[int, None, None]:
    """
    Yield consecutive prime numbers

    :yields: next prime number
    """
    primes = {2}
    yield 2

    current = 3
    while True:
        if all(current % i for i in primes):
            # nothing has divided our prime - it is a prime
            primes.add(current)
            yield current
        current += 2  # omit even numbers


def test_generate_primes():
    prime_gen = generate_primes()
    primes = [next(prime_gen) for _ in range(20)]
    ok = [2,    3,  5,  7,  11, 13, 17, 19, 23, 29,
          31,   37, 41, 43, 47, 53, 59, 61, 67, 71]
    assert primes[:20] == ok

    for _ in range(1000 - 20):
        prime = next(prime_gen)
    assert prime == 7919


def get_exponent(root: complex, prime: int) -> float:
    """
    Take a complex number and the prime number that potentially was rose to some
    power following the rules of the Polynomial language and return that power

    Rules of exponentiation:
        For real numbers: x -> p^x
        For complex numbers: (a + bi) -> (a + (p^b)i)
        where:
            x, a, b are some real numbers
            p is some prime number

    :param prime: the prime number that potentially was rose to some power
    :param root: the complex number that we're currently deciphering
    :return: exponent;
    """
    if root.imag:  # a+bi -> a + (p^b)i
        exp = math.log(root.imag) / math.log(prime)
    elif not root.imag and root.real < 0:  # a -> p^a
        # -root.real to get abs and -np.log to save the orig. sign
        exp = -math.log(-root.real) / math.log(prime)
    elif not root.imag and root.real >= 0:
        exp = math.log(root.real) / math.log(prime)
    else:
        raise Exception()
    return exp


def test_get_exponent():
    assert math.isclose(get_exponent(2 ** 1, 2), 1)
    assert math.isclose(get_exponent(2 ** 4, 2), 4)
    assert math.isclose(get_exponent(2 ** 1.2, 2), 1.2)
    assert math.isclose(get_exponent(7 ** 11, 7), 11)


def get_actual_zeros(roots: Union[List[complex], np.ndarray]) -> List[complex]:
    """
    Essential thing about parsing Polynomial code - take original polynomial's
    zeros and find the operations that they really correspond to

        Essential thing about parsing Polynomial code - take the polynomial's zeros
    after hiding them as exponents and after obfuscating them by multiplying
    by themselves. Find the primes and the exponents and return the original
    zeros - so the commands of our Polynomial program

    :param roots: array containing cleaned up roots of the polynomial
    :return: commands to perform when executing the Polynomial program, in order
    """
    # todo: we iterate over roots that have already been calculated
    # todo: we during every iteration calcualte log(prime)
    # todo: we could use NumPy to log() the whole array for efficiency
    # todo: why don't we make a generator from it?
    prime_gen = generate_primes()

    commands = []
    for prime, lim in zip(prime_gen, range(len(roots))):
        for root in roots:
            operator = get_exponent(root, prime)
            if math.isclose(operator, round(operator),
                            rel_tol=1e-6, abs_tol=1e-6):
                operator = int(round(operator, 8))

                # if root is complex, there is an operand; if it's a real
                # root - it's just a control flow operation, without an operand
                if root.imag:
                    operator = operator * 1j
                    operand = root.real
                    if math.isclose(operand, round(operand),
                                    rel_tol=1e-6, abs_tol=1e-6):
                        operand = int(round(operand, 8))
                else:
                    operand = 0
                commands += [complex(operand + operator)]
    return commands


def test_get_actual_zeros():
    roots = [4826809, 243, 25j, 1+11j, 7j, 1+2j]
    commands = [1+1j, 5, 2j, 1j, 1+1j, 6]
    assert get_actual_zeros(roots) == commands


def translate_to_python(polynomial_code: List[complex]) -> str:
    """
    Take the actual zeros of our Polynomial and translate it to Python, so we
    won't have to fool with interpreting it

    :param polynomial_code: a sequence of consecutive commands in Polynomial's
                            syntax (in order)
    :return: a string that is semantically correct and executable Python code
    """
    # i replaced i and 2i for output and input (respectively) by -i and -2i
    # to make lookups easier
    translations = {
        -1j: 'print(chr(ACC), end="")',  # i, (0 + 1j), output the register
        -2j: 'ACC = ord(input())',  # 2i, (0 + 2j), input the register
        1j: 'ACC += {}',
        2j: 'ACC -= {}',
        3j: 'ACC *= {}',
        4j: 'ACC /= {}',
        5j: 'ACC %= {}',
        6j: 'ACC **= {}',
        1: 'if ACC > 0:',
        2: '',  # end if
        3: 'if ACC < 0:',
        4: 'if not ACC:',
        5: 'while ACC > 0:',
        6: '',  # end while
        7: 'while ACC < 0:',
        8: 'while not ACC:'
    }

    python_code = 'ACC = 0\n'

    indent = 0
    for cmd in polynomial_code:
        python_code += '\t' * indent

        if cmd.imag:
            # i, 2i -> -i, -2i
            if cmd in {1j, 2j}:
                cmd = -cmd
            python_code += translations[cmd.imag * 1j].format(int(cmd.real))
        elif not cmd.imag:
            if cmd in {1, 3, 4, 5, 7, 8}:
                indent += 1
            elif cmd in {2, 6}:
                indent -= 1
            python_code += translations[int(cmd.real)]

        python_code += '\n'
    return python_code


def test_translate_to_python():
    pass


def run_tests():
    test_get_roots()
    test_parse_monomial()
    test_parse_polynomial()
    test_fix_coefficients()
    test_generate_primes()
    test_get_exponent()
    test_get_actual_zeros()
    test_translate_to_python()


def convert(source: str) -> str:
    """
    Convert Polynomial source code to Python source code

    :param source: correct Polynomial source code
    :return: correct Python source code
    """
    polynomial = parse_polynomial(source)
    coefficients = get_coefficients(polynomial)
    roots = get_roots(coefficients)
    commands = get_actual_zeros(roots)
    python = translate_to_python(commands)
    return python

if __name__ == '__main__':
    run_tests()#

    # this program writes its input to std output; CAT program
    prog = 'f(x) = x^10 - 4827056x^9 + 1192223600x^8 - 8577438158x^7 + 958436165464x^6 - 4037071023854x^5 + 141614997956730x^4 - 365830453724082x^3 + 5225367261446055x^2 - 9213984708801250x + 21911510628393750'
    # prog = 'f(x) = x^4 - 27x^3 + 59x^2 - 243x + 450'
    interpretable = convert(prog)

    print(interpretable)
