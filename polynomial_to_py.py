#!/usr/bin/env python3
"""
This module implements an translator for the Polynomial esoteric language
Save your correct Polynomial code as a string variable and apply convert()
function on it
"""

from collections import namedtuple
from typing import List, Tuple, Union
import pathlib
import math
import sys

import numpy as np

try:
    import src.utils as Utilities
except ImportError:
    import utils as Utilities

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

    # if there's no coefficient given, this string it's '', so we make or '1'
    # cause default coefficient is 1
    coefficient = int(monomial[:coefficient_end] or '1')
    return coefficient, power


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
    return exp


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
    prime_gen = Utilities.generate_primes()

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


def main(args: List[str]) -> None:
    file_path = pathlib.PurePath(__file__)
    cat = pathlib.Path('test/CAT.py')
    in_file = pathlib.PurePath(args[1]) if len(args) >= 2 else cat

    with open(in_file) as doc:
            lines = doc.readlines()
            # trim comments
            program = ''.join(i for i in lines if not i.startswith('#'))

    python_code = convert(program)
    print(python_code)


if __name__ == '__main__':
    main(sys.argv)
