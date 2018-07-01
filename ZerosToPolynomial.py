#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
This module implements a translator from raw Polynomial <actual-zeros>, so the
operations that we actually want to perform, to a correct Polynomial source code

Let's suppose that we want to make a Polyonmial program that writes its input
to standard output (CAT program):

The zeros we want are, in order:
    prog = 1+i, 5, 2i, i, 1+i, 6
Now we apply convert() on them:
    code = convert(prog)

convert() firstly encodes its order in prime numbers:
    prog = 1+2^1i, 3^5, 5^2i, 7^1i, 1+11^1i, 13^6
Then adds complex conjugates since they also have to be zeros of this polynomial
    prog = 1+2i, 1-2i, 243, 25i, -25i, 7i, -7i, 1+11i, 1-11i, 13^6
Then calcualtes the polynomial from our zeros:
    f(x) = (x - (1+2i))(x - (1-2i))(x - 243)...(x - 13^6)
Formats it to a piece of correct Polynomial code and outputs
"""
from typing import List, Generator, Union, NewType
import math

import numpy as np

import Polynomial

Commands = NewType('Commands', List[Union[complex, int]])


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


def add_conjugates(commands: Commands) -> Commands:
    """
    Add missing complex conjugates right after the original command

    :param commands: original array of commands
    :return: completed array of commands
    """
    index = 0
    while index < len(commands):
        cmd = commands[index].imag
        if not math.isclose(cmd, 0, abs_tol=1e-8) and cmd > 0:
            commands.insert(index + 1, commands[index].conjugate())
        index += 1

    return commands


def test_add_conjugates():
    # Cat program:
    commands = 1+1j, 5, 2j, 1j, 1+1j, 6
    result = 1+1j, 1-1j, 5, 2j, -2j, 1j, -1j, 1+1j, 1-1j, 6
    commands = list(map(complex, commands))
    result = list(map(complex, result))
    assert add_conjugates(commands) == result


def add_primes(commands: Commands) -> Commands:
    """
    Encode order of commands by rising consecutive prime numbers to their power
    There cannot be any complex conjugates(!) - they will be added later

    :param commands: original array of commands
    :return: encoded array of commands
    """
    for i, prime in zip(range(len(commands)), generate_primes()):
        cmd = commands[i]
        if not cmd.imag:
            commands[i] = prime ** cmd.real
        elif cmd.imag:
            commands[i] = cmd.real + (prime ** cmd.imag) * 1j
    return commands


def test_add_primes():
    # 1, i, 2 -> 2^1, 3^1i, 5^2 -> 2^1, 3^1i, -3^1i, 5^2
    commands = list(map(complex, [1, 1j, 2]))
    result = list(map(complex, [2, 3j, 25]))
    assert add_primes(commands) == result


def polynomial_to_string(p: np.ndarray) -> str:
    """
    Create a string representing a polynomial just from its coefficients

    :param p: polynomial's coefficients
    :return: string representation, human-readable
    """
    string = 'f(x) = '
    for power, coefficient in reversed(list(enumerate(p[::-1]))):
        sgn = ['-', '+'][coefficient > 0]
        string += f'{sgn} '
        if coefficient != 1:
            string += str(abs(int(coefficient)))

        if power == 1:
            string += f'x '
        elif power not in {0, 1}:
            string += f'x^{power} '
    if string[7] == '+':
        string = string[:7] + string[9:]
    return string


def test_polynomial_to_string():
    polynomial = np.poly1d([1, 2, 3, -4])
    string = 'f(x) = x^3 + 2x^2 + 3x - 4'
    assert polynomial_to_string(polynomial.c) == string


def run_tests():
    test_generate_primes()
    test_add_conjugates()
    test_add_primes()
    test_polynomial_to_string()


def convert(commands: Commands) -> str:
    """
    Convert Polynomial-lang's raw commands to it's correct source code
    :param commands: raw operations, without complex conjugates
    :return: resulting polynomial in the form of its coefficient's array
    """
    ordered = add_primes(commands)
    complete = add_conjugates(ordered)
    polynomial = np.poly1d(complete, True)
    coefficients = polynomial.coefficients
    string = polynomial_to_string(coefficients)
    return string

if __name__ == '__main__':
    run_tests()

    # this program writes its input to standard output; CAT program
    prog = [1+1j, 5, 2j, 1j, 1+1j, 6]
    pol = convert(prog)
    print(Polynomial.convert(pol))
