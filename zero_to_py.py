#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
This module implements a translator from Polynomial "zeros" to a correct Polynomial source code

Let's suppose that we want to make a Polynomial program that writes its input
to standard output (CAT program):

The zeros we want are, in order:
    zeros = 1+i, 5, 2i, i, 1+i, 6
Now we apply convert() on them:
    polynomial_code = convert(zeros)

convert() firstly encodes operations' order:
    zeros = 1+2^1i, 3^5, 5^2i, 7^1i, 1+11^1i, 13^6
Then inserts complex conjugates since they also have to be zeros of this polynomial
    zeros = 1+2i, 1-2i, 243, 25i, -25i, 7i, -7i, 1+11i, 1-11i, 13^6
Then calculates a polynomial which has our zeros and only our zeros
    polynomial_code = "f(x) = (x - (1+2i))(x - (1-2i))(x - 243)...(x - 13^6)"
Formats it to a piece of correct Polynomial code and outputs

todo: add the whole package test suite with some open() redirection etc.
"""
__all__ = ['Commands', 'add_conjugates', 'add_primes', 'polynomial_to_string', 'parse_source']

from typing import List, Union
import pathlib
import math
import sys

import numpy as np

import polynomial_to_py as PolynomialToPython
import utils as Utilities

Commands = List[Union[complex, int]]


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


def add_primes(commands: Commands) -> Commands:
    """
    Encode order of commands by rising consecutive prime numbers to their power
    There cannot be any complex conjugates(!) - they will be added later

    :param commands: original array of commands
    :return: encoded array of commands
    """
    for i, prime in zip(range(len(commands)), Utilities.generate_primes()):
        cmd = commands[i]
        if not cmd.imag:
            commands[i] = prime ** cmd.real
        elif cmd.imag:
            commands[i] = cmd.real + (prime ** cmd.imag) * 1j
    return commands


def polynomial_to_string(p: np.ndarray) -> str:
    """
    Create a string representing a polynomial just from its coefficients

    :param p: polynomial's coefficients
    :return: string representation, human-readable
    """
    string = 'f(x) = '
    for power, coefficient in reversed(list(enumerate(p[::-1]))):
        sgn = ['-', '+'][int(coefficient) > 0]
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


def parse_source(source: str) -> Commands:
    """
    Extract zeros from a correct 'zeros' source code

    :param source: source file containing zeros and comments
    :return: zeros extracted from the source file, in order of occurence
    """
    commands = []
    for line in source.splitlines():
        # line is in a form of e.g. "72+1j      # print 72"
        events = line.split()
        for i in events:
            # events is in a form of e.g. ['72+1j', '#', 'print', '72']
            try:
                commands.append(complex(i))
            except ValueError:
                if i == '#':
                    break

    return commands


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


def main(args: List[str]) -> None:
    pol_flag = '-pol' in args
    if pol_flag:
        args.remove('-pol')

    file_path = pathlib.PurePath(__file__)

    # defaults
    in_file = pathlib.Path.joinpath(file_path.parents[1], 'test', 'zeros')

    if len(args) >= 2:
        in_file = pathlib.PurePath(args[1])
    py_out_file = in_file.with_suffix('.py')
    if len(args) >= 3:
        py_out_file = pathlib.PurePath(args[2])

    with open(in_file) as doc:
        _source = doc.read()

    zeros = parse_source(_source)
    polynomial_code = convert(zeros)

    if pol_flag:
        pol_out_file = py_out_file.with_suffix('.pol')
        with open(pol_out_file, 'w') as doc:
            doc.write(polynomial_code)

    python_code = PolynomialToPython.convert(polynomial_code)
    with open(py_out_file, 'w') as doc:
        doc.write(python_code)


if __name__ == '__main__':
    main(sys.argv)
