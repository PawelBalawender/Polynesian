#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from typing import Generator


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
