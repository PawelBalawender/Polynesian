# Polynesian
This package contains two utilities that are must-have in a [Polynomial](https://esolangs.org/wiki/Polynomial) programmer toolchain.

Those are: \
-*zeros* to *polynomial* converter[<sup>1</sup>](https://github.com/) \
-*polynomial* to Python converter[<sup>2</sup>]

# Motivation
This project was created because of my participation in [WWW workshops](https://warsztatywww.pl/) in Zabrze, Poland
for which to qualify I had to do accomplish a fixed set of qualifying exercises, amongst which was the exercise No. 1
from the file "Ezoteryczne języki programowania.pdf" - it was about implementing an interpreter for the Polynomial
esoteric programming language. I have written it, but wasn't satisfied with testing it with the example Polynomial
programmes copy-pasted from the [esolangs.org](https://esolangs.org/wiki/Polynomial), so decided to write some tool
to help me deal with this task

# Usage
Save your Polynomial code in a *.pol file or your *zeros* in a *.z file to
easily distinguish between them two
Run tests:
    python3 -m unittest TestPolynomialToPython.py

This package contains: \
    - a Polynomial to Python translator\
    - Raw polynomial commands (aka zeros) to correct Polynomial source code\
        (aka polynomial) translator

All the calculations needed to parse the Polynomial code are done using Numerical Python aka NumPy. \
 \
Usage: \
You can use it by passing your code/commands to string variables\
at the end of modules (somewhere inside the if \_\_name__ == '\_\_main__' clause)
