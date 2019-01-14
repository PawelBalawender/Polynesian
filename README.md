# Polynesian
This package contains two utilities that are must-have in a [Polynomial](https://esolangs.org/wiki/Polynomial) programmer toolchain.

Those are: \
-*zeros* to *polynomial* converter[<sup>1</sup>](https://github.com/) \
-*polynomial* to Python converter[<sup>2</sup>]

# Motivation
This project was created because of my participation in [WWW workshops](https://warsztatywww.pl/) in Zabrze, Poland for which to qualify I had to \
accomplish a fixed set of qualifying exercises, amongst which was the exercise No. 1 from [this PDF file](https://github.com/PawelBalawender/Polynesian/blob/master/Ezoteryczne%20j%C4%99zyki%20programowania.pdf)  - it was \
about implementing an interpreter for the Polynomial esoteric programming language. I have written it, but wasn't satisfied with testing it with the \
example Polynomial programmes copy-pasted from the [esolangs.org](https://esolangs.org/wiki/Polynomial), so decided to write some tool to help me deal \
with this task

# Usage
Save your Polynomial code in a *.pol file or your *zeros* in a *.z file to easily distinguish between them
You can add inline Python-style comments to the code (example: [HelloWorld.z](https://github.com/PawelBalawender/Polynesian/blob/master/HelloWorld.z)
Run it as: \
`python3 ZerosToPython.py <your .z file> [<output .pol file>]` \
or \
`python3 PolynomialToPython.py <your .pol file> [<output .py file>]` \

If <output .* file> unspecified, it will save the output in a file with the same name as the input file, but extension changed properly \
Be sure to give the script write permissions so it's able to save its output \

# Run tests
All the test suites are in the test_polynesian.py file. Run: \
`python3 -m unittest TestPolynomialToPython.py` \
From the projects's root directory

# External libraries used
- [NumPy](http://www.numpy.org/) (special credit! this tool wouldn't exist without NumPy's easiness of counting polynomials)
- [Coverage.py](https://coverage.readthedocs.io/en/coverage-4.5.1a/#)

# Todo
Add RaspberryPi support
Add a Makefile for test running
Make 100% coverage
Find redundant or invalid tests
Update Usage and Run tests sections

