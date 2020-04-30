date: June 2018

1. Why did I do this?
I wanted to solve problems from this doc
https://github.com/eerio/polynomial/blob/master/task-set.pdf

2. What's that?
basically, programming in the Polynomial esolang might be difficult;
https://esolangs.org/wiki/Polynomial

but you can as well just write your program in `zeros` (it's Polynomial-
specific) and then it's not only easy to execute it, but also you can
directly translate it to a proper Polynomial program by just multiplying
these `zeros`

here you can find two scripts to help you become a Polynomial programmer:
-`zeros` to Polynomial converter
-Polynomial to Python translator

so you can basically just write your progam in `zeros`, convert it into Python
and then execute the obtained Python source code natively on you machine

3. How do I use it?
$ polynomial-to-py.py test/hello.pol
<python-subset equivalent of your Polynomial program>
$ zero_to_py.py test/hello.z
<python-subset equivalent of your `zeros` program>
$ zero_to_py.py test/hello.z -pol
<Polynomial form of your `zeros` program>

4. How do I test it?
You don't, there were some test, but it was just pain in the ass when I moved
some files here and there, so currently the coverage == 0%; it works though

