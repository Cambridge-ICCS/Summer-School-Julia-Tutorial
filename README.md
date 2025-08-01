<img src="https://cambridge-iccs.github.io/assets/images/iccs-logo.png"  width="50%" align="left" style='clear:both'>


<br><br><br><br>

# Introduction to Computational Science in Julia

![GitHub](https://img.shields.io/github/license/Cambridge-ICCS/Summer-School-Julia-Tutorial)

This is an intensive 90 minute workshop for the Julia language, including 3 web-based Pluto notebooks covering concepts from basic to intermediate level in Julia. It assumes that the attendees have some basic familiarity with programming. It also has an emphasis on scientific computing, so it would be beneficial if you have some experience in this field as well. The goal of the workshop is to help you gain experience and have fun in Julia programming.

To participate in this workshop, we recommend that you follow these steps to get started with a Julia programming environment:

## Preparation

### Clone this GitHub repository

Clone this repository using `Git` (which we assume you have some basic experience with; if not, run `git clone https://github.com/Cambridge-ICCS/Summer-School-Julia-Tutorial.git`).

### Install Julia

If you have not done so, download Julia from https://julialang.org/downloads/, and install it following the instructions on the same page.

### Run the Julia REPL

In your command line, run `julia` to open an interactive REPL (read-evaluate-print loop) session. Enter expressions like `1 + 2`, `[1, 2, 3][1:2]`, `2sqrt(pi)` to play around a bit.

### Install Julia Packages

In the Julia REPL session, you can install packages by entering `]` (which changes the prompt to `pkg>`) followed by `add PACKAGE`. Now try to install the web-based notebook package `Pluto`. `Pluto` is similar to the `Jupyter` notebook (which supports both Python and Julia), but it has some advantages like reactivity and reproducibility). Do these steps to install Pluto: Go into the package mode, `add Pluto`.


### Run a Pluto Notebook Session

Enter a backspace to return to the normal mode from the package mode. Then run either `import Pluto`, or `using Pluto; Pluto.run()` to open a Pluto Notebook session (a webpage will automatically pop up). 

### Run the Notebooks in this Repository

After a webpage for the Pluto Notebook session has poped up in your browser, open a notebook by entering its file path. Start with `basics.jl`.

## FAQ

### How to type Unicode characters?
Type the corresponding LaTeX-like character sequence (starting with `\`), followed by a `tab`.

## Further Reading

The following pages give further background on the Julia language:

* https://cheatsheet.juliadocs.org/
* https://github.com/mitmath/julia-mit/blob/master/Julia-intro.pdf
* https://computationalthinking.mit.edu/

