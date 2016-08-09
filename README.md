# Repository for helpful functions and classes

A short overview of the classes and functions is given:

## ml_utilities
Contains useful machine learning functions:
  * optimisePars: calculates score of a set of hyper-parameters to be able to choose the most performing algorithm.


## ValAndErr.py
Class that handles simple calculations with values and uncertainties.


## efficiency.py 
Class that handles efficiencies:
  * calc_efficiency(nEvents, nAll): Returns ValAndErr object with the efficiency and a binomial uncertainty.
  * eff(df, cut, prev_cut="", weight=""): calculates efficiencies on Pandas DataFrames with cuts. Optional weighted data can be processed and previous cuts can be given.


## utilities.py
General useful functions:
  * showUniques(pandas.DataFrame): prints unique values per column.
  * ensure_dir(directory): checks if directory exists and if not creates it.
  * print_bumper(test, c='=',n=-1): prints text between lines of characters _c_. Number of _c_ characters can be altered by the _n_ argument. Automatically, `n=len(text)` _c_ characters are printed.


## Plotting
Plotting scripts and utilities. Description given in sub-directory.


# Installation
* You should install scikit-learn, pandas, numpy, scipy prior cloning the `pythonTools` repository
* Get repository: git clone git@github.com:mschlupp/pythonTools.git
* Add `pythonTools` directory to your `$PATH` variable: (e.g. Linux) in your `.bashrc` add `export PATH="/path/to/the/git/clone/pythonTools:$PATH"`
