# Machine Learning Project 1: Finding Higgs Boson
The goal of this project is to apply machine learning techniques on a set of data samples from CERN in order to be able to discernate between Higgs Boson particles and background particles.

## Getting started
In order to run our code, all you need to do is to type in the console at the location of run.py:
```
python run.py 
```
The data csv files (test.csv and train.csv) must be located in the same folder as run.py. The output submission.csv file will be written at the same location as run.py.
 
## Dependencies
The only external libraries that we use are numpy and csv. In order to be executed, the run.py file needs the following files in the same folder:
```
csv_helper.py
data_cleaning.py
enhanced_implementations.py
poly_expansion.py
```
All those files should be already included in the code.zip file at the right location.

## implementations.py
The file implementations.py contains all 6 functions we were asked to write. Additionally, it also contains some helper functions at the end of the file, that help to build the 6 required functions.
Some functions from implementations.py are needed for the final program (run.py) so they are exactly the same as in enhanced_implementations.py, but we decided to copy them in order to make run.py not dependent on implementations.py.

## Credits
We used some provided functions for csv_helpers.py, and we used a slightly modified version of batch_iter from a lab's solutions. Some of the rest of our functions can be reusing some of the functions in the lab's solutions, but apart from those cited before, we always reworked them quite a bit to do it our way or to make them fit our situation and needs.
