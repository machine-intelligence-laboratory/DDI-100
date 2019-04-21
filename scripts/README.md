In this directory you can find some python scripts with usefull functions and classes for working with our dataset.
There are functions for visualization dataset and convinient comparison of ground truth and predicted masks.
Also class Generator are presented. It has get_doc(), get_string() and get_char() methods that randomly choses document, word or char sample from dataset correspondingly.

get_string() and get_char() methods also removes geometric transformations of the box as otherwise adjacent lines can be captured.

All functions and generator methods have docstrings which clarify there signatures and applications.

Before using make sure you have the following packages installed in your environment: numpy, opencv, pickle, glob, pathlib.
