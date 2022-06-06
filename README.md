About pyQIC
=

```pyQIC``` is a python package for generating quasi-isodynamic stellarator configurations using an expansion about magnetic axis.  pyQIC is written in pure python. This makes pyQIC user-friendly, with no need for compilation. though it is slower.

This code implements the equations derived by Garren and Boozer (1991) for MHD equilibrium near the magnetic axis.


Requirements
=
```pyQIC``` has minimal dependencies - it requires only python3, numpy, scipy, matplotlib. If you don't already have numpy, scipy and matplotlib, they will be installed automatically by the ***pip install*** step in the **Run the Code** section.



USEFUL LINKS
=
If you need more help click [here](https://landreman.github.io/pyQSC/getting_started.html#)


RUN THE CODE
=

To install this code you will need to open your Shell and insert the following command:
```
pip install .
```
<br>

To run this code you will need to use your Python and insert the following command:

```
from qic import Qic
stel = Qic.from_paper('r2 section 5.2')
```
<br>

First of all you need to copy the folders and the files with the "git clone" command followed by the github repository's link.
Example:
```
git clone _link_
``` 
<br>

Then intall the package to your local python environment with:
```
cd pyQIC
pip install -e
```
<br>

Then you also need to install the librarys below:<br>
  ***numpy<br>
  scipy<br>
  matplotlib***
  
Example: 
```
pip install numpy scipy matplotlib
``` 
<br>

Post-Installation
=

If the installation is successful, pyQIC will be added to your python environment. To use it in python, simply import the module as:
```
>>> from qic import Qic
```