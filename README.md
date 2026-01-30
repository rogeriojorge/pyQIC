About pyQIC
=

```pyQIC``` is a python package for generating quasi-isodynamic stellarator configurations using an expansion about magnetic axis.  pyQIC is written in pure python. This makes pyQIC user-friendly, with no need for compilation. though it is slower.

This code implements the equations derived by Garren and Boozer (1991) for MHD equilibrium near the magnetic axis.
It is similar to pyQSC but does not assume quasisymmetry.

Requirements
=
```pyQIC``` has minimal dependencies - it requires only python3, numpy, scipy, matplotlib. If you don't already have numpy, scipy and matplotlib, they will be installed automatically by the ***pip install*** step in the **Run the Code** section.



USEFUL LINKS
=
If you need more help regarding near-axis stellarators, you can check the documentation for the quasi-symmetric version of pyQIC: click [here](https://landreman.github.io/pyQSC/getting_started.html#)


INSTALLATION
=
Installation from pypi can be done via
```
pip install qicna
``` 
<br>

To install the code locally, you can run
<br>

First of all you need to copy the folders and the files with the "git clone" command, followed by the GitHub repository's link.
Example:
```
git clone https://github.com/rogeriojorge/pyQIC.git
``` 
<br>


Then, install the package in your local Python environment with:
```
cd pyQIC
pip install -e .
```
<br>

Then you also need to install the libraries below:<br>
  ***numpy<br>
  scipy<br>
  matplotlib***
  
Example: 
```
pip install numpy scipy matplotlib
``` 
<br>

RUN THE CODE
=

To run this code, you will need to use Python and insert the following command:

```
from qic import Qic
stel = Qic.from_paper('r2 section 5.2')
```
<br>

