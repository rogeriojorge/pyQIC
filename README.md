About pyQIC
=

```pyQIC``` is a python package for generating quasi-isodynamic stellarator configurations using an expansion about magnetic axis.  pyQIC is written in pure python. This makes pyQIC user-friendly, with no need for compilation. though it is slower.

This code implements the equations derived by Garren and Boozer (1991) for MHD equilibrium near the magnetic axis.


Requirements
=
```pyQIC``` has minimal dependencies - it requires only python3, numpy, scipy, matplotlib. If you don't already have numpy, scipy and matplotlib, they will be installed automatically by the ***pip install*** step in the **Run the Code** section.



USEFUL LINKS
=
If you need more help regarding near-axis stellarators, you can check the documentation for the quasi-symmetric version of pyQIC: click [here](https://landreman.github.io/pyQSC/getting_started.html#)


INSTALLATION
=
To install the code you will need to follow the steps bellow:
<br>

First of all you need to copy the folders and the files with the "git clone" command followed by the github repository's link.
Example:
```
git clone https://github.com/rogeriojorge/pyQIC.git
``` 
<br>


Then, install the package to your local python environment with:
```
cd pyQIC
pip install -e .
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

RUN THE CODE
=

To run this code you will need to use your Python and insert the following command:

```
from qic import Qic
stel = Qic.from_paper('r2 section 5.2')
```
<br>

