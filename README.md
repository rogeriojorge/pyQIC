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


INSTALLATION
=
To install the code you will need to follow the steps bellow:
<br>

To install this code you will need to open your Shell and insert the following command:
```
pip install .
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

RUN THE CODE
=

To run this code you will need to use your Python and insert the following command:

```
from qic import Qic
stel = Qic.from_paper('r2 section 5.2')
stel.plot_boundary()
```
<br>

What this command do is, he goes through your files and it will search for the picture you told it to search and it will open. In this case is two pictures but you can dicide how many you want it to open. More you open, more it takes to open

