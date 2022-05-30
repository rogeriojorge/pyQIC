INTODRUCTION
# pyQic
Python implementation of the Quasisymmetric Stellarator Construction method

This code implements the equations derived by Garren and Boozer (1991) for MHD equilibrium near the magnetic axis.


USEFUL LINKS


#CONNECT??


RUN THE CODE

To install this code you will need to open your Shell and insert the following command:




   pip install .
  


To run this code you will need to use your Python and insert the following command:


from qic import Qic
stel = Qic.from_paper('r2 section 5.2')


First of all you need to copy the folders and the files with the "git clone" command followed by the github repository's link.
Example:

    git clone _link_
    
   
   
Then you will need to install the librarys below:
  numpy
  scipy
  matplotlib
  
Example: 

    pip install numpy scipy matplotlib
    
