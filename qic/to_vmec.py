"""
This module contains the routines to output a
near-axis boundary to a VMEC input file
"""
from datetime import datetime
import numpy as np
from .Frenet_to_cylindrical import Frenet_to_cylindrical
from .Frenet_to_cylindrical import Frenet_to_cylindrical_1stOrder
from .util import mu0, to_Fourier

def to_vmec(self, filename, r=0.1, params=dict(), ntheta=20, ntorMax=14,firstOrderSurface=False):
    """
    Outputs the near-axis configuration calculated with pyQIC to
    a text file that is able to be read by VMEC.

    Args:
        filename: name of the text file to be created
        r:  near-axis radius r of the desired boundary surface
        params: a Python dict() instance containing one/several of the following parameters: mpol,
          delt, nstep, tcon0, ns_array, ftol_array, niter_array
        ntheta: resolution in the poloidal angle theta for the Frenet_to_cylindrical and VMEC calculations
        ntorMax: maximum number of NTOR in the resulting VMEC input file
        
    """
    if "mpol" not in params.keys():
        mpol1d = 100 # maximum number of mode numbers VMEC can handle
        mpol = int(np.floor(min(ntheta / 2, mpol1d)))
    else:
        mpol = int(params["mpol"])
    if "ntor" not in params.keys():
        # We should be able to resolve (N_phi-1)/2 modes (note integer division!), but in case N_phi is very large, don't attempt more than the vmec arrays can handle.
        ntord = 100 # maximum number of mode numbers VMEC can handle
        ntor = int(min(self.nphi / 2, ntord))
    else:
        ntor = int(params["ntor"])
    if "delt" not in params.keys():
        params["delt"] = 0.9
    if "nstep" not in params.keys():
        params["nstep"] = 200
    if "tcon0" not in params.keys():
        params["tcon0"] = 2.0
    if "ns_array" not in params.keys():
        params["ns_array"] = [16,49,101]
    if "ftol_array" not in params.keys():
        params["ftol_array"] = [1e-14,1e-13,1e-13]
    if "niter_array" not in params.keys():
        params["niter_array"] = [2000,2000,2000]

    phiedge = np.pi * r * r * self.spsi * self.Bbar

    # Set pressure Profile
    temp = - self.p2 * r * r
    am = [temp,-temp]
    pmass_type='power_series'
    pres_scale=1

    # Set current profile:
    ncurr = 1
    pcurr_type = 'power_series'
    ac = [1]
    curtor = 2 * np.pi / mu0 * self.I2 * r * r
   
    #firstOrderSurface = True
   
    if firstOrderSurface == True:
        R1c, R1s, Z1c, Z1s = Frenet_to_cylindrical_1stOrder(self,r,ntheta)
        self.R1c = R1c
        self.R1s = R1s
        self.Z1s = Z1s
        self.Z1c = Z1c
        rbc1 = np.zeros((2*ntor+1,1))
        rbs1 = np.zeros((2*ntor+1,1))
        zbc1 = np.zeros((2*ntor+1,1))
        zbs1 = np.zeros((2*ntor+1,1))

        rbc1[ntor] = np.mean(R1c)
        rbs1[ntor] = np.mean(R1s)
        zbc1[ntor] = np.mean(Z1c)
        zbs1[ntor] = np.mean(Z1s)
 
        for n in range(0, ntor):
            # RBC
            half_sum = np.mean(R1c * np.cos(self.nfp*n*self.phi))
            half_difference = np.mean(R1s * np.sin(self.nfp*n*self.phi))
            rbc1[ntor+n] = half_sum + half_difference
            rbc1[ntor-n] = half_sum - half_difference
        
            # ZBC
            half_sum = np.mean(Z1c * np.cos(n*self.nfp*self.phi))
            half_difference = np.mean(Z1s * np.sin(n*self.nfp*self.phi))
            zbc1[ntor+n] = half_sum + half_difference
            zbc1[ntor-n] = half_sum - half_difference
        
            # RBS
            half_sum = np.mean(R1s * np.cos(self.nfp*n*self.phi))
            half_difference = -np.mean(R1c * np.sin(self.nfp*n*self.phi))
            rbs1[ntor+n] = half_sum + half_difference
            rbs1[ntor-n] = half_sum - half_difference
        
            # ZBS
            half_sum = np.mean(Z1s * np.cos(self.nfp*n*self.phi))
            half_difference = -np.mean(Z1c * np.sin(self.nfp*n*self.phi))
            zbs1[ntor+n] = half_sum + half_difference
            zbs1[ntor-n] = half_sum - half_difference
    
        rbc1 = -rbc1 * r
        rbs1 = -rbs1 * r
        zbc1 = zbc1 * r
        zbs1 = zbs1 * r
        
        ### only valid for stellarator symmetry 
        RBC = np.zeros((int(2 * ntor + 1), int(mpol + 1)))
        RBS = np.zeros((int(2 * ntor + 1), int(mpol + 1)))
        ZBC = np.zeros((int(2 * ntor + 1), int(mpol + 1)))
        ZBS = np.zeros((int(2 * ntor + 1), int(mpol + 1)))

        RBC[:,1] = rbc1.transpose()
        RBS[:,1] = rbs1.transpose()
        ZBC[:,1] = zbc1.transpose()
        ZBS[:,1] = -zbs1.transpose()
        RBC[ntor:ntor+self.nfourier,0] = self.rc
        RBS[ntor:ntor+self.nfourier,0] = self.rs
        ZBC[ntor:ntor+self.nfourier,0] = self.zc
        ZBS[ntor:ntor+self.nfourier,0] = self.zs
    else:
    # Get surface shape at fixed off-axis toroidal angle phi
        R_2D, Z_2D, phi0_2D = self.Frenet_to_cylindrical(r, ntheta)
    
    # Fourier transform the result.
    # This is not a rate-limiting step, so for clarity of code, we don't bother with an FFT.
        RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, self.nfp, mpol, ntor, self.lasym)
    
    # Write to VMEC file
    file_object = open(filename,"w+")
    file_object.write("! This &INDATA namelist was generated by pyQIC: github.com/rogeriojorge/pyQIC\n")
    file_object.write("! Date: "+datetime.now().strftime("%B %d, %Y")+", Time: "+datetime.now().strftime("%H:%M:%S")+" UTC"+datetime.now().astimezone().strftime("%z")+"\n")
    file_object.write('! Near-axis parameters:  radius r = '+str(r)+', etabar = '+str(self.etabar)+'\n')
    file_object.write('! nphi = '+str(self.nphi)+', order = '+self.order+', sigma0 = '+str(self.sigma0)+', I2 = '+str(self.I2)+', B0 = '+str(np.mean(self.Bbar))+'\n')
    file_object.write('! Resolution parameters: ntheta = '+str(ntheta)+', mpol = '+str(mpol)+', ntor = '+str(ntor)+'\n')
    file_object.write('!----- Runtime Parameters -----\n')
    file_object.write('&INDATA\n')
    file_object.write('  DELT = '+str(params["delt"])+'\n')
    file_object.write('  NSTEP = '+str(params["nstep"])+'\n')
    file_object.write('  TCON0 = '+str(params["tcon0"])+'\n')
    file_object.write('  NS_ARRAY = '+str(params["ns_array"])[1:-1]+'\n')
    file_object.write('  FTOL_ARRAY = '+str(params["ftol_array"])[1:-1]+'\n')
    file_object.write('  NITER_ARRAY = '+str(params["niter_array"])[1:-1]+'\n')
    file_object.write('!----- Grid Parameters -----\n')
    file_object.write('  LASYM = '+str(self.lasym)+'\n')
    file_object.write('  NFP = '+str(self.nfp)+'\n')
    file_object.write('  MPOL = '+str(mpol)+'\n')
    file_object.write('  NTOR = '+str(min(ntor,ntorMax))+'\n')
    file_object.write('  PHIEDGE = '+str(phiedge)+'\n')
    file_object.write('!----- Pressure Parameters -----\n')
    file_object.write('  PRES_SCALE = '+str(pres_scale)+'\n')
    file_object.write("  PMASS_TYPE = '"+pmass_type+"'\n")
    file_object.write('  AM = '+str(am)[1:-1]+'\n')
    file_object.write('!----- Free Boundary Parameters -----\n')
    file_object.write('  LFREEB = F')
    file_object.write('!----- Current/Iota Parameters -----\n')
    file_object.write('  CURTOR = '+str(curtor)+'\n')
    file_object.write('  NCURR = '+str(ncurr)+'\n')
    file_object.write("  PCURR_TYPE = '"+pcurr_type+"'\n")
    file_object.write('  AC = '+str(ac)[1:-1]+'\n')
    file_object.write('!----- Axis Parameters -----\n')
    # To convert sin(...) modes to vmec, we introduce a minus sign. This is because in vmec,
    # R and Z ~ sin(m theta - n phi), which for m=0 is sin(-n phi) = -sin(n phi).
    file_object.write('  RAXIS_CC = '+str(self.rc)[1:-1]+'\n')
    if self.lasym:
        file_object.write('  RAXIS_CS = '+str(-self.rs)[1:-1]+'\n')
        file_object.write('  ZAXIS_CC = '+str(self.zc)[1:-1]+'\n')
    file_object.write('  ZAXIS_CS = '+str(-self.zs)[1:-1]+'\n')
    file_object.write('!----- Boundary Parameters -----\n')
    for m in range(mpol+1):
        for n in range(-ntor,ntor+1):
            if RBC[n+ntor,m]!=0 or ZBS[n+ntor,m]!=0:
                file_object.write(    '  RBC('+f"{n:03d}"+','+f"{m:03d}"+') = '+f"{RBC[n+ntor,m]:+.16e}"+',    ZBS('+f"{n:03d}"+','+f"{m:03d}"+') = '+f"{ZBS[n+ntor,m]:+.16e}"+'\n')
                if self.lasym:
                    file_object.write('  RBS('+f"{n:03d}"+','+f"{m:03d}"+') = '+f"{RBS[n+ntor,m]:+.16e}"+',    ZBC('+f"{n:03d}"+','+f"{m:03d}"+') = '+f"{ZBC[n+ntor,m]:+.16e}"+'\n')
    file_object.write('/\n')
    file_object.close()

    self.RBC = RBC.transpose()
    self.ZBS = ZBS.transpose()
    if self.lasym:
        self.RBS = RBS.transpose()
        self.ZBC = ZBC.transpose()
    else:
        self.RBS = RBS
        self.ZBC = ZBC
