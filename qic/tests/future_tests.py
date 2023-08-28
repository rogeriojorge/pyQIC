    def test_omnigenity_from_boozxform(self):
        
        def booz_xform_compare(dic, nphi=200):
            import matplotlib.pyplot as plt
            for name, py in dic.items():
                vmecfile = "/Users/rogeriojorge/local/NearAxis_Optimization/Results/"+name+"/wout_"+name+".nc"
                boozfile = "/Users/rogeriojorge/local/NearAxis_Optimization/Results/"+name+"/boozmn_"+name+".nc"
                order=str(name[-2::])
                stel = Qic.from_boozxform(booz_xform_file=boozfile, vmec_file=vmecfile, order=order, omn=py.omn, N_phi=nphi, nNormal=-py.N_helicity)
                plt.figure()
                plt.title(name+' B0')
                plt.plot(stel.B0_boozxform_array, 'r-', label='B0 VMEC')
                plt.plot(py.B0,                   'r.', label='B0 Near-Axis')
                plt.legend()
                plt.figure()
                plt.title(name+' B1')
                plt.plot(stel.B1c_boozxform_array, 'r-', label='B1c VMEC')
                plt.plot(py.B1c,                   'r.', label='B1c Near-Axis')
                plt.plot(stel.B1s_boozxform_array, 'b-', label='B1s VMEC')
                plt.plot(py.B1s,                   'b.', label='B1s Near-Axis')
                plt.legend()
                if order!='r1':
                    plt.figure()
                    plt.title(name+' B2')
                    plt.plot(stel.B20_boozxform_array, 'r-', label='B20 VMEC')
                    plt.plot(py.B20,                   'r--', label='B20 Near-Axis')
                    plt.plot(stel.B2c_boozxform_array, 'b-', label='B2c VMEC')
                    plt.plot(py.B2c_array,             'b--', label='B2c Near-Axis')
                    plt.plot(stel.B2s_boozxform_array, 'm-', label='B2s VMEC')
                    plt.plot(py.B2s_array,             'm--', label='B2s Near-Axis')
                    plt.legend()
                plt.show()

        dic  = {}
        nphi = 351

        name   = 'QI_NFP4_r1'
        rc      = [ 1.0,0.0,-0.015384615384615385,0.0,0.0,0.0,0.0 ]
        zs      = [ 0.0,0.0,0.008853685778690713,0.0,0.000246103120755672,0.0,1.241037713848815e-05 ]
        B0_vals = [ 1.0,0.11678284595922511 ]
        d_svals = [ 0.0,1.2943134984231057,0.02025266272263474,-0.09056889724435162 ]
        delta   = 0.7075140861631667
        nfp     = 4
        py      = Qic(rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, delta=delta)
        dic[name] = py

        # booz_xform_compare(dic,nphi=nphi)

    def test_B_rms_difference_VMEC(self):
        import booz_xform as bx
        from os import path 
        from subprocess import run
        from scipy.io import netcdf
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        # Input parameters
        executables_path = "/Users/rogeriojorge/bin/"
        name="QI"
        r_edge = np.exp(-np.linspace(2,6,6))
        Ntheta = 80
        Nzeta  = 140
        # Theta and Phi Arrays
        theta = np.linspace(0,2*np.pi,Ntheta)
        zeta = np.linspace(0,2*np.pi,Nzeta)
        zeta2D,theta2D = np.meshgrid(zeta,theta)
        rc      = [ 1.0,0.0,-0.4056622889934463,0.0,0.07747378220100756,0.0,-0.007803860877024245,0.0,0.0,0.0,0.0,0.0,0.0 ]
        zs      = [ 0.0,0.0,-0.24769666390049602,0.0,0.06767352436978152,0.0,-0.006980621303449165,0.0,-0.0006816270917189934,0.0,-1.4512784317099981e-05,0.0,-2.839050532138523e-06 ]
        B0_vals = [ 1.0,0.16915531046156507 ]
        omn_method ='non-zone-fourier'
        k_buffer = 3
        k_second_order_SS   = 0.0
        d_over_curvature   = 0.5183783762725197
        d_svals = [ 0.0,0.003563114185517955,0.0002015921485566435,-0.0012178616509882368,-0.00011629450296628697,-8.255825435616736e-07,3.2011540526397e-06 ]
        delta   = 0.1
        nfp     = 1
        nphi    = 251
        stel    =  Qic(omn_method = omn_method, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, delta=delta, d_over_curvature=d_over_curvature, k_second_order_SS=k_second_order_SS)
        # Loop over r_edge
        b_RMS_VMEC_diff = []
        b_RMS_BOOZ_diff = []
        for r in r_edge:
            print('r =',r)
            # Run VMEC
            if not path.exists('wout_'+name+'r'+str(r)+'.nc'):
                # Output to VMEC
                stel.to_vmec('input.'+name+'r'+str(r),r=r,
                params={"ns_array": [16,49],#,101,151],
                        "ftol_array": [1e-13,2e-14],#,1e-13,1e-14],
                        "niter_array": [600,1200],#,1000,2000],
                        "mpol": 14,
                        "ntor": 80
                        }, ntheta=28, ntorMax=80)
                # Run VMEC
                bashCommand = executables_path+"/./xvmec2000 input."+name+'r'+str(r)
                run(bashCommand.split())
                # os.remove('mercier.'+name+'r'+str(r))
                # os.remove('input.'+name+'r'+str(r))
                # os.remove('threed1.'+name+'r'+str(r))
                # os.remove('parvmecinfo.txt')
                # os.remove('timings.txt')
                # try:
                #     os.remove('jxbout_'+name+'r'+str(r)+'.txt')
                # except:
                #     print('')
            else:
                sys.path.insert(1, "/Users/rogeriojorge/local/NearAxis_Optimization/Plotting")
                # import vmecPlot2
                # vmecPlot2.main('wout_'+name+'r'+str(r)+'.nc',stel,r,savefig=False)
                # plt.show()
                # exit()
            # Run BOOZ_XFORM
            b1 = bx.Booz_xform()
            if not path.exists('boozmn_'+name+'r'+str(r)+'.nc'):
                b1.read_wout('wout_'+name+'r'+str(r)+'.nc')
                b1.compute_surfs = [1,2,4,8,15,25,35,47]#,55,80,99]
                b1.mboz = 150
                b1.nboz = 80
                b1.run()
                b1.write_boozmn('boozmn_'+name+'r'+str(r)+'.nc')
            b1.read_boozmn('boozmn_'+name+'r'+str(r)+'.nc')
            # plt.figure(); bx.surfplot(b1, js=0,  fill=False, ncontours=35)
            # plt.figure(); bx.surfplot(b1, js=3, fill=False, ncontours=35)
            # plt.figure(); bx.surfplot(b1, js=6, fill=False, ncontours=35)
            # plt.figure(); bx.symplot(b1, helical_detail = True, sqrts=True)
            plt.figure(); bx.modeplot(b1, sqrts=True); plt.xlabel(r'$s=\sqrt{\psi/\psi_b}$')
            plt.savefig('BOOZ_modes_r'+str(r)+'.pdf'); plt.close()
            # Get |B| on surface from pyQic
            bQic = stel.B_mag(r=r, theta=np.mod(-theta2D+np.pi,2*np.pi), phi=zeta2D, Boozer_toroidal=True, B0=1)
            # Get |B| on surface using VMEC
            f = netcdf.netcdf_file('wout_'+name+'r'+str(r)+'.nc','r',mmap=False)
            xn_nyq = f.variables['xn_nyq'][()]
            xm_nyq = f.variables['xm_nyq'][()]
            ns = f.variables['ns'][()]
            bmnc = f.variables['bmnc'][()]
            lasym = f.variables['lasym__logical__'][()]
            iotaVMEC = f.variables['iotaf'][()][0]
            print('iota Qic =',stel.iota,', iota VMEC =',iotaVMEC)
            if lasym==1:
                bmns = f.variables['bmns'][()]
            else:
                bmns = 0*bmnc
            iradius = ns-1
            bVMEC = np.zeros([Ntheta,Nzeta])
            for imode in range(len(xn_nyq)):
                angle = xm_nyq[imode]*theta2D - xn_nyq[imode]*zeta2D
                bVMEC += bmnc[iradius,imode]*np.cos(angle) + bmns[iradius,imode]*np.sin(angle)
            # Find RMS difference
            bDiff_VMEC = (bVMEC - bQic)**2
            b_RMS_VMEC_diff.append(np.sqrt(np.sum(bDiff_VMEC) / Nzeta / Ntheta))
            # Get |B| on surface using BOOZ_XFORM
            f = netcdf.netcdf_file('boozmn_'+name+'r'+str(r)+'.nc',mmap=False)
            bmnc = f.variables['bmnc_b'][()]
            xn = f.variables['ixn_b'][()]
            xm = f.variables['ixm_b'][()]
            isurf = bmnc.shape[0]-1
            bBOOZ = np.zeros([Ntheta,Nzeta])
            B0_Booz_temp = np.zeros([Ntheta,Nzeta])
            for imn in range(len(xm)):
                angle = xm[imn] * theta2D - xn[imn] * zeta2D
                bBOOZ = bBOOZ + bmnc[isurf,imn] * np.cos(angle)
                if xm[imn]==0:
                    B0_Booz_temp = B0_Booz_temp + (bmnc[0,imn] * 1.5 - bmnc[1,imn] * 0.5) * np.cos(angle)
            normalization=np.mean(B0_Booz_temp)
            B0_Booz = B0_Booz_temp[0]
            # B0_Booz = normalization * (1 + stel.B0_vals[1] * np.cos(zeta2D))
            B1_Booz = bBOOZ - B0_Booz
            B0vmec,B1cvmec,B1svmec,B20vmec,B2cvmec,B2svmec,iotaVMECt = stel.from_boozxform(
                booz_xform_file='boozmn_'+name+'r'+str(r)+'.nc', max_s_for_fit = 0.8,
                N_phi = stel.nphi, max_n_to_plot = 2, vmec_file='wout_'+name+'r'+str(r)+'.nc',
                nNormal=stel.iota-stel.iotaN, input_stel=stel, show=False, savefig=False)
            plt.figure()
            plt.plot(stel.varphi, B0vmec, label='B0 from pyQics from_boozxform')
            plt.plot(zeta, B0_Booz_temp[0], label='B0 from BOOZ in this script')
            plt.plot(zeta, normalization * (1 + stel.B0_vals[1] * np.cos(stel.nfp*zeta2D))[0], label='B0 analytical')
            plt.legend()
            plt.title('r='+str(r))
            plt.savefig('B0_diff_r'+str(r)+'.pdf')
            plt.figure()
            b1Qic = stel.B_mag(r=r, theta=np.mod(-theta2D+np.pi,2*np.pi), phi=zeta2D, Boozer_toroidal=True, B0=0)
            plt.contour(zeta2D,theta2D,b1Qic,10,cmap='plasma', linestyles = 'dashed')
            plt.contour(zeta2D,theta2D,B1_Booz,10,cmap='viridis')
            plt.title('r='+str(r))
            plt.savefig('b1Qic_B1booz_diff_r'+str(r)+'.pdf')
            # Find RMS difference
            bDiff_BOOZ = (bBOOZ - bQic)**2
            b_RMS_BOOZ_diff.append(np.sqrt(np.sum(bDiff_BOOZ) / Nzeta / Ntheta))
            f.close()
        # Create fitting functions
        aspect_ratio = 1/r_edge
        quadratic_function = lambda x,a: a/x/x

        # Check quadratic scaling
        popt, _ = curve_fit(quadratic_function, aspect_ratio, b_RMS_BOOZ_diff)
        y_quadratic = quadratic_function(aspect_ratio,popt[0])
        plt.figure()
        plt.plot(aspect_ratio, b_RMS_BOOZ_diff, 'r*', label=r'RMS($B_{\mathrm{VMEC}}-B_{\mathrm{construction}}$) [T]')
        plt.plot(aspect_ratio, y_quadratic, 'b-', label=r'$\propto 1/A^2$')
        plt.yscale('log') 
        plt.xscale('log')
        plt.xlabel('Aspect Ratio A')
        plt.legend()
        plt.savefig('RMS_VMEC.png')