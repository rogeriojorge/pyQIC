from qic import Qic
def optimized_configuration_nfp2(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.0891900098107225,0.0,0.009073213236900652,0.0,-0.0005070944387328263]
    zs      = [0.0,0.0,-0.07985550600916536,0.0,0.011013352147885291,0.0,-0.00031790031138316565]
    B0_vals = [1.0,0.1911414662451819]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.580044972561008,-0.16173713638269638,-0.07204934493020146]
    delta   = 0.1
    d_svals = []
    nfp     = 2
    iota    = -1.1586237546283766
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [-0.27271824369578124,1.408962185789325,0.029550485422345667]
    X2c_svals = [0.0,1.3985747409931824,1.9696375265576096,-0.05425156701114692]
    p2      = 0.0
    # B20QI_deviation_max = 0.004651410350959201
    # B2cQI_deviation_max = 1.413483624551122
    # B2sQI_deviation_max = 0.010928868241634526
    # Max |X20| = 3.4744861695464038
    # Max |Y20| = 6.689669100169661
    # gradgradB inverse length: 5.253942815139203
    # d2_volume_d_psi2 = 328.01465854796237
    # max curvature_d(0) = 0.49609240811012806
    # max d_d(0) = 0.17194520733758895
    # max gradB inverse length: 3.073663320419462
    # Max elongation = 7.002220358829136
    # Initial objective = 14.949194326703394
    # Final objective = 14.834935276230924
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)