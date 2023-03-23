from qic import Qic
def optimized_configuration_nfp3(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.038392826984787465,0.0,0.0031900007967624825,0.0,-0.0001292785141336116]
    zs      = [0.0,0.0,-0.032705279851039104,0.0,0.0038796364284059562,0.0,-6.976585807351741e-05]
    B0_vals = [1.0,0.19128653965591466]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.6996215427705272,-0.26839194225777063,-0.04927968019298633]
    delta   = 0.1
    d_svals = []
    nfp     = 3
    iota    = -1.1875322962059593
    X2s_svals = []
    X2c_cvals = [0.0]
    X2s_cvals = [-0.011131410883046108,3.3005667404508596,-1.5566823895346698]
    X2c_svals = [0.0,2.4980533005569265,0.9707302905374766,-0.4836693453391415]
    p2      = 0.0
    # B20QI_deviation_max = 0.0030309864906531425
    # B2cQI_deviation_max = 1.158247715264283
    # B2sQI_deviation_max = 0.007827501635547573
    # Max |X20| = 5.240110790387989
    # Max |Y20| = 6.620593045440855
    # gradgradB inverse length: 6.918894201047701
    # d2_volume_d_psi2 = 242.64577274930136
    # max curvature_d(0) = 0.9687964320342839
    # max d_d(0) = 0.37041956419361205
    # max gradB inverse length: 4.126562899000218
    # Max elongation = 5.754014078512894
    # Initial objective = 14.437761691201814
    # Final objective = 14.437796856280386
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)