from qic import Qic
def optimized_configuration_nfp1(nphi=131,order = "r2"):
    rc      = [1.0,0.0,-0.37537290728836,0.0,0.06404308000119187,0.0,-0.005726157394012481]
    zs      = [0.0,0.0,-0.2554856289793238,0.0,0.05035951117462987,0.0,-0.004456641189208413]
    B0_vals = [1.0,0.18565683157242402]
    omn_method = "non-zone-fourier"
    k_buffer = 1
    p_buffer = 2
    d_over_curvature_cvals = [0.6272470054082003,-0.02079325174724665,-0.01805265418147066]
    delta   = 0.1
    d_svals = []
    nfp     = 1
    iota    = -0.5558710657417609
    X2s_svals = []
    X2c_cvals = [0]
    X2s_cvals = [0.000911589607133522,-0.00525431485777589,-0.030101373928577653,-0.08686267828253884,-0.18833972139975164,-0.35841469175142443,-0.6237416175526145,-0.9811736184171744,-1.3806416826242953,-1.7454299477759898,-2.0084058555631996,-2.1354415950327437,-2.1279381605210745,-2.012207883178431,-1.8256706165123886,-1.6057295359689578,-1.3831463718112667,-1.179473601710811,-1.0072627328745065,-0.8717169456066395,-0.772756628691107,-0.7068627226230748,-0.6684182204077171,-0.650526021463673,-0.6454380273579504,-0.6448060920107836,-0.6399812043202091,-0.6225513401909114,-0.5852167235284964,-0.5229488108619365,-0.43418506724093486,-0.3216434614003315,-0.19231292098030442,-0.056382712733746294,0.07471768665711395,0.19058827046498614,0.2837055835884865,0.3503244707338708,0.3903425856260858,0.4064230473520178,0.40283141644687076,0.38437447498586114,0.35564627850277153,0.3206134106385287,0.2824679243742635,0.24364655992281992,0.20593365495937813,0.17060583994419348,0.138618824858121,0.11086501962236732,0.08852831128421267,0.07350921290950074,0.06877514086450551,0.07832251190962347,0.1063050243428656,0.15498329602443317,0.2217677635145118,0.29684913648199296,0.36393422262188985,0.40535032042970187,0.40833105661292207,0.36699021639001733,0.28183286474364727,0.16620377273724943,0.056031413914838964,0.0006104668449231266,0.029248207920803508,0.126459163163626,0.24540667919836015,0.3431946900990628,0.39959774931274616,0.4110345242106841,0.381368422970633,0.3210113699177623,0.24657587143147702,0.17555728564049194,0.12022976287419786,0.08542487522097013,0.07015455017042496,0.07060837906301813,0.08258641630405146,0.10274259507152549,0.12884867015185872,0.15952989878251603,0.19384100502122792,0.2308805114873232,0.2695022744127205,0.3081089790271191,0.34449528090612946,0.37572852035000415,0.39809421537141654,0.4071766377499266,0.3981735773475692,0.3665339937236587,0.30893000204332804,0.22442100960177527,0.11547745113219192,-0.011590222514245033,-0.1471752635008645,-0.27994550476598873,-0.3990502680479583,-0.496262948061617,-0.5673900111749773,-0.6126252083487318,-0.6359654034151759,-0.6440908694989954,-0.6451570183839924,-0.6478094647302327,-0.6605383779616467,-0.6913176044796546,-0.7473620813691705,-0.8347833902954772,-0.9579216469811133,-1.1181857968133828,-1.312361424044641,-1.5305679400309582,-1.7543730259409915,-1.9559655890777672,-2.099642324196637,-2.1469670274129835,-2.0664486430491857,-1.84687083948364,-1.5099215850474335,-1.113251396272318,-0.7342070086953013,-0.43549388377307263,-0.23587950193394921,-0.11484099323937629,-0.044944420371418624,-0.010717570802817298,1.4408865069663647e-05]
    X2c_svals = [-0.04705694098610795,0.0676965864598738,0.22081572266188954,0.44356113940379993,0.7342606325792292,1.1246333808184668,1.6295476965589724,2.184798904505872,2.6652632781069334,2.9590173285130197,3.0204230224351183,2.8723642804844047,2.577867644415905,2.208971768732576,1.8265424793495926,1.4722325145916162,1.168701996574934,0.9238224595650226,0.7358391406326945,0.597852399679344,0.5010090761270408,0.4364067328785233,0.3959923004175215,0.3727997698510191,0.3608255010622599,0.3547596786752749,0.34972140172884836,0.3410987309777773,0.32456900609541944,0.2963478937887811,0.25366092660978706,0.19533645387089854,0.12231197831208829,0.03779727456387897,-0.0530836032849348,-0.1441473336426104,-0.2292572324558102,-0.30332823575528534,-0.3629497536975715,-0.406532272243261,-0.4340672006416629,-0.4466837018326844,-0.4461801907012467,-0.43464344386010073,-0.41419444792466253,-0.3868497026780336,-0.3544722774855902,-0.31881102367165937,-0.2816866441291,-0.2454699300384708,-0.21407600511179453,-0.19468841605525974,-0.20018612880808295,-0.2516028879959259,-0.37883942620445316,-0.6166717845852177,-0.9932459781578457,-1.5121202046698758,-2.1371590739575237,-2.794252868320132,-3.388206155538382,-3.802350992443456,-3.863123304761057,-3.3428081325934094,-2.0980516522811055,-0.1601665181361863,1.5340002617981883,3.00884091637324,3.765138977379208,3.8726707342292195,3.5532328833659474,3.004338215692824,2.357056761623647,1.7116856363336908,1.1512811717399427,0.7259124998191391,0.44414735761307267,0.2838989576797628,0.2109175031848579,0.19290442207628938,0.20578650300932022,0.23420278006718448,0.2693651166173002,0.30649987607737283,0.34286457648575697,0.3765286536260565,0.40575051917153554,0.42871482143525097,0.4434547188705142,0.4478735363918765,0.43985109050445415,0.41745660312617855,0.3792885308846202,0.3249225474132714,0.2553803889314755,0.17345959044744336,0.08373266254395831,-0.007917203474327787,-0.09519704241490272,-0.17251550772104554,-0.2359575709597297,-0.28382281552679234,-0.3166243084779853,-0.3366690205138568,-0.34746621963375496,-0.35319866081268053,-0.3583947680158654,-0.3678355664928895,-0.3866618696970231,-0.4206139063600666,-0.47631221563695614,-0.5614506240079031,-0.6847087259405694,-0.8551109524130608,-1.0804952290533405,-1.364770486036393,-1.7038358931390538,-2.080523195101173,-2.4598103537493277,-2.7868465461893956,-2.9917271447450715,-3.005374122795073,-2.7877687726467277,-2.359787940898271,-1.8146064951075154,-1.2815207809903093,-0.8514075464951375,-0.5318984999307901,-0.2880148529088276,-0.11035930960312798,0.00037177599958616213]
    p2      = 0.0
    # B20QI_deviation_max = 0.0005806187415919872
    # B2cQI_deviation_max = 1.7039702981946903e-12
    # B2sQI_deviation_max = 0.000492028171457437
    # Max |X20| = 3.1701146799678974
    # Max |Y20| = 7.047186551631843
    # gradgradB inverse length: 4.626018304223949
    # d2_volume_d_psi2 = 278.4566978203537
    # max curvature_d(0) = 0.9226009562492274
    # max d_d(0) = 0.54287873228956
    # max gradB inverse length: 1.6065682554645502
    # Max elongation = 4.048662872240823
    return Qic(omn_method = omn_method, delta=delta, p_buffer=p_buffer, p2=p2, k_buffer=k_buffer, rc=rc,zs=zs, nfp=nfp, B0_vals=B0_vals, d_svals=d_svals, nphi=nphi, omn=True, B2c_cvals=X2c_cvals, B2s_svals=X2s_svals, order=order, d_over_curvature_cvals=d_over_curvature_cvals, B2c_svals=X2c_svals, B2s_cvals=X2s_cvals)