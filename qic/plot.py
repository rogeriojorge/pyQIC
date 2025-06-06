"""
This module contains a function to plot a near-axis surface.
"""

import numpy as np
from scipy.interpolate import interp2d, interp1d
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as clr
from matplotlib.colors import LightSource
import matplotlib.ticker as tck
from .util import to_Fourier

def plot(self, newfigure=True, show=True, savefig=None):
    """
    Generate a matplotlib figure with an array of plots, showing the
    toroidally varying properties of the configuration.

    Args:
        newfigure: Whether to create a new matplotlib figure.
        show: Whether to call matplotlib's ``show()`` function after making the plots.
    """
    if newfigure:
        f = plt.figure(figsize=(14, 7))
    plt.rcParams.update({'font.size': 6})
    if self.order == 'r1':
        if self.omn:
            nrows = 4
            ncols = 6
        else:
            nrows = 3
            ncols = 6
    elif self.order == 'r2':
        nrows = 4
        ncols = 8
        if self.omn:
            nrows=6
    elif self.order == 'r3':
        nrows = 6
        ncols = 8
    else:
        raise RuntimeError('Should not get here')
    jplot = 1

    def subplot(title, data=None, y0=False):
        """
        data is assumed to correspond to title, unless specified otherwise.
        Set y0 to True to avoid suppressed 0 on the y axis.
        """
        nonlocal jplot
        if data is None:
            data = eval('self.' + title)
        plt.subplot(nrows, ncols, jplot)
        jplot = jplot + 1
        plt.plot(self.phi, data, label=title)
        plt.xlabel(r'$\phi$')
        plt.title(title)
        if y0:
            plt.ylim(bottom=0)
        plt.xlim((0, self.phi[-1]))

    subplot('R0')
    subplot('Z0')
    subplot('R0p')
    subplot('Z0p')
    subplot('R0pp')
    subplot('Z0pp')
    subplot('R0ppp')
    subplot('Z0ppp')
    subplot('curvature')
    subplot('torsion')
    subplot('sigma')
    subplot('X1s')
    subplot('X1c')
    subplot('Y1c')
    subplot('Y1s')
    subplot('elongation', y0=True)
    subplot('L_grad_B', y0=True)
    subplot('1/L_grad_B', data=self.inv_L_grad_B)
    # if self.omn:
    #     subplot('alpha')
    if self.order != 'r1':
        jplot -= 2
        subplot('L_grad_grad_B')
        plt.title('scale lengths')
        plt.legend(loc=0, fontsize=5)
        plt.ylim(0, max(max(self.L_grad_B), max(self.L_grad_grad_B)))
        
        subplot('1/L_grad_grad_B', self.grad_grad_B_inverse_scale_length_vs_varphi)
        plt.title('inv scale lengths')
        plt.legend(loc=0, fontsize=5)
        plt.ylim(0, max(max(self.inv_L_grad_B), max(self.grad_grad_B_inverse_scale_length_vs_varphi)))
        
        subplot('V1')
        subplot('V2')
        subplot('V3')
        subplot('X20')
        subplot('X2c')
        subplot('X2s')
        subplot('Y20')
        subplot('Y2c')
        subplot('Y2s')
        subplot('Z20')
        subplot('Z2c')
        subplot('Z2s')
        data = self.r_singularity_vs_varphi
        data[data > 1e20] = np.NAN
        subplot('r_singularity', data=data, y0=True)
    if self.omn:
        subplot('alpha')
        jplot -= 1
        subplot('alpha_no_buffer')
        plt.title('alpha')
        plt.legend(loc=0, fontsize=5)

        plt.subplot(nrows, ncols, jplot)
        d_alpha_iota_d_varphi_der    = [self.d_alpha_iota_d_varphi]
        d_alpha_notIota_d_varphi_der = [self.d_alpha_notIota_d_varphi]
        d_alpha_der = [d_alpha_iota_d_varphi_der[0] * self.iota + d_alpha_notIota_d_varphi_der[0]]
        ders = [1]
        plt.plot(d_alpha_der[0], label='n=1')
        plt.xlabel(r'$\phi$')
        plt.title(r'$\alpha^{(n)}$')
        self.d_alpha_der_diff = [d_alpha_der[0][-1]-d_alpha_der[0][0]]
        for n in range(2,5):
            d_alpha_iota_d_varphi_der.append(np.matmul(self.d_d_varphi,d_alpha_iota_d_varphi_der[n-2]))
            d_alpha_notIota_d_varphi_der.append(np.matmul(self.d_d_varphi,d_alpha_notIota_d_varphi_der[n-2]))
            d_alpha_der.append(d_alpha_iota_d_varphi_der[n-1] * self.iota + d_alpha_notIota_d_varphi_der[n-1])
            self.d_alpha_der_diff.append(d_alpha_der[n-1][-1]-d_alpha_der[n-1][0])
            ders.append(n)
            plt.plot(d_alpha_der[n-1], label='n='+str(n))
        plt.legend()
        jplot += 1

        plt.subplot(nrows, ncols, jplot)
        plt.plot(ders,self.d_alpha_der_diff)
        plt.xlabel(r'$n=$Order of the derivative')
        plt.title(r'$\alpha^{(n)}(2\pi)-\alpha^{(n)}(0)$')
        jplot += 1
    if self.order != 'r1':
        subplot('B20')
        if self.omn:
            # jplot -= 1
            # subplot('B2QI_exact')
            # plt.legend(loc=0, fontsize=5)
            subplot('B2cQI')
            subplot('B2sQI')
            # jplot -= 1
            # subplot('B2QI_exact')
            # plt.legend(loc=0, fontsize=5)
            subplot('B20QI_deviation')
            subplot('B2cQI_deviation')
            subplot('B2sQI_deviation')
            subplot('B2c_array')
            subplot('B2s_array')
    if self.omn:
        subplot('d')
        subplot('gamma')
    if self.order != 'r1':
        if self.order != 'r2':
            subplot('X3c1')
            subplot('X3s1')
            subplot('Y3c1')
            subplot('Y3s1')

    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig+'.pdf')
    if show:
        plt.show()

def set_axes_equal(ax):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Args:
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def create_subplot(ax, x_2D_plot, y_2D_plot, z_2D_plot, colormap, elev=90, azim=45, dist=7, **kwargs):
    '''
    Construct the surface given a surface in cartesian coordinates
    x_2D_plot, y_2D_plot, z_2D_plot already with phi=[0,2*pi].
    A matplotlib figure with elements fig, ax
    must have been previously created.

    Args:
        ax: matplotlib figure instance
        x_2d_plot: 2D array for the x coordinates of the surface
        y_2d_plot: 2D array for the x coordinates of the surface
        z_2d_plot: 2D array for the x coordinates of the surface
        elev: elevation angle for the camera view
        azim: azim angle for the camera view
        distance: distance parameter for the camera view
    '''
    ax.plot_surface(x_2D_plot, y_2D_plot, z_2D_plot, facecolors=colormap,
                    rstride=1, cstride=1, antialiased=False,
                    linewidth=0, alpha=1., shade=False, **kwargs)
    set_axes_equal(ax)
    ax.set_axis_off()
    ax.dist = dist
    ax.elev = elev
    ax.azim = azim

def create_field_lines(qic, alphas, X_2D, Y_2D, Z_2D, phimax=2*np.pi, nphi=500):
    '''
    Function to compute the (X, Y, Z) coordinates of field lines at
    several alphas, where alpha = theta-iota*varphi with (theta,varphi)
    the Boozer toroidal angles. This function relies on a 2D interpolator
    from the scipy library to smooth out the lines

    Args:
      qic: instance of self
      alphas: array of field line labels alpha
      X_2D: 2D array for the x components of the surface
      Y_2D: 2D array for the y components of the surface
      Z_2D: 2D array for the z components of the surface
      phimax: maximum value for the field line following angle phi
      nphi: grid resolution for the output fieldline
    '''
    phi_array = np.linspace(0,phimax,nphi,endpoint=False)
    fieldline_X = np.zeros((len(alphas),nphi))
    fieldline_Y = np.zeros((len(alphas),nphi))
    fieldline_Z = np.zeros((len(alphas),nphi))
    [ntheta_RZ,nphi_RZ] = X_2D.shape
    phi1D   = np.linspace(0,2*np.pi,nphi_RZ)
    theta1D = np.linspace(0,2*np.pi,ntheta_RZ)
    X_2D_spline = interp2d(phi1D, theta1D, X_2D, kind='cubic')
    Y_2D_spline = interp2d(phi1D, theta1D, Y_2D, kind='cubic')
    Z_2D_spline = interp2d(phi1D, theta1D, Z_2D, kind='cubic')
    for i in range(len(alphas)):
        for j in range(len(phi_array)):
            phi_mod = np.mod(phi_array[j],2*np.pi)
            varphi0=qic.nu_spline(phi_array[j])+2*phi_array[j]-phi_mod
            theta_fieldline=qic.iota*varphi0+alphas[i]
            theta_fieldline_mod=np.mod(theta_fieldline,2*np.pi)
            fieldline_X[i,j] = X_2D_spline(phi_mod,theta_fieldline_mod)[0]
            fieldline_Y[i,j] = Y_2D_spline(phi_mod,theta_fieldline_mod)[0]
            fieldline_Z[i,j] = Z_2D_spline(phi_mod,theta_fieldline_mod)[0]
    return fieldline_X, fieldline_Y, fieldline_Z

def create_subplot_mayavi(mlab, R, alphas, x_2D_plot, y_2D_plot, z_2D_plot,
                          fieldline_X, fieldline_Y, fieldline_Z,
                          Bmag, degrees_array_x, degrees_array_z, shift_array):
    '''
    Plotting routine for a mayavi figure instance that plots
    both the surface and the field lines together. The number
    of surfaces to plot is specified by the length of the
    array degrees_array_x

    Args:
      mlab: mayavi package
      R: scipy rotation vector package
      alphas: array of field line labels alpha
      x_2D_plot: 2D array for the x components of the surface
      y_2D_plot: 2D array for the y components of the surface
      z_2D_plot: 2D array for the z components of the surface
      fieldline_X: 2D array for the x components of the field line
      fieldline_Y: 2D array for the x components of the field line
      fieldline_Z: 2D array for the x components of the field line
      Bmag: 2D array for the magnetic field modulus on the (theta,phi) meshgrid
      degrees_array_x: 1D array with the rotation angles in the x direction for each surface
      degrees_array_z: 1D array with the rotation angles in the z direction for each surface
      shift_array: 1D array with a shift in the y direction for each surface
    '''
    assert len(degrees_array_x) == len(degrees_array_z) == len(shift_array)
    for i in range(len(degrees_array_x)):
        # The surfaces and field lines are rotated first in the
        # z direction and then in the x direction
        rx= R.from_euler('x', degrees_array_x[i], degrees=True)
        rz= R.from_euler('z', degrees_array_z[i], degrees=True)
        # Initialize rotated arrays
        x_2D_plot_rotated = np.zeros((x_2D_plot.shape[0],x_2D_plot.shape[1]))
        y_2D_plot_rotated = np.zeros((x_2D_plot.shape[0],x_2D_plot.shape[1]))
        z_2D_plot_rotated = np.zeros((x_2D_plot.shape[0],x_2D_plot.shape[1]))
        fieldline_X_rotated = np.zeros((fieldline_X.shape[0],fieldline_X.shape[1]))
        fieldline_Y_rotated = np.zeros((fieldline_X.shape[0],fieldline_X.shape[1]))
        fieldline_Z_rotated = np.zeros((fieldline_X.shape[0],fieldline_X.shape[1]))
        # Rotate surfaces
        for th in range(x_2D_plot.shape[0]):
            for ph in range(x_2D_plot.shape[1]):
                [x_2D_plot_rotated[th,ph], y_2D_plot_rotated[th,ph], z_2D_plot_rotated[th,ph]] = rx.apply(rz.apply(np.array([x_2D_plot[th,ph], y_2D_plot[th,ph], z_2D_plot[th,ph]])))
        # Rotate field lines
        for th in range(fieldline_X.shape[0]):
            for ph in range(fieldline_X.shape[1]):
                [fieldline_X_rotated[th,ph], fieldline_Y_rotated[th,ph], fieldline_Z_rotated[th,ph]] = rx.apply(rz.apply(np.array([fieldline_X[th,ph], fieldline_Y[th,ph], fieldline_Z[th,ph]])))
        # Plot surfaces
        mlab.mesh(x_2D_plot_rotated, y_2D_plot_rotated-shift_array[i], z_2D_plot_rotated, scalars=Bmag, colormap='viridis', opacity=1.0)
        # Plot field lines
        for j in range(len(alphas)):
            mlab.plot3d(fieldline_X_rotated[j], fieldline_Y_rotated[j]-shift_array[i], fieldline_Z_rotated[j], color=(0,0,0), line_width=0.002, tube_radius=0.008)

def get_boundary(self, r=0.1, ntheta=40, nphi=130, ntheta_fourier=20, mpol=13, ntor=25):
    '''
    Function that, for a given near-axis radial coordinate r, outputs
    the [X,Y,Z,R] components of the boundary. The resolution along the toroidal
    angle phi is equal to the resolution nphi for the axis, while ntheta
    is specified by the used.

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the poloidal angle.
      nphi   (int): Number of grid points to plot in the toroidal angle.
      ntheta_fourier (int): Resolution in the Fourier transform to cylindrical coordinates
      mpol: resolution in poloidal Fourier space
      ntor: resolution in toroidal Fourier space
    '''
    # Get surface shape at fixed off-axis toroidal angle phi
    R_2D, Z_2D, _ = self.Frenet_to_cylindrical(r, ntheta=ntheta_fourier)
    # Get Fourier coefficients in order to plot with arbitrary resolution
    RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, self.nfp, mpol=mpol, ntor=ntor, lasym=self.lasym)
    if not self.lasym:
        RBS = np.zeros((int(2*ntor+1),int(mpol+1)))
        ZBC = np.zeros((int(2*ntor+1),int(mpol+1)))

    theta1D = np.linspace(0, 2*np.pi, ntheta)
    phi1D = np.linspace(0, 2*np.pi, nphi)
    phi2D, theta2D = np.meshgrid(phi1D, theta1D)
    R_2Dnew = np.zeros((ntheta, nphi))
    Z_2Dnew = np.zeros((ntheta, nphi))
    for m in range(mpol + 1):
        for n in range(-ntor, ntor + 1):
            angle = m * theta2D - n * self.nfp * phi2D
            R_2Dnew += RBC[n+ntor,m] * np.cos(angle) + RBS[n+ntor,m] * np.sin(angle)
            Z_2Dnew += ZBC[n+ntor,m] * np.cos(angle) + ZBS[n+ntor,m] * np.sin(angle)

    # X, Y, Z arrays for the whole surface
    x_2D_plot = R_2Dnew * np.cos(phi1D)
    y_2D_plot = R_2Dnew * np.sin(phi1D)
    z_2D_plot = Z_2Dnew

    return x_2D_plot, y_2D_plot, z_2D_plot, R_2Dnew

def plot_boundary(self, r=0.1, ntheta=80, nphi=150, ntheta_fourier=20, nsections=8,
         fieldlines=False, savefig=None, colormap=None, azim_default=None, threeD=True, n_field_lines=1,
         show=True, **kwargs):
    """
    Plot the boundary of the near-axis configuration. There are two main ways of
    running this function.

    If ``fieldlines=False`` (default), 2 matplotlib figures are generated:

        - A 2D plot with several poloidal planes at the specified radius r with the
          corresponding location of the magnetic axis.

        - A 3D plot with the flux surface and the magnetic field strength
          on the surface.

    If ``fieldlines=True``, both matplotlib and mayavi are required, and
    the following 2 figures are generated:

        - A 2D matplotlib plot with several poloidal planes at the specified radius r with the
          corresponding location of the magnetic axis.

        - A 3D mayavi figure with the flux surface the magnetic field strength
          on the surface and several magnetic field lines.

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the poloidal angle.
      nphi   (int): Number of grid points to plot in the toroidal angle.
      ntheta_fourier (int): Resolution in the Fourier transform to cylindrical coordinates
      nsections (int): Number of poloidal planes to show.
      fieldlines (bool): Specify if fieldlines are shown. Using mayavi instead of matplotlib due to known bug https://matplotlib.org/2.2.2/mpl_toolkits/mplot3d/faq.html
      savefig (str): Filename prefix for the png files to save.
        Note that a suffix including ``.png`` will be appended.
        If ``None``, no figure files will be saved.
      colormap (cmap): Custom colormap for the 3D plots
      azim_default: Default azimuthal angle for the three subplots in the 3D surface plot
      show: Whether or not to call the matplotlib/mayavi ``show()`` command.
      kwargs: Any additional key-value pairs to pass to matplotlib's plot_surface.

    This function generates plots similar to the ones below:

    .. image:: 3dplot1.png
       :width: 200

    .. image:: 3dplot2.png
       :width: 200

    .. image:: poloidalplot.png
       :width: 200
    """
    x_2D_plot, y_2D_plot, z_2D_plot, R_2D_plot = self.get_boundary(r=r, ntheta=ntheta, nphi=nphi, ntheta_fourier=ntheta_fourier)
    phi = np.linspace(0, 2 * np.pi, nphi)  # Endpoint = true and no nfp factor, because this is what is used in get_boundary()
    R_2D_spline = interp1d(phi, R_2D_plot, axis=1)
    z_2D_spline = interp1d(phi, z_2D_plot, axis=1)
    
    # Define a list of colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']

    # Counter to keep track of the color index
    color_index = 0

    # Function to get the next color from the list
    def get_next_color():
        nonlocal color_index
        color = colors[color_index]
        color_index = (color_index + 1) % len(colors)
        return color
    
    ## Poloidal plot
    phi1dplot_RZ = np.linspace(0, 2 * np.pi / self.nfp, nsections, endpoint=False)
    fig_poloidal = plt.figure(figsize=(7, 5), dpi=80)
    ax  = plt.gca()
    for i, phi in enumerate(phi1dplot_RZ):
        phinorm = phi * self.nfp / (2 * np.pi)
        if phinorm == 0:
            label = r'$\phi$=0'
        elif phinorm == 0.125:
            label = r'$\phi={\pi}/$' + str(4 * self.nfp)
        elif phinorm == 0.25:
            label = r'$\phi={\pi}/$' + str(2 * self.nfp)
        elif phinorm == 0.375:
            label = r'$\phi={3\pi}/$' + str(4 * self.nfp)
        elif phinorm == 0.5:
            label = r'$\phi=\pi/$' + str(self.nfp)
        elif phinorm == 0.625:
            label = r'$\phi={5\pi}/$' + str(4 * self.nfp)
        elif phinorm == 0.75:
            label = r'$\phi={3\pi}/$' + str(2 * self.nfp)
        elif phinorm == 0.875:
            label = r'$\phi={7\pi}/$' + str(4 * self.nfp)
        else:
            label = '_nolegend_'
        color = get_next_color()
        # Plot location of the axis
        plt.plot(self.R0_func(phi), self.Z0_func(phi), marker="x", linewidth=2, label=label, color=color)
        # Plot poloidal cross-section
        plt.plot(R_2D_spline(phi), z_2D_spline(phi), color=color)
    plt.xlabel('R (meters)', fontsize=14)
    plt.ylabel('Z (meters)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    plt.legend(loc=2, prop={'size': 8})
    plt.tight_layout()
    ax.set_aspect('equal')
    if savefig != None:
        fig_poloidal.savefig(savefig + '_poloidal.pdf')

    ## 3D plot
    # Set the default azimuthal angle of view in the 3D plot
    # QH stellarators look rotated in the phi direction when
    # azim_default = 0
    if threeD==True:
        if azim_default == None:
            if self.omn == True:
                azim_default = -90
            else:
                if self.helicity == 0:
                    azim_default = 0
                else:
                    azim_default = 45
            # Define the magnetic field modulus and create its theta,phi array
            # The norm instance will be used as the colormap for the surface
            theta1D = np.linspace(0, 2 * np.pi, ntheta)
            phi1D = np.linspace(0, 2 * np.pi, nphi)
            phi2D, theta2D = np.meshgrid(phi1D, theta1D)
            # Create a color map similar to viridis 
            Bmag = self.B_mag(r, theta2D, phi2D)
            norm = clr.Normalize(vmin=Bmag.min(), vmax=Bmag.max())
            if fieldlines==False:
                if colormap==None:
                    # Cmap similar to quasisymmetry papers
                    # cmap = clr.LinearSegmentedColormap.from_list('qs_papers',['#4423bb','#4940f4','#2e6dff','#0097f2','#00bacc','#00cb93','#00cb93','#7ccd30','#fbdc00','#f9fc00'], N=256)
                    cmap = cm.viridis
                    # Add a light source so the surface looks brighter
                    ls = LightSource(azdeg=0, altdeg=10)
                    cmap_plot = ls.shade(Bmag, cmap, norm=norm)
                # Create the 3D figure and choose the following parameters:
                # gsParams: extension in the top, bottom, left right directions for each subplot
                # elevParams: elevation (distance to the plot) for each subplot
                fig = plt.figure(constrained_layout=False, figsize=(4.5, 8))
                gsParams = [[1.02,-0.3,0.,0.85], [1.09,-0.3,0.,0.85], [1.12,-0.15,0.,0.85]]
                elevParams = [90, 30, 5]
                for i in range(len(gsParams)):
                    gs = fig.add_gridspec(nrows=3, ncols=1,
                                        top=gsParams[i][0], bottom=gsParams[i][1],
                                        left=gsParams[i][2], right=gsParams[i][3],
                                        hspace=0.0, wspace=0.0)
                    ax = fig.add_subplot(gs[i, 0], projection='3d')
                    create_subplot(ax, x_2D_plot, y_2D_plot, z_2D_plot, cmap_plot, elev=elevParams[i], azim=azim_default, **kwargs)
                # Create color bar with axis placed on the right
                cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
                m = cm.ScalarMappable(cmap=cmap, norm=norm)
                m.set_array([])
                cbar = plt.colorbar(m, cax=cbar_ax)
                cbar.ax.set_title(r'$|B| [T]$')
                # Save figure
                if savefig != None:
                    fig.savefig(savefig + '3D.png')
                if show:
                    # Show figures
                    plt.show()
            else:
                ## X, Y, Z arrays for the field lines
                # Plot different field lines corresponding to different alphas
                # where alpha=theta-iota*varphi with (theta,varphi) the Boozer angles
                #alphas = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
                alphas = np.linspace(0.7, 2 * np.pi + 0.7, n_field_lines, endpoint=False)
                # Create the field line arrays
                fieldline_X, fieldline_Y, fieldline_Z = create_field_lines(self, alphas, x_2D_plot, y_2D_plot, z_2D_plot)
                # Define the rotation arrays for the subplots
                degrees_array_x = [0., 81.]#[0., -66., 81.] # degrees for rotation in x
                degrees_array_z = [azim_default, azim_default]#[azim_default, azim_default, azim_default] # degrees for rotation in z
                shift_array   = [-0.8, 1.0]#[-0.9, 0.6, 1.8]
                # Import mayavi and rotation packages (takes a few seconds)
                from mayavi import mlab
                from scipy.spatial.transform import Rotation as R
                if show:
                    # Show RZ plot
                    plt.show()
                # Create 3D figure
                fig_3d = mlab.figure(bgcolor=(1,1,1), size=(580,600))
                # Create subplots
                create_subplot_mayavi(mlab, R, alphas, x_2D_plot, y_2D_plot, z_2D_plot,
                                    fieldline_X, fieldline_Y, fieldline_Z,
                                    Bmag, degrees_array_x, degrees_array_z, shift_array)
                # Create a good camera angle
                mlab.view(azimuth=0, elevation=0, distance=8.0, focalpoint=(0,0,0), figure=fig_3d)
                # Create the colorbar and change its properties
                cb = mlab.colorbar(orientation='vertical', title='|B| [T]', nb_labels=7)
                cb.scalar_bar.unconstrained_font_size = True
                cb.label_text_property.font_family = 'times'
                cb.label_text_property.bold = 0
                cb.label_text_property.font_size=20
                cb.label_text_property.color=(0,0,0)
                cb.title_text_property.font_family = 'times'
                cb.title_text_property.font_size=20
                cb.title_text_property.color=(0,0,0)
                cb.title_text_property.bold = 1
                # Save figure
                if savefig != None:
                    mlab.savefig(filename=savefig+'3D_fieldlines.png', figure=fig_3d)
                if show:
                    # Show mayavi plot
                    mlab.show()
                    # Close mayavi plots
                    mlab.close(all=True)
    else:
        plt.show()

def B_fieldline(self, r=0.1, alpha=0, phimax=None, nphi=400, show=True, savefig=None):
    '''
    Plot the modulus of the magnetic field B along a field line with
    the Boozer toroidal angle varphi acting as a field-line following
    coordinate

    Args:
      r (float): near-axis radius r where to create the surface
      alpha (float): Field-line label
      phimax (float): Maximum value of the field-line following parameter varphi
      nphi (int): resolution of the phi grid
      show (bool): Whether or not to call the matplotlib ``show()`` command.
    '''
    if phimax is None:
        phimax = 30 * np.pi / abs(self.iota)
    varphi_array = np.linspace(0, phimax, nphi)
    fig_B_fieldline, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.xlabel(r'$\varphi$')
    plt.ylabel(r'$B(\varphi)$')
    plt.title("r = " + str(r) + ", alpha = " + str(alpha))
    theta = alpha + self.iota * varphi_array
    plt.plot(varphi_array/np.pi, self.B_mag(r, theta, varphi_array, Boozer_toroidal=True))
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter(r'%g $\pi$'))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=2))
    plt.tight_layout()
    if savefig != None:
        fig_B_fieldline.savefig(savefig + '_B_fieldline.pdf')
    if show:
        plt.show()

def B_contour(self, r=0.1, ntheta=100, nphi=120, ncontours=20, B0=1, show=True, savefig=None):
    '''
    Plot contours of constant B, with B the modulus of the
    magnetic field, as a function of Boozer coordinates theta and varphi

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the Boozer poloidal angle.
      nphi   (int): Number of grid points to plot in the Boozer toroidal angle.
      ncontours (int): number of contours to show in the plot
      show (bool): Whether or not to call the matplotlib ``show()`` command.
    '''
    theta_array=np.linspace(0,2*np.pi,ntheta)
    phi_array=np.linspace(0,2*np.pi/self.nfp,nphi)
    theta_2D, phi_2D = np.meshgrid(theta_array,phi_array)
    magB_2D = self.B_mag(r,theta_2D,phi_2D,Boozer_toroidal=True,B0=B0)
    fig_B_contour, ax=plt.subplots(1,1)
    contourplot = ax.contour(phi_2D/np.pi, theta_2D/np.pi, magB_2D, ncontours, cmap=cm.plasma, linewidths=2.0)
    fig_B_contour.colorbar(contourplot)
    # ax.set_title('|B| for r=' + str(r))
    ax.set_xlabel(r'Boozer $\varphi$', fontsize=16)
    ax.set_ylabel(r'Boozer $\theta$', fontsize=16)
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=0.5/self.nfp))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=0.5))
    plt.tight_layout()
    if savefig != None:
        fig_B_contour.savefig(savefig + '_B_contour.pdf')
    if show:
        plt.show()

def B_densityplot(self, r=0.1, ntheta=250, nphi=250, B0=1, show=True, savefig=None):
    '''
    Density plot of B, with B the modulus of the
    magnetic field, as a function of Boozer coordinates theta and varphi

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the Boozer poloidal angle.
      nphi   (int): Number of grid points to plot in the Boozer toroidal angle.
      show (bool): Whether or not to call the matplotlib ``show()`` command.
    '''
    theta_array=np.linspace(0,2*np.pi,ntheta)
    phi_array=np.linspace(0,2*np.pi/self.nfp,nphi)
    theta_2D, phi_2D = np.meshgrid(theta_array,phi_array)
    magB_2D = self.B_mag(r,theta_2D,phi_2D,Boozer_toroidal=True,B0=B0)
    # contourplot = ax.contour(phi_2D/np.pi, theta_2D/np.pi, magB_2D, ncontours, cmap=cm.plasma, linewidths=2.0)
    fig_B_densityploy, ax=plt.subplots(1,1)
    contourplot = ax.imshow(magB_2D.transpose(), extent=[0, 2/self.nfp, 0, 2], cmap=cm.plasma, aspect='auto')
    fig_B_densityploy.colorbar(contourplot)
    ax.set_title('|B| for r=' + str(r))
    ax.set_xlabel(r'$\varphi$')
    ax.set_ylabel(r'$\theta$')
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=0.5/self.nfp))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=0.5))
    plt.tight_layout()
    if savefig != None:
        fig_B_densityploy.savefig(savefig + '_B_densityploy.pdf')
    if show:
        plt.show()

def plot_axis(self, nphi=100, frenet=True, nphi_frenet=80, frenet_factor=0.12, savefig=None, show=True):
    '''
    Plot axis shape and the Frenet-Serret frame along
    the axis (optional). If frenet is true, creates
    a mayavi instance showing the axis and nphi_frenet
    times 3 vectors, corresponding to the tangent, normal and
    binormal vectors. If frenet is false, creates a
    matplotlib instance with only a single axis shape
    curve shown.

    Args:
      nphi (int): Number of grid points in the axis shape
      frenet (bool): True plots the Frenet-Serret frame, False it doesn't
      nphi_frenet (int): Number of Frenet-Serret vectors to show
      frenet_factor (float): Size of Frenet-Serret vectors
      savefig (string): filename to save resulting figure in png format.
        Note that ``.png`` will be appended.
        If ``None``, no figure file will be saved.
      show (bool): Whether or not to call the matplotlib/mayavi ``show()`` command.
    '''
    # Create array of toroidal angles along the axis
    # where the axis points will be created
    phi_array = np.linspace(0, 2 * np.pi, nphi)
    # Calculate the x, y and z components of the axis
    R0 = self.R0_func(phi_array)
    Z0 = self.Z0_func(phi_array)
    x_plot = R0 * np.cos(phi_array)
    y_plot = R0 * np.sin(phi_array)
    z_plot = Z0
    if frenet:
        # Show Frenet-Serret frame
        # Initiate mayavi instance
        from mayavi import mlab
        fig_axis = mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.), size=(650,500))
        # Plot the magnetic axis
        s = mlab.plot3d(x_plot, y_plot, z_plot, color=(0,0,0), line_width=0.001, tube_radius=0.01)
        # Show the x,y,z axis
        ax = mlab.axes(s,xlabel=r'X [m]',ylabel=r'Y [m]',zlabel=r'Z [m]',line_width=1.0,nb_labels=4)
        ax.axes.font_factor = 1.3
        ax.axes.label_format = '    %4.2f'
        ax.label_text_property.bold = False
        ax.label_text_property.italic = False
        # Create array of toroidal angles where the Frenet-Serret is shown
        phi_array = np.linspace(0, 2 * np.pi, nphi_frenet)
        # Calculate origin and vector arrays for the Frenet-Serret frame
        R0 = self.R0_func(phi_array)
        Z0 = self.Z0_func(phi_array)
        x_plot = R0*np.cos(phi_array)
        y_plot = R0*np.sin(phi_array)
        z_plot = Z0
        # Normal vector (red)
        normal_R   = self.normal_R_spline(phi_array)
        normal_phi = self.normal_phi_spline(phi_array)
        normal_Z   = self.normal_z_spline(phi_array)
        normal_X   = normal_R * np.cos(phi_array) - normal_phi * np.sin(phi_array)
        normal_Y   = normal_R * np.sin(phi_array) + normal_phi * np.cos(phi_array)
        mlab.quiver3d(x_plot, y_plot, z_plot,
                      normal_X, normal_Y, normal_Z,
                      scale_factor=frenet_factor,
                      color=(1, 0, 0), reset_zoom=False, name='Normal')
        # Biormal vector (blue)
        binormal_R   = self.binormal_R_spline(phi_array)
        binormal_phi = self.binormal_phi_spline(phi_array)
        binormal_Z   = self.binormal_z_spline(phi_array)
        binormal_X   = binormal_R * np.cos(phi_array) - binormal_phi * np.sin(phi_array)
        binormal_Y   = binormal_R * np.sin(phi_array) + binormal_phi * np.cos(phi_array)
        mlab.quiver3d(x_plot, y_plot, z_plot,
                      binormal_X, binormal_Y, binormal_Z,
                      scale_factor=frenet_factor,
                      color=(0, 0, 1), reset_zoom=False, name='Binormal')
        # Tangent vector (green)
        tangent_R   = self.tangent_R_spline(phi_array)
        tangent_phi = self.tangent_phi_spline(phi_array)
        tangent_Z   = self.tangent_z_spline(phi_array)
        tangent_X   = tangent_R * np.cos(phi_array) - tangent_phi * np.sin(phi_array)
        tangent_Y   = tangent_R * np.sin(phi_array) + tangent_phi * np.cos(phi_array)
        mlab.quiver3d(x_plot, y_plot, z_plot,
                      tangent_X, tangent_Y, tangent_Z,
                      scale_factor=frenet_factor,
                      color=(0, 1, 0),
                      reset_zoom=False, name='Tangent')
        if self.omn: # Show where curvature is zero if in quasi-isodynamism
            # zero_index = np.where(np.diff(self.sign_curvature_change)==-2)[0][0]
            # phi_at_zero = self.phi[zero_index]
            for i in range(2 * self.nfp):
                phi_at_zero = np.pi * i / self.nfp
                R0_at_zero = self.R0_func(phi_at_zero)
                Z0_at_zero = self.Z0_func(phi_at_zero)
                x_at_zero = R0_at_zero * np.cos(phi_at_zero)
                y_at_zero = R0_at_zero * np.sin(phi_at_zero)
                z_at_zero = Z0_at_zero
                mlab.points3d(x_at_zero, y_at_zero, z_at_zero, color=(0.9,0.9,0.9),scale_factor=0.09,figure=fig_axis)
        # Plot legends
        # Mayavi does not have its own legend function
        # Save figure
        if savefig != None:
            mlab.savefig(savefig + '.png')
        if show:
            # Show figure
            mlab.show()
    else:
        # Do not show Frenet-Serret frame
        # Initiate matplotlib instance
        fig_axis = plt.figure(figsize=(6, 5))
        ax = plt.axes(projection='3d')
        # Plot the magnetic axis
        plt.plot(x_plot, y_plot, z_plot)
        set_axes_equal(ax)
        ax.grid(False)
        ax.set_xlabel('X [m]', fontsize=10)
        ax.set_ylabel('Y [m]', fontsize=10)
        ax.set_zlabel('Z [m]', fontsize=10)
        plt.tight_layout()
        fig_axis.subplots_adjust(left=-0.05, top=1.05)
        # Save figure
        if savefig != None:
            fig_axis.savefig(savefig + '.pdf')
        if show:
            # Show figure
            plt.show()
