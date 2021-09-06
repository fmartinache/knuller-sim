''' ----------------------------------------------------------------
Python module primarily designed to simulate simple observations
with a 4-beam nuller at the VLTI.

The module introduces one major Nuller() class and some additional
tools that compute outputs and trace several plots that seem to be
useful to look at when designing an observation with a ground-based
nuller.

It is primarily designed to investigate the properties of VIKiNG,
but without too much trouble, the Nuller class should be usable
in slightly different contexts (SAM fed nulller, spinning space
interferometers, ...)

Frantz Martinache (2021).
---------------------------------------------------------------- '''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.io.fits as pf
from scipy.optimize import leastsq

dtor = np.pi/180

vlti_ut = np.array([[-9.925,  14.887, 44.915, 103.306],
                    [-20.335, 30.502,  66.183,  44.999]])
vlti_lat = -24.62794830  # latitude of Paranal


# ======================================================================
def dec_deg(dd, mm=0, ss=0):
    ''' -------------------------------------------------------
    Converts a declination in deg, minutes, seconds in degrees
    ------------------------------------------------------- '''
    tmp = np.abs(dd) + mm/60 + ss/3600
    tmp = tmp if dd > 0 else -tmp
    return tmp


# ==================================================================
def mas2rad(x):
    ''' ----------------------------------------------------------------
    Simple utility: convert milliarcsec to radians
    ---------------------------------------------------------------- '''
    return(x * 4.8481368110953599e-09)  # = x*np.pi/(180*3600*1000)


# ======================================================================
def har(hw=4, nh=1):
    ''' -------------------------------------------------------
    Returns an array of *nh* hour angles *in radians* across
    a +/- *hw* hour window

    Note:
    ----
    if only 1 point is requested, the returned ha will be at
    transit (ha = 0).
    ------------------------------------------------------- '''
    tmp = np.linspace(-hw, hw, nh) * np.pi/12
    if nh == 1 or hw == 0:
        tmp = np.array(0.0)
    return tmp


# ======================================================================
def knuller_matrix_4T(phasor=np.pi/2):
    ''' ----------------------------------------------------------------
    Returns the complex matrix that describes the effect of the
    kernel-nuller on the dark outputs only.
    ---------------------------------------------------------------- '''
    phi1 = np.exp(1j*phasor)
    phi2 = -np.conj(phi1)    # for concision

    MM1 = (1.0 / 4.0) * np.array(
        [[1+phi1,  1-phi1, -1+phi1, -1-phi1],
         [1+phi2, -1+phi2,  1-phi2, -1-phi2],
         [1+phi1,  1-phi1, -1-phi1, -1+phi1],
         [1+phi2, -1+phi2, -1-phi2,  1-phi2],
         [1+phi1, -1-phi1,  1-phi1, -1+phi1],
         [1+phi2, -1-phi2, -1+phi2,  1-phi2]])
    return MM1


# ======================================================================
def nuller_matrix_4T():
    ''' ----------------------------------------------------------------
    Returns the complex matrix that describes the effect of a 4T
    nuller on the dark outputs only.
    ---------------------------------------------------------------- '''
    MM0 = 0.5 * np.array(
        [[1,  1,   1,  1],  # the nuller matrix, alone
         [1,  1,  -1, -1],
         [1, -1,   1, -1],
         [1, -1,  -1,  1]])

    MM1 = MM0[1:]
    return MM1


# ======================================================================
def kernel_matrix_6O():
    ''' ----------------------------------------------------------------
    Returns the Kernel- matrix of the ideal kernel-nuller with the
    pi/2 phase shift.
    ---------------------------------------------------------------- '''
    KK = np.array([[1, -1,  0,  0,  0,  0],
                   [0,  0,  1, -1,  0,  0],
                   [0,  0,  0,  0,  1, -1]])
    return KK


# ======================================================================
# ======================================================================
class Nuller():
    # ==================================================================
    def __init__(self, encoords=vlti_ut, lat=vlti_lat, ref=0,
                 label="VLTI 4-UTs (Paranal)"):
        ''' --------------------------------------------------------
        An interferometer contains information about the location of
        an array of telescopes (cartesian coordinates) in a local
        reference frame

        Parameters:
        ----------
        - encoords: (2xN) array of (x,y) coordinates (in meters)
        - lat: array mean latitude (in degrees)
        - ref: index of aperture used as a (0,0) ref (default: 0)
        -------------------------------------------------------- '''
        self.encoords = encoords  # x,y coordinates of the array (meters)
        self.nap = encoords.shape[1]  # number of apertures in array

        self.lat = lat          # mean latitude of the observatory (deg)
        self._lat = lat * dtor  # mean latitude (radians)

        self.label = label            # a string for plots
        self.center_array(ref=ref)    # shifted coordinate array

        self.update_observation(
            cwavel=3.6e-6, fovsize=15.0, gsize=100, tdec=self.lat,
            combiner="nuller", hawidth=4, npoints=17)

        self._compute_grid()  # update f.o.v grid map

    # ==================================================================
    def __str__(self):
        ''' --------------------------------------------------------
        To get a sensible output when you want to print the object!
        -------------------------------------------------------- '''
        msg = "Nuller simulation:\n"
        msg += "-----------------\n"
        msg += "Configuration:      %s\n" % (self.label)
        msg += "Combiner:           %s\n" % (self.combiner)
        msg += "Latitude:           %+.2f degrees\n" % (self.lat)
        msg += "Wavelength:         %.2f microns\n" % (self.cwavel * 1e6)
        msg += "Target declination: %+.2f deg %s\n" % (self.tdec, self.decnote)
        msg += "Field of view:      +/-%d mas\n" % (self.fovsize)
        msg += "Observing window:   +/-%d hours\n" % (self.hawidth)
        msg += "# of points:        %d\n" % (self.npoints)
        return msg

    # ==================================================================
    def update_observation(
            self, cwavel=None, tdec=None, fovsize=None, gsize=None,
            combiner="nuller", hawidth=None, npoints=None):
        ''' --------------------------------------------------------
        Updates the parameters of an observation:

        Parameters:
        ----------
        - cwavel: the wavelength of observation (in meters)
        - tdec: the declination of the target (in degrees)
        - fovsize: the radius of the f.o.v (in mas)
        - gsize: the resolution of the f.o.v (in pixels)
        - combiner: the type of combiner (string)
        - hawidth: the half-width of the observation (in hours)

        Note:
        ----
        For the combiner type: any string that contains "kernel"
        will use the matrix of a kernel nuller. Anything else will
        simulate a regular nulller.
        -------------------------------------------------------- '''
        if cwavel is not None:   # central wavelength of observation
            self.cwavel = cwavel
            print("Wavelength: %.2f microns\n" % (self.cwavel * 1e6))

        if tdec is not None:      # target declination
            self.tdec = tdec
            self._tdec = tdec * dtor
            if self.tdec == self.lat:
                self.decnote = "(@Zenith !)"
            else:
                self.decnote = ""

        if hawidth is not None:  # half-width of hour angles
            self.hawidth = hawidth

        if npoints is not None:  # number of points over the observing window
            self.npoints = npoints

        # update the array of hour angles (in radians)
        self.har = har(self.hawidth, self.npoints)

        if fovsize is not None:  # "radius" of the field of view
            self.fovsize = fovsize
            self._compute_grid()

        if gsize is not None:    # field of view "grid" size
            self.gsize = gsize
            self._compute_grid()

        if combiner is not None:
            if "kernel" in combiner.lower():  # nuller matrix
                self.MM = knuller_matrix_4T()
                self.combiner = "4T -> 6-output Ker-nuller"
                print(self.combiner)
            else:
                self.MM = nuller_matrix_4T()
                self.combiner = "4T -> 3-output nuller"
                print(self.combiner)

    # ==================================================================
    def _compute_grid(self):
        ''' --------------------------------------------------------
        Helper function that computes and updates the coordinate
        arrays _xx & _yy that are later used for the computation
        over a grid of positions.

        _xx & _yy coordinates are expressedn in milliarcsec
        -------------------------------------------------------- '''
        try:
            _ = self.fovsize * self.gsize
            print("Updating fov grid!")
        except AttributeError:
            print("fov grid not ready yet...")
            return

        pos = np.linspace(-self.fovsize, self.fovsize, self.gsize)
        self._xx, self._yy = np.meshgrid(pos, pos)

    # ==================================================================
    def center_array(self, ref=0):
        ''' --------------------------------------------------------
        Returns new coordinates for the array, measured relative to
        one aperture index used as the local origin.

        Shifts all coordinates so that the *reference aperture* of
        index *ref* is at the (0,0) point.

        Parameters:
        ----------
        - ref: the index of the reference aperture.
        -------------------------------------------------------- '''
        self._apc = self.encoords.copy()  # original array coordinates
        self._apc[0] -= self._apc[0, ref]  # aperture coordinate
        self._apc[1] -= self._apc[1, ref]  # aperture coordinate
        self.ref = ref            # index of (0,0) reference aperture
        self._papc = self._apc.copy()  # projected aperture coordinates

    # ==================================================================
    def projected_array(self, ha=0.0, dec=0.0):
        ''' --------------------------------------------------------
        Returns the projected array from the POV of a star located
        at hour angle *ha* and declination *dec*

        Parameters:
        ----------
        - ha: hour angle (in radians)
        - dec: declination (in radians)

        Note:
        ----
        1. If "ha" is an array, the function returns the baselines
         computed for all the hour angles.
        -------------------------------------------------------- '''
        _ha = np.array(ha)  # precaution
        nh = _ha.size       # number of hour angles to compute
        dE = self._apc[0]   # east coordinates
        dN = self._apc[1]   # north coordinates
        lat = self._lat     # observatory latitude

        xx = np.outer(np.cos(_ha), dE) - \
            np.outer(np.sin(_ha), dN) * np.sin(lat)

        yy = np.sin(dec) * np.outer(np.sin(_ha), dE) + \
            np.sin(lat) * np.sin(dec) * np.outer(np.cos(_ha), dN) + \
            np.cos(lat) * np.cos(dec) * np.outer(np.ones(nh), dN)

        tmp = np.transpose(np.array([xx, yy]), axes=[0, 2, 1])

        if nh == 1:  # baselines for a single hour angle
            self._papc = tmp[:, :, 0]  # tmp[0] in the other order
        else:
            self._papc = tmp
        return self._papc

    # ==================================================================
    def plot_projected_array_tracks(
            self, figsize=(5, 5), ssz=50, tlabel="UT"):
        ''' --------------------------------------------------------
        Produces a plot of the projected array tracks over the
        provided range of hour angle for a target specified by its
        declination

        Parameters:
        ----------
        - figsize: figure size in inches (matplotlib)
        - ssz: the mean size of the plot symbol (default = 50)
        - tlabel: a label for the plot (default = "UT")
        -------------------------------------------------------- '''
        tmp = self.projected_array(ha=self.har, dec=self._tdec)

        f1, ax = plt.subplots(1, 1)
        f1.set_size_inches(*figsize, forward=True)
        ssz0 = 50  # symbol size

        if self.hawidth > 1e-1:
            ssz = ssz0 * (1 + self.har/self.hawidth * 12/np.pi)
        else:
            ssz = ssz0

        for ii in range(self.nap):
            ax.scatter(tmp[0, ii], tmp[1, ii],
                       s=ssz, label="%s%d" % (tlabel, ii+1,))

        ax.axis("equal")
        ax.legend()
        pmsg = r"$\delta$=%+.2f$^o$" % (self.tdec)
        if self.tdec == self.lat:
            pmsg = "through Zenith"
        ax.set_title(
            r"Projected array (+/- %dh, %s)" % (self.hawidth, pmsg))
        ax.set_xlabel(
            "East location relative to %s%d (meters)" % (tlabel, self.ref+1))
        ax.set_ylabel(
            "North location relative to %s%d (meters)" % (tlabel, self.ref+1))

        f1.set_tight_layout(True)
        return f1, ax

    # ==================================================================
    def theoretical_signal_companion(self, dra=4.8, ddec=1.8, con=1e-2):
        ''' --------------------------------------------------------
        Produces the theoretical astrophysical null induced by a
        single off-axis companion

        Parameters:
        ----------
        - dra: companion position right ascension offset (in mas)
        - ddec: companion position declination offset (in mas)
        - con: companion contrast
        -------------------------------------------------------- '''
        _ = self.projected_array(ha=self.har, dec=self._tdec)  # updates _papc
        off_axis = mas2rad(self._papc[0] * dra + self._papc[1] * ddec)
        efield = np.exp(-1j*2*np.pi/self.cwavel * off_axis)

        # compute nuller outputs
        output = con * np.abs(np.tensordot(self.MM, efield, axes=1))**2
        return output

    # ==================================================================
    def theoretical_kernel_companion(self, dra=4.8, ddec=1.8, con=1e-2):
        ''' --------------------------------------------------------
        Produces the theoretical astrophysical kernel induced by a
        single off-axis companion

        Parameters:
        ----------
        - dra: companion position right ascension offset (in mas)
        - ddec: companion position declination offset (in mas)
        - con: companion contrast
        -------------------------------------------------------- '''
        if "ker" in self.combiner.lower():  # it's a kernel-nuller
            signal = self.theoretical_signal_companion(
                dra=dra, ddec=ddec, con=con)
            KK = kernel_matrix_6O()
            kernel = KK.dot(signal)
        else:
            kernel = None
            print("Current configuration not a kernel-nuller!")
        return kernel

    # ==================================================================
    def kernel_signal(self, signal):
        if "ker" in self.combiner.lower():  # it's a kernel-nuller
            KK = kernel_matrix_6O()
            kernel = np.tensordot(KK, signal, axes=1)
        else:
            kernel = None
            print("Current configuration not a kernel-nuller!")
        return kernel

    # ==================================================================
    def mc_perturbed_signal_companion(self, dra=4.8, ddec=1.8, con=1e-2,
                                      rms=50, nmc=100):
        ''' --------------------------------------------------------
        Produces a recording of "experimental" astrophysical null
        induced by a unique off-axis companion in the presence of
        perturbation

        Parameters:
        ----------
        - dra: companion position right ascension offset (in mas)
        - ddec: companion position declination offset (in mas)
        - con: companion contrast
        - rms: the amount of resitual piston (in nanometers)
        - nmc: number of MC iterations per epoch
        -------------------------------------------------------- '''

        _ = self.projected_array(ha=self.har, dec=self._tdec)  # updates _papc
        off_axis = mas2rad(self._papc[0] * dra + self._papc[1] * ddec)

        piston = rms * 1e-9 * np.random.randn(self.nap, self.npoints, nmc)
        piston[self.ref, :, :] = 0.0  # measured relative to ref aperture
        ef_on = np.exp(-1j*2*np.pi/self.cwavel * piston)  # on-axis efield

        for ii in range(nmc):
            piston[:, :, ii] += off_axis

        ef_off = np.exp(-1j*2*np.pi/self.cwavel * piston)  # off-axis efield

        # compute nuller outputs
        output = np.abs(np.tensordot(self.MM, ef_on, axes=1))**2 + \
            con * np.abs(np.tensordot(self.MM, ef_off, axes=1))**2

        return output

    # ==================================================================
    def plot_theoretical_signal_companion(self, dra=4.8, ddec=1.8, con=1e-2):
        ''' --------------------------------------------------------
        Produces a plot of the theoretical astrophysical null induced
        by a single off-axis companion

        Parameters:
        ----------
        - dra: companion position right ascension offset (in mas)
        - ddec: companion position declination offset (in mas)
        - con: companion contrast
        -------------------------------------------------------- '''
        data = self.theoretical_signal_companion(dra=dra, ddec=ddec, con=con)

        fig, ax = plt.subplots()
        for ii in range(data.shape[0]):
            ax.plot(
                self.har * 12/np.pi, data[ii], label="output #%d" % (ii+1,))
        ax.legend()
        ax.set_xlabel("Hour angle")
        ax.set_ylabel("Nuller output (fraction of telescope flux)")
        mytitle = self.combiner
        mytitle += r" (%.1e companion @ (%+.1f, %+.1f))" % (con, dra, ddec)
        ax.set_title(mytitle)
        fig.set_tight_layout(True)

    # ==================================================================
    def plot_experimental_kernel(self, signal, title=""):
        ''' --------------------------------------------------------
        Produces a plot of the experimental kernel from the raw
        output of the kernel-nuller

        Parameters:
        ----------
        - signal:
        - title: string describing the plot

        Note: the concept for this plot comes from A. Meilland's
        example shown in one of the OCAPY meetups!
        -------------------------------------------------------- '''
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        kernel = self.kernel_signal(signal)  # kernel of outputs

        har = self.har * 12/np.pi
        kmean = kernel.mean(axis=2)
        kerr = kernel.std(axis=2)
        xx = np.concatenate([har, np.flip(har)])  # for the errorbar plot!

        fig, ax = plt.subplots()
        for ii in range(kernel.shape[0]):
            ys = kmean[ii]
            dy = kerr[ii]
            yy = np.concatenate([ys-dy, np.flip(ys+dy)])
            ax.fill(xx, yy, alpha=0.3, color=colors[ii])
            ax.plot(
                har, kmean[ii], label="kernel-output #%d" % (
                    ii+1,), color=colors[ii])
        ax.legend()
        ax.set_xlabel("Hour angle")
        ax.set_ylabel("Kernel of the nuller-output")
        if title != "":
            mytitle = title
        else:
            mytitle = "Kernel of Experimental output"
        ax.set_title(mytitle)
        fig.set_tight_layout(True)

    # ==================================================================
    def compute_ideal_output_maps(self):
        ''' --------------------------------------------------------
        Produces the ideal nuller output maps for the nuller and
        the specified observing conditions
        -------------------------------------------------------- '''
        nh = self.npoints   # number of hour angles
        gsz = self.gsize    # computation grid size
        nap = self.nap      # number of apertures

        _ = self.projected_array(ha=self.har, dec=self._tdec)  # updates _papc

        tmp = mas2rad(np.outer(self._papc[0], self._xx) +
                      np.outer(self._papc[1], self._yy))
        tmp = np.exp(-1j*2*np.pi/self.cwavel * tmp)

        if nh == 1:
            efield = tmp.reshape((nap, gsz, gsz))
        else:
            efield = tmp.reshape((nap, nh, gsz, gsz))

        # compute nuller outputs
        output = np.abs(np.tensordot(self.MM, efield, axes=1))**2
        return output

    # ==================================================================
    def compute_ideal_kmaps(self):
        ''' --------------------------------------------------------
        Produces the ideal kernel maps for the nuller and
        the specified observing conditions
        -------------------------------------------------------- '''
        if "ker" in self.combiner.lower():  # it's a kernel-nuller
            KK = kernel_matrix_6O()
            output = self.compute_ideal_output_maps()
            kmaps = np.tensordot(KK, output, axes=1)
        else:
            kmaps = None
            print("Current configuration not a kernel-nuller!")

        return kmaps

    # ==================================================================
    def mc_perturbed_grid_map(self, rms=50, nmc=100):
        ''' --------------------------------------------------------
        Produces a recording of "experimental" astrophysical null
        for every position on the fov grid.

        Parameters:
        ----------
        - rms: the amount of residual piston (in nanometers)
        - nmc: the number of MC iterations per epoch and per position

        Note:
        ----
        Returns: a large multi dimensional array, either:
        (nap, gsz, gsz, nmc) or (nap, nh, gsz, gsz, nmc)

        This computation can therefore take long and require a lot
        of memory: even for only 100 MC, on a 100x100 grid, you
        already have 1e6 computation per aperture and per hour
        angle... on my laptop, I wasn't able to run it for nmc=1000:
        not enough memory.
        -------------------------------------------------------- '''
        nh = self.npoints   # number of hour angles
        gsz = self.gsize    # computation grid size
        nap = self.nap      # number of apertures

        _ = self.projected_array(ha=self.har, dec=self._tdec)  # updates _papc

        off_axis = mas2rad(np.outer(self._papc[0], self._xx) +
                           np.outer(self._papc[1], self._yy))

        if nh == 1:
            off_axis = off_axis.reshape((nap, gsz, gsz))
            piston = rms * 1e-9 * np.random.randn(nap, gsz, gsz, nmc)
            piston[self.ref] = 0.0  # measured relative to ref aperture

            for ii in range(nmc):
                piston[:, :, :, ii] += off_axis

                print("\rMC prelim computation %3d/%3d" % (ii+1, nmc),
                      end="", flush=True)
        else:
            off_axis = off_axis.reshape((nap, nh, gsz, gsz))
            piston = rms * 1e-9 * np.random.randn(nap, nh, gsz, gsz, nmc)
            piston[self.ref] = 0.0  # measured relative to ref aperture
            for ii in range(nmc):
                piston[:, :, :, :, ii] += off_axis
                print("\rMC prelim computation %3d/%3d" % (ii+1, nmc),
                      end="", flush=True)

        efield = np.exp(-1j*2*np.pi/self.cwavel * piston)

        # compute nuller outputs
        print("... Final dot product computing...", end="", flush=True)
        output = np.abs(np.tensordot(self.MM, efield, axes=1))**2
        print("Done!", flush=True)
        return output

    # ==================================================================
    def export_ideal_output_maps_as_fits(self, fname="output_maps.fits"):
        ''' --------------------------------------------------------
        Computes and save the nuller response maps as a multi-dimensional
        fits file array and provides context information in the header.

        Parameters:
        ----------
        - fname: the name of the file to write to disk

        Note:
        ----
        Will overwrite mercilessly the fits file.
        -------------------------------------------------------- '''
        mmaps = self.compute_ideal_output_maps()

        hdu = pf.PrimaryHDU(mmaps)
        hdr = hdu.header
        hdr.comments['NAXIS1'] = "Delta-RA (%.2f mas/pixel)" % (
            self.fovsize * 2 / self.gsize,)
        hdr.comments['NAXIS2'] = "Delta-Dec (%.2f mas/pixel)" % (
            self.fovsize * 2 / self.gsize,)
        hdr.comments['NAXIS3'] = "Hour angle increment"
        hdr.comments['NAXIS4'] = "Nuller output index"
        hdr['CONFIG'] = (self.label, "Interferometric array configuration")
        hdr['COMBINER'] = (self.combiner, "Nuller description")
        hdr['CWAVEL'] = (self.cwavel, "Wavelength (in meters)")
        hdr['DEC'] = (self.tdec, "Target declination (in degrees)")
        hdr['HARANGE'] = (self.hawidth,
                          "Range of hour angles (+/- # of hours)")
        hdr['comment'] = "Created by the knuller python module"

        hdu.writeto(fname, overwrite=True)

    # ==================================================================
    def colinearity_map(self, omaps, signal):
        ''' --------------------------------------------------------
        Computes the colinearity map between nulled output maps
        and a signal.

        Parameters:
        ----------
        - omaps: a 3 or 4D array of output maps
        - signal: a 1 or 2D array of astrophysical signal

        Note:
        ----
        Assumes that you know what you do and have the same number
        of hour angles for both *omaps* and *signal*
        -------------------------------------------------------- '''
        if len(omaps.shape) == 4:
            tmp = np.tensordot(signal, omaps, axes=([0, 1], [0, 1]))
        else:
            tmp = np.tensordot(signal, omaps, axes=1)

        return tmp / np.tensordot(signal, signal)

    # ==================================================================
    def plot_fov_map(self, dmap, figsize=5, ftitle="", cmap=cm.jet,
                     cbar=False):
        if not cbar:
            fig, ax = plt.subplots()
            fig.set_size_inches(figsize, figsize, forward=True)
            self._dress_datamap_ax(ax, dmap, ftitle=ftitle, cmap=cmap)

        else:
            fsize = np.array([1, 0.2]) * figsize
            fig, axes = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': fsize.tolist()})
            fig.set_size_inches(fsize.sum(), figsize, forward=True)
            self._dress_datamap_ax(axes[0], dmap, ftitle=ftitle, cmap=cmap)

            im = axes[0].images[0]
            fig.colorbar(im, cax=axes[1], orientation="vertical")

        fig.set_tight_layout(True)

    # ==================================================================
    def _dress_datamap_ax(
            self, ax, dmap, cmap=cm.jet, vmin=None, vmax=None, ftitle=""):
        ''' --------------------------------------------------------
        Helper function that dresses the several types of maps that
        the object generates with axis labels and titles.

        Makes other parts of the code more compact!
        -------------------------------------------------------- '''
        mmax = self.fovsize
        ax.imshow(
            dmap, extent=(-mmax, mmax, -mmax, mmax), origin="lower",
            cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel(r"$\Delta$ R.A. (mas)")
        ax.set_ylabel(r"$\Delta$ Dec (mas)")
        ax.plot([0, 0], [0, 0], "w*", ms=16)
        ax.grid(ls="--")
        if ftitle != "":
            ax.set_title(ftitle)

        return ax

    # ==================================================================
    def plot_ideal_output_maps(self, mmaps, figsize=5, vmin=None, vmax=None,
                               cmap=cm.jet, clabel=""):
        ''' --------------------------------------------------------
        Plot a grid of maps sharing a colorbar

        Parameters:
        ----------
        - nmaps: a cube of maps to display
        - figsize: size of individual map figure
        - vmin: the minimum value of the colormap
        - vmax: the maximum value of the colormap
        - cmap: the colormap (from matplotlib.cm) to use
        - clabel: a label for the colorbar.
                 (default: "Transmission (x 1 telescope flux)")
        -------------------------------------------------------- '''
        nout = mmaps.shape[0]  # number of outputs

        fsize = np.array([1, 1, 1, 0.1]) * figsize

        if clabel == "":
            clabel = "Transmission (x 1 telescope flux)"

        if nout <= 3:
            fig, axes = plt.subplots(
                1, 4, gridspec_kw={'width_ratios': fsize.tolist()})
            fig.set_size_inches(fsize.sum(), figsize, forward=True)

            for kk in range(nout):
                self._dress_datamap_ax(
                    axes[kk], mmaps[kk], cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    ftitle="Nuller-output #%d" % (kk+1,))

            im = axes[0].images[0]
            cbar = fig.colorbar(im, cax=axes[3], orientation="vertical")
            cbar.set_label(clabel)
            fig.set_tight_layout(True)

        else:
            fig, axes = plt.subplots(
                2, 4, gridspec_kw={'width_ratios': fsize.tolist()})
            fig.set_size_inches(fsize.sum(), figsize*2, forward=True)

            for kk in range(nout):
                self._dress_datamap_ax(
                    axes[kk // 3, kk % 3], mmaps[kk], cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    ftitle="Nuller-output #%d" % (kk+1,))

            im = axes[0, 0].images[0]
            cbar1 = fig.colorbar(im, cax=axes[0, 3], orientation="vertical")
            cbar1.set_label(clabel)
            cbar2 = fig.colorbar(im, cax=axes[1, 3], orientation="vertical")
            cbar2.set_label(clabel)
            fig.set_tight_layout(True)

        return 0

    # ======================================================================
    #                 binary model fitting tools
    # ======================================================================
    def knull_fit_residuals(self, params, kvect, kerror):
        ''' ----------------------------------------------------------------
        Cost function for a binary-fitting procedure.
        
        Parameters:
        ----------

        - params: 3-component vector (dra, ddec, con)
        - kvect: the dataset to attempt to fit to
        - kerror: the uncertainties associated to the dataset

        Returns:
        -------
        The residuals (model - data) / error
        ----------------------------------------------------------------- '''
        model = self.theoretical_kernel_companion(
            dra=params[0], ddec=params[1], con=params[2]).flatten()
        return (model - kvect) / kerror

    # ======================================================================
    def knull_fit(self, p0, kvect, kerror):
        ''' ----------------------------------------------------------------
        Wrapper for the scipy leastsq procedure, specialized for the binary
        model fit of observations by the kernel-nuller.

        Parameters:
        ----------

        - p0: the initial 3-component binary parameter guess
        - kvect: the dataset to attempt to fit to
        - kerror: the uncertainties associated to the dataset

        Returns:
        -------
        The whole output of leastsq so refer to the documentation of that
        function to know what to do with it.
        ---------------------------------------------------------------- '''
        soluce = leastsq(self.knull_fit_residuals,
                         p0, args=((kvect.flatten(), kerror.flatten())),
                         full_output=1)
        return soluce
