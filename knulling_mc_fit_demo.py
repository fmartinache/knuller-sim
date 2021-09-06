#!/usr/bin/env python

import numpy as np
import knuller_sim as kn
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -----------------------------------------------------
#              simulation demo parameters
# -----------------------------------------------------
dec0 = kn.dec_deg(-64, 42, 45)  # target declination
dra, ddec, con = 5, 5, 1e-2     # test companion!
prms = 50                       # rms of piston residuals
npoints = 17                    # number of hour angles
tindex = npoints // 2 + 1       # h.a. index @ transit
mycmap = cm.rainbow             # color map
burst_mode = False              # script rhythm control

# -----------------------------------------------------
# instantiating and tuning a Kernel-Nuller
# -----------------------------------------------------
myk = kn.Nuller()  # default nuller is a 4T -> 3 output nuller
myk.update_observation(tdec=dec0)  # change the pointing
myk.update_observation(hawidth=4, npoints=npoints, combiner="kernel")

# -----------------------------------------------------
#                  MC simulation
#
# K-nulling observations in the presence of residual
# piston errors of increasing magnitude (from 10 to
# 150 nm RMS).
# for each MC realisation, the program attempts to find
# the best fit parameters, using a leastsq algorithm.
#
# the idea here is to verify (experimentally) how
# increasing piston errors result in increasing
# astrometric uncertainty.
#
# the leastsq fitting of a large number of experiments
# can take a while, depending on how many iterations
# one asks for, and how fast/powerful one's computer is
# -----------------------------------------------------
nmc = 200   # number of MC iterations per piston

p0 = dra+1, ddec-1, con*1.3  # just to start slightly off the mark!

npst = 15  # number of piston rms values
prms = np.arange(1, npst+1) * 10.0  # series of pistons to test for

pfit = np.zeros((npst, nmc, 3))  # array storing the best fit parameters

for jj in range(npst):
    signal = myk.mc_perturbed_signal_companion(
        dra=dra, ddec=ddec, con=con, rms=prms[jj], nmc=10000)
    kernel = myk.kernel_signal(signal)
    kerr = kernel.std(axis=2)  # kernel statistical uncertainty

    print("\n>> %02d/%02d : piston RMS = %d nm" % (jj+1, npst, prms[jj]))
    for ii in range(nmc):
        print("\r%05d/%05d" % (ii+1, nmc), end="", flush=True)
        tmp = myk.knull_fit(p0, kernel[:, :, ii], kerr)
        pfit[jj, ii] = tmp[0]

# -----------------------------------------------------
# plots!
# -----------------------------------------------------

f1 = plt.figure(figsize=(21, 5))
ax1 = f1.add_subplot(1, 3, 1)
ax1.errorbar(prms, pfit.mean(axis=1)[:,0], yerr=pfit.std(axis=1)[:,0])
ax1.set_xlabel("Residual piston rms (in nanometers)")
ax1.set_ylabel("Best fit location (in mas)")
ax1.set_title("Right ascension offset")

ax2 = f1.add_subplot(1, 3, 2)
ax2.errorbar(prms, pfit.mean(axis=1)[:,1], yerr=pfit.std(axis=1)[:,1])
ax2.set_xlabel("Residual piston rms (in nanometers)")
ax2.set_ylabel("Best fit location (in mas)")
ax2.set_title("Declination offset")

ax3 = f1.add_subplot(1, 3, 3)
ax3.errorbar(prms, pfit.mean(axis=1)[:,2], yerr=pfit.std(axis=1)[:,2])
ax3.set_xlabel("Residual piston rms (in nanometers)")
ax3.set_ylabel("Best fit location (in mas)")
ax3.set_title("Contrast")

f1.suptitle("MC simulation (n=%d)" % (nmc))
f1.set_tight_layout(True)
