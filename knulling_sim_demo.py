#!/usr/bin/env python

import numpy as np
import knuller_sim as kn
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# -----------------------------------------------------
#              simulation demo parameters
# -----------------------------------------------------
dec0 = kn.dec_deg(-64, 42, 45)  # target declination
dra, ddec, con = 5, 5, 1e-3     # test companion!
prms = 50                       # rms of piston residuals
npoints = 17                    # number of hour angles
tindex = npoints // 2 + 1       # h.a. index @ transit
mycmap = cm.rainbow             # color map
burst_mode = False              # script rhythm control
# -----------------------------------------------------

print("""
========================================================================
                   knuller_sim demo starting!
========================================================================

The program will produce a series of outputs one at a time,
and give you an idea of what it's doing as it's doing it.
Of course, check the actual code to see how you can use and
adapt it to your own needs.

Interrupt the execution at any time by pressing Ctrl-D.

The first step is to create a default instance of the Nuller class.
- display the current assumed setup
- produce a plot of the projected array tracks
======================================================================== """)


if burst_mode is False:
    _ = input("Press key to continue..")

myk = kn.Nuller()  # default nuller is a 4T -> 3 output nuller
print(myk)         # tells you everything about the setup

myk.plot_projected_array_tracks()

print("""
========================================================================
For the next step, we update the pointing and the width of the observing
window and can see how it affects the projected array tracks.
======================================================================== """)
if burst_mode is False:
    _ = input("Press key to continue..")

myk.update_observation(tdec=dec0)  # change the pointing
print(myk)
myk.plot_projected_array_tracks()

print("""
========================================================================
Now, this will compute the theoretical signal of a single companion
and plot for the 4T->6 output KERNEL nuller.

For this it needs to uppdate the type of combiner -> kernel
Then it can do a plot of the theoretical kernel signal.
======================================================================== """)
if burst_mode is False:
    _ = input("Press key to continue..")


# update the conditions of the observation sequence!
myk.update_observation(hawidth=4, npoints=npoints, combiner="kernel")
print(myk)  # check that things are the way you want them

# you can direclty generate a plot
myk.plot_theoretical_signal_companion(dra=dra, ddec=ddec, con=con)

print("""
========================================================================
The next part will compute complete response of the nuller over the fov
and export them as a fits file that you can open with a tool like DS9
======================================================================== """)
if burst_mode is False:
    _ = input("Press key to continue..")

nmaps = myk.compute_ideal_output_maps()

# you can plot the 6 maps for one epoch (here at transit)
myk.plot_ideal_output_maps(nmaps[:, tindex], figsize=4, cmap=mycmap)

# and save the whole sequence as a fits file
myk.export_ideal_output_maps_as_fits("output_maps.fits")

print("""
========================================================================
A useful tool to predict the efficiency of a given observing scenario
is to look at the time averaged integrated output of the nuller.

The next part of the code will compute and display the overall throughput 
map for this observation & identify possible "blind spots"
======================================================================== """)
if burst_mode is False:
    _ = input("Press key to continue..")

gmap = nmaps.sum(axis=0).mean(axis=0)  # sum the channels, average the epochs
myk.plot_fov_map(
    gmap, ftitle="Global nuller throughput map",
    cmap=mycmap, cbar=True)

print("""
========================================================================
The colinearity map is a useful diagnostic tool that computes over a
grid of possible positions, in the high-contrast regime approximation,
the dot product between the signal that a single companion present would
induce and a data-set, be it real or simulated.

Here, we use a simulated test_binary located at:
- delta RA  = +5 mas
- delta DEC = +5 mas
- contrast  = 1e-3
======================================================================== """)
if burst_mode is False:
    _ = input("Press key to continue..")

# compute the signal of a binary companion (RA, dec, contrast)
test_binary = myk.theoretical_signal_companion(dra=dra, ddec=ddec, con=con)


cmap = myk.colinearity_map(nmaps, test_binary)
myk.plot_fov_map(cmap, ftitle="Raw output Colinearity map ",
                 cmap=mycmap, cbar=True)

print("""
========================================================================
MC simulation!

Thus far, computations were for perfect observing conditions.

We introduce some amount of piston residuals: prms = 50 nm and for each
of the hour angles, we compute a large number (nmc = 10000) simulated
datasets produced by the 6 raw outputs of the kernel-nuller.

Because of the way output are distributed, it is tricky to make a simple
plot that makes sense. We are just going to compare the theoretical 
signal of the binary to the min() of the distribution for all h.a. ... 
which is just to do something, but is not that interesting.
======================================================================== """)
if burst_mode is False:
    _ = input("Press key to continue..")

signal = myk.mc_perturbed_signal_companion(
    dra=dra, ddec=ddec, con=con, rms=prms, nmc=10000)
signalmean = np.min(signal, axis=2)

# redo the plot of the theoretical signal
myk.plot_theoretical_signal_companion(dra=dra, ddec=ddec, con=con)

# and overplot the mean of the distributed value for all 6 output*
plt.gca().set_prop_cycle(None)  # reset color cycle
for ii in range(6):
    plt.plot(myk.har * 12/np.pi, signalmean[ii], '--')

print("""
========================================================================
KERNELS!

Now we are moving into kernel-territory. Previous plots and outputs were
looking at the raw outputs. The next plot still uses the MC simulations:
it plots the mean kernel and their 1-sigma uncertainty as a fat line.

Given how the kernels are distributed, it is much easier to get a sense
for the SNR.
======================================================================== """)
if burst_mode is False:
    _ = input("Press key to continue..")


kernel = myk.kernel_signal(signal)

myk.plot_experimental_kernel(
    signal,
    title="Kernel of experimental output for %d rms piston residuals" % (
        prms,))

print("""
========================================================================
Next: show what the kernel-response maps do look like at the time of the
transit.
======================================================================== """)
if burst_mode is False:
    _ = input("Press key to continue..")

# -----------------------------------------------------
#           show the k-maps @ transit time
# -----------------------------------------------------
kmaps = myk.compute_ideal_kmaps()

myk.plot_ideal_output_maps(
    kmaps[:, tindex], figsize=4, cmap=mycmap, clabel="Kernel-output")

print("""
========================================================================
Next, compute the colinearity map, this time for kernel-output and not
for raw outputs as was done before.

We still use the same simulated test_binary located at:
- delta RA  = +5 mas
- delta DEC = +5 mas
- contrast  = 1e-3
======================================================================== """)
if burst_mode is False:
    _ = input("Press key to continue..")

kernel0 = myk.theoretical_kernel_companion(dra=dra, ddec=ddec, con=con)

kcmap = myk.colinearity_map(kmaps, kernel0)
myk.plot_fov_map(kcmap, ftitle="Kernel colinearity map ",
                 cmap=mycmap, cbar=True)

print("""
========================================================================
Since kernels can take negative values, to make some comments about
sensitivity of this technique, it is useful to look at the norm of the
kernel-signal.
======================================================================== """)
if burst_mode is False:
    _ = input("Press key to continue..")

knorm = np.sqrt((np.abs(kmaps)**2).sum(axis=0)).mean(axis=0)
myk.plot_fov_map(knorm, ftitle="Kernel norm map",
                 cmap=mycmap, cbar=True)

print("""
========================================================================
The end.
======================================================================== """)
if burst_mode is False:
    _ = input("Press key to continue..")

# -----------------------------------------------------
# MC computation over the whole fov grid of positions
# this is commented out because it takes memory and cpu
#
# nevertheless, there is an interesting plot that is
# added here... maps of null std deviation. Not sure
# they are particularly meaningful... but they show
# how one could use the result of this MC computation.
# -----------------------------------------------------

# mc_test = myk.mc_perturbed_grid_map(rms=150, nmc=200)
# myk.plot_ideal_output_maps(
#     mc_test[:, tindex].std(axis=3), figsize=4, cmap=mycmap,
#     clabel="Nuller output stdev")
