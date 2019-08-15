# Overpressure functions of U.S. and Soviet origin.

import numpy as np
from scipy.integrate import quad
from glasstone.utilities import convert_units, ValueOutsideGraphError

# First, some utility functions; these should go in a dedicated file eventually
def scale_range(bomb_yield, ground_range):
    return ground_range / (bomb_yield**(1.0 / 3))

def scale_height(bomb_yield, burst_height):
    return burst_height / (bomb_yield**(1.0 / 3))

# Overpressure function from pp. 60-71 of H.L. Brode, _Airblast From Nuclear Bursts-
# Analytic Approximations_ (Pacific-Sierra Research Corporation, 1986).
# Appears to have been made using Fourier transforms on data from an model of a blast
# wave over an ideal surface.  The many local functions it contains are of no general
# physical significance.

def _brode(z, r, y):
    """Brode equation for approximating peak static overpressure for a 1 kT burst.
    Units are kT (yield) and kilofeet (ground range, burst height).
    Caveats: This is accurate to ~10%, and presumes sea level ambient pressure.
    """
    def a(z):
        return 1.22 - ((3.908 * z**2) / (1 + 810.2 * z**5))
    def b(z):
        return 2.321 + ((6.195 * z**18) / (1 + 1.113 * z**18)) - ((0.03831 * z**17) / (1 + 0.02415 * z**17)) + (0.6692 / (1 + 4164 * z**8))
    def c(z):
        return 4.153 - ((1.149 *  z**18) / (1 + 1.641 * z **18)) - (1.1 / (1 + 2.771 * z**2.5))
    def d(z):
        return -4.166 + ((25.76 * z**1.75) / (1 + 1.382 * z**18)) + ((8.257 * z) / (1 + 3.219 * z))
    def e(z):
        return 1 - ((0.004642 * z**18) / (1 + 0.003886 * z**18))
    def f(z):
        return 0.6096 + ((2.879 * z**9.25) / (1 + 2.359 * z**14.5)) - ((17.5 * z**2) / (1 + 71.66 * z**3))
    def g(z):
        return 1.83 + ((5.361 * z**2) / (1 + 0.3139 * z**6))
    def h(z, r, y):
        return ((8.808 * z**1.5) / (1 + 154.5 * z**3.5)) - ((0.2905 + 64.67 * z**5) / (1 + 441.5 * z**5)) - ((1.389 * z) / (1 + 49.03 * z**5)) + ((1.094 * r**2) / ((781.2 - (123.4 * r) + (37.98 * r**1.5) + r**2) * (1 + (2 * y))))
    def j(y):
        return ((0.000629 * y**4) / (3.493e-9 + y**4)) - ((2.67 * y**2) / (1 + (1e7 * y**4.3)))
    def k(y):
        return 5.18 + ((0.2803 * y**3.5) / (3.788e-6 + y**4))
    return (10.47 / r**a(z)) + (b(z) / r**c(z)) + ((d(z) * e(z)) / (1 + (f(z) * r**g(z)))) + h(z, r, y) + (j(y) / r**k(y))

def _brodeop(bomb_yield, ground_range, burst_height):
    """Calculate overpressure for arbitrary-height airbursts using Brode 
    equation.

    Units: kT, kilofeet
    
    Warning: ground_range = 0 results in divide-by-zero error.
    """
    z = (burst_height / ground_range)
    y = scale_height(bomb_yield, burst_height)
    x = scale_range(bomb_yield, ground_range)
    r = (x**2 + y**2)**0.5
    return _brode(z, r, y)

def brode_overpressure(y, r, h, yunits='kT', dunits='m', opunits='kg/cm^2'):
    """Estimate peak static overpressure at radius r from the epicenter of a burst
with yield y and a burst height of h using the Brode equation."""
    yld = convert_units(y, yunits, 'kT')
    ground_range = convert_units(r, dunits, 'kilofeet')
    height = convert_units(h, dunits, 'kilofeet')
    op = _brodeop(yld, ground_range, height)
    return convert_units(op, 'psi', opunits)


# Functions from NRDC, _The U.S. Nuclear War Plan: A Time for Change_
# These were, in turn, sourced from the help files of the Reagan-era
# Defense Nuclear Agency DOS programs BLAST and WE

# AIR-BURST DESCRIPTION
# DESCRIPTION
# This  program  calculates the peak overpressure and the peak  dynamic
# pressure at  the surface of the earth as  a function of height  of  burst,
# ground  range,   and weapon  yield.    The surface  is  treated  as  "near-
# ideal,"  with  no  account   of  mechanical effects  of  the  surface  (for
# example,  buildings  and terrain with appreciably varying slope affect  the
# the  shock front,   and deformable material will absorb some of  the  blast
# energy).   Also neglected are thermal effects due to absorption  of heat by
# the  surface  (to form a layer of heated air that distorts the blast  wave)
# and  involvement  of surface material ("dust") with the  flow.  Hence,  the
# calculations  apply  to  relatively  flat,   rigid,   clean  and  thermally
# reflective surfaces.

# The  incident  blast wave is reflected by the surface,  and beyond   a
# certain ground range (called the Mach stem formation range) this  reflected
# wave  merges with the incident wave to produce a stronger shock in what  is
# called  the Mach reflection region.  The region inside the Mach stem   for-
# mation range is known as the regular reflection region.  For peak overpres-
# sures  below  about 350 kPa this merging has a marked effect:  the  highest
# peak  overpressure  at  a given yield and ground range is  produced  for  a
# height of burst somewhat above zero.

# The  air-burst  peak overpressure fit is based in part on the  DNA  1-
# kiloton free-air Standard.  The results at all ground ranges from a surface
# burst  are  consistent with this model (using  twice-yield  scaling);   the
# results  for   the point on the ground directly below the  burst  are  also
# consistent  (using   the  free-air  normal  reflection  factor),   for  all
# heights of  burst  within   the  model  limits.     Other  points  on  the
# overpressure  curves are from curve fits based on data from Carpenter  (see
# SOURCES OF DATA below).

# The  air-burst peak dynamic pressure of a reflected shock in the  Mach
# reflection region is calculated from Rankine-Hugoniot relations applied  to
# the merged shock front.  In the regular reflection region, the shape of the
# dynamic-pressure   height-of-burst  curves  (contours  in  height-of-burst,
# ground-range  space)  is  determined from theory,  but  approximated  by  a
# simpler function in this program.   The two regions, regular and Mach, have
# been merged by a switching function that is centered near the onset of Mach
# stem formation.

# A  calculation is provided to determine the Mach stem formation  range
# (as a function of height of burst), and the height of the triple point (the
# point  at which the Mach stem,  incident shock front,  and reflected  shock
# meet) for distances larger than the Mach stem formation range.

# The  other  air-burst calculations this program provides  involve  the
# dependence  of pressure (either overpressure or dynamic pressure) on  time.
# These  time-dependent calculations (also known as wave-form calculations)
# are:

#     o Partial and total impulses
#     o Time-dependent pressures
#     o Positive phase durations

# The  impulse  calculations  numerically integrate  the  time-dependent
# pressure  functions with  respect to time.   A total impulse is  integrated
# from  the time of the blast wave arrival until the time-dependent  pressure
# reaches zero (that is,  for the entire duration of the positive phase).   A
# partial impulse is a similar integration of pressure versus time,  but to a
# final time less than the positive phase duration.

# The  classical shape of the waveform is a sharp peak at the  time  the
# blast waves arrives,  followed by a quick decay to zero.   However,  it has
# been  found that in certain cases the highest pressures do not occur at the
# time of arrival of the shock front.   In the transition region from regular
# to Mach reflection, for peak overpressures larger than roughly 700 kPa, the
# overpressure  waveform can have two peaks,  and the second maximum  may  be
# larger  than  the first.   Building upon empirical waveform experiments  by
# Carpenter and on calculations, Speicher and Brode have developed a model of
# the double-peaked overpressure waveform.

# The  dynamic  pressure  waveform  model is  approximated  by  using  a
# variation  on the overpressure waveform as a starting point.   It is  known
# that  the  resulting dynamic pressure waveforms are not  always  realistic,
# since  investigations have shown that a dynamic pressure maximum can  occur
# between  the  two overpressure peaks.   Nevertheless,  it is  a  reasonable
# approximation  in  light of scarce dynamic pressure data.   In  the  fully-
# developed  Mach region,  where waveforms are similar to the classical wave-
# form model, the dynamic pressure waveforms are good.

# The waveform calculation models used by this program are derived  from
# work  done  by  Speicher and Brode (see SOURCES OF DATA below) with  a  few
# modifications.   Simpler  fits to Carpenter's peak overpressure  data  have
# been  substituted  for those of Speicher and Brode,  and used in  the  peak
# dynamic pressure models.  A slightly different time-of-arrival function has
# also been used.

# Other changes have been made to the Speicher and Brode waveform models
# to  bring some of the total impulse calculations into closer agreement with
# the  REFLECT-4 code work done by Smiley,  Ruetenik and  Tomayko.   For  the
# overpressure  waveform  model a slight modification of the  positive  phase
# duration  was  made.   This modification has not degraded the  overpressure
# versus time fit;  in fact, comparisons with REFLECT-4 waveforms showed some
# improvements in the overpressure model.

# More  extensive changes have been made to the dynamic pressure  model.
# The  positive  phase  duration  was replaced with a new fit  based  on  the
# REFLECT-4  data.   Then  the total impulse calculation  was  optimized  (by
# changing one of the waveform parameters) to match the results of REFLECT-4.
# As with the overpressure model,  these modifications have not significantly
# degraded the dynamic pressure versus time fit.

# ACCURACY
# Because the particular conditions at the surface reflecting the  blast
# are critical in determining the resultant blast-wave characteristics, over-
# pressures  (and other blast parameters) are somewhat less predictable  than
# those  of  free-air  bursts.   Carpenter's data for  ground  distances  are
# estimated to be reliable to within +-15%,  for yields between 1 and  20,000
# kT.   This  may correspond to uncertainties in the overpressure as much  as
# 30%,  depending upon the height of burst.   Figures for overpressure  above
# 700  kPa are purely theoretical and are estimated to be reliable to  within
# +-20%. Scaling for yields below 1 kT and scaled heights of burst between 50
# and 300 meters has shown large errors corresponding to factors of 2 or 3 in
# effective yield.

# The  peak overpressure fits used are typically accurate to +-4% with  a
# worst  case of 11% when compared with Carpenter's data.   The average error
# increases to about +-10% if the fits are extended to 350,000 kPa.

# As  noted before,  the dynamic pressure fit was not based directly  on
# data  from  Carpenter,  but  was derived using  the  overpressure  fit  and
# Rankine-Hugoniot   theory.    In   fact,   very  little  dynamic   pressure
# experimental data exists.   However,  when compared with a major  hydrocode
# program  REFLECT-4,  the dynamic pressure fit had an average difference  of
# +-32%. When the data points used in the comparison were limited to the Mach
# region the average difference dropped to +-19%.

# The  pressure  versus time fits (impulses and waveform profiles)  also
# have  been  compared with the REFLECT-4 results.   The  overpressure  total
# impulse was off in the worst case by about 20% and on the average by  about
# +-10%.  The  overpressure  positive phase duration comparison showed  large
# differences:  40% in the worst case and +-15% on the average.   The general
# tendency of the errors was to increase with scaled height of burst with the
# worst case occurring above 200 meters.

# The discrepancies between REFLECT-4 and the dynamic pressure  waveform
# fits  are more compartmentalized;  that is,  the worst case errors tend  to
# occur  in  well-defined regions.   In comparing the dynamic pressure  total
# impulse with the REFLECT-4 results,  for scaled ground range less than  300
# meters  the average difference is in the 30% to 40% range,  with errors  of
# the  order of 80% for HOB below 60 meters and as large as 60% for HOB above
# 200 meters.   For scaled ground ranges greater than 300 meters, the average
# error  is better than +-5%, with worst cases of about 15% for HOB below  60
# meters.

# The  dynamic  pressure  positive  phase duration  was  fit  using  the
# REFLECT-4  data.   The largest differences occur for the region  of  scaled
# ground ranges less than 300 meters and for scaled heights of burst below 60
# meters.   In  this region the worst case error is about 35% and the average
# around  10%.   Outside this region the difference decreases  substantially.
# The  worst case discrepancy is less than 8% and the average is better  than
# 3%.

# The  accuracy  of the Gauss-Legendre Quadrature (20  points)  used  to
# compute the impulse integrals has been compared extensively with that given
# by Simpson's rule using about 2000 time points per integral. The worst case
# error  for  partial and total impulses (dynamic pressure  and  overpressure
# waveforms) is found to be less than 1%.

# SOURCES OF DATA
# Bleakney,  W.,  and A.  H.  Taub,  "Interaction of Shock Waves," Reviews of
# Modern Physics, Vol. 21, p. 584, October 1949.

# Carpenter,  H.  J., Height-of-Burst Curves, RDA letter to M. Atkins, E. E.
# Conrad,  J. F. Moulton, Jr., E. Sevin, and G. W. Ullrich, dated 8 June
# 1978.

# DNA Effects  Manual Number 1.  Capabilities of  Nuclear  Weapons,  Stanford
# Research Institute, (unpublished).

# Hikida,  Shuichi, "Triple Point Path Fit (1 kt)," S-Cubed, Albuquerque, NM,
# letter to Tom Schroeder, Horizons Technology, Inc., dated 12 May 1982.

# Sachs,  Dr. Donald, editor, EM-1 Air Blast Phenomena (Preliminary Version),
# Kaman Sciences Corporation KW-81-11U(R), 2 February 1981.

# Smiley,  Robert F., J. Ray Ruetenik, and Michael A. Tomayko, REFLECT-4 Code
# Computations  of 40 kT Nuclear Blast Waves Reflected from the  Ground,
# Kaman Avidyne KA TR-201, 1 November 1982.

# Speicher, S. J., and H. L. Brode, Airblast Overpressure Analytic Expression 
# for Burst Height,  Range and Time Over an Ideal Surface, PSR Note 385,
# Pacific-Sierra Research Corporation, Los Angeles, CA, November 1981.

# Speicher,  S.  J., "PSR QUICK FIX Analytic Expression for Dynamic Pressure-
# Time versus Range,  Burst Height,  and Yield," Pacific-Sierra Research
# Corporation,  Los  Angeles,  CA,  letter to Horizon Technology,  Inc.,
# dated 20 June 1983.

# OTHER REFERENCES
# Brode,  H.  L., Height-of-Burst Effects at High Overpressures,  RAND
# Corporation, RM-6301-DASA, DASA 2506, July 1970.

# id., Theoretical  Descriptions  of the Blast and Fireball for  a  Sea
# Level Megaton Explosion,  RM-2248, The Rand Corporation, Santa Monica,
# 850CA, 1959.

# id., Analytic Approximations to Dynamic Pressure and Impulse and Other
# Fits  for Nuclear Blasts,  PSR Note 529,  Pacific-Sierra Research Cor-
# poration, Los Angeles, CA, 20 May 1983.

# Carpenter,  H.  Jerry, Overpressure Impulse HOB Curves,  RDA letter to  M.
# Atkins,    J. Moulton, E. Sevin, G. Stockton, and G. Ullrich, dated 31
# August 1976.

# Speicher,  S.  J.,  and  H.  L.  Brode,
# Revised Procedure for the Analytic Approximation of Dynamic Pressure Versus Time,
# PSR Note 320, Pacific-Sierra Research Corporation, Los Angeles, CA, May 1980.


# Altitude scaling factors (SP, SD, and ST)

def _altitude_t(alt):
    if 0 <= alt < 11000:
        return 1 - (2 * 10**9)**-0.5 * alt
    if 11000 <= alt < 20000:
        return 0.7535 * (1 + (2.09 * 10**-7) * alt)
    if alt >= 20000:
        return 0.684 *  (1 + (5.16 * 10**-6) * alt)

def _altitude_p(alt):
    if 0 <= alt < 11000:
        return _altitude_t(alt)**5.3
    if 11000 <= alt < 20000:
        return 1.6**0.5 * (1 + (2.09 * 10**-7) * alt)**-754 
    if alt >= 20000:
        return 1.4762 *  (1 + (5.16 * 10**-6) * alt)**-33.6

def _altitude_sp(alt):
    return _altitude_p(alt)

def _altitude_sd(alt):
    return _altitude_sp(alt)**(-1.0/3)

def _altitude_st(alt):
    return _altitude_sd(alt) * _altitude_t(alt)**-0.5

# 'Altitude-dependent speed of sound'
# 'rule of thumb: C increases 1.8% for each 10 deg. C rise above 15 deg. C'

def _altitude_speed_of_sound(alt):
    return (340.5 * _altitude_sd(alt)) / _altitude_st(alt)

# The Defense Nuclear Agency 1kT Free-Air Overpressure Standard
def _DNA1kTfreeairop(r):
    return (3.04 * 10**11)/r**3 + (1.13 * 10**9)/r**2 + (7.9 * 10**6)/(r * (np.log(r / 445.42 + 3 * np.exp(np.sqrt(r / 445.42) / -3.0)))**0.5)

# PFREE
def _DNAfreeairpeakop(r, y, alt):
    r1 = r / (_altitude_sd(alt) * y**(1.0/3))
    return _DNA1kTfreeairop(r1) * _altitide_sp(alt)

# r in these functions is not scaled for yield to facilitate their re-use for the
# airburst functions. 
def _shock_strength(op):
    return op / 101325 + 1

def _shock_gamma(op):
    xi = _shock_strength(op)
    t = 10**-12 * xi**6
    z = np.log(xi) - (0.47 * t) / (100 + t)
    return 1.402 - (3.4 * 10**-4 * z**4) / (1+ 2.22 * 10**-5 * z**6)

def _shock_mu(g):
    return (g + 1) / (g - 1)

def _mass_density_ratio(op):
    xi = _shock_strength(op)
    mu = _shock_mu(_shock_gamma(op))
    return (1 + mu * xi) / (5.975 + xi)

def _DNA1kTfreeairdyn(r):
    op = _DNA1kTfreeairop(r)
    return 0.5 * op * (_mass_density_ratio(op) - 1)

# QFREE
def _DNAfreeairpeakdyn(r, y, alt):
    r1 = r / (_altitude_sd(alt) * y**(1.0/3))
    return _DNA1kTfreeairdyn(r1) * _altitide_sp(alt)

def _scaledfreeairblastwavetoa(r):
    r2 = r * r
    return (r2 * (6.7 + r)) / (7.12e6 + 7.32e4 * r + 340.5 * r2)

def _freeairblastwavetoa(r, y, alt):
    return _scaledfreeairblastwavetoa(r) * _altitude_st(alt) * y**(1.0/3)

# Rankine-Hugoniot Factors
def _normal_reflection_factor(op):
    g = _shock_gamma(op)
    n = _mass_density_ratio(op)
    return 2 + ((g + 1) * (n - 1)) / 2

def _peak_particle_mach_number(pfree):
    n = _mass_density_ratio(pfree)
    return ((pfree * (1 - (1 /n))) / 142000)**0.5

def _shock_front_mach_number(pfree):
    n = _mass_density_ratio(pfree)
    vc = _peak_particle_mach_number(pfree)
    return vc / (1 - 1 / n)

def _scale_slant_range(r, y, alt):
    sgr = r / y**(1.0/3)
    shob = alt / y**(1.0/3)
    return np.sqrt(sgr**2 + shob**2)

def _regular_mach_merge_angle(r, y, alt):
    pfree = _DNA1kTfreeairop(_scale_slant_range(r, y, alt))
    t = 340 / pfree**0.55
    u = 1 / (7782 * pfree**0.7 + 0.9)
    return np.arctan(1 / (t + u))

def _merge_region_width(r, y, alt):
    pfree = _DNA1kTfreeairop(_scale_slant_range(r, y, alt))
    t = 340 / pfree**0.55
    w = 1 / (7473 * pfree**0.5 + 6.6)
    v = 1 / (647 * pfree**0.8 + w)
    return np.arctan(1 / (t + v))

def _regular_mach_switching_parameter(r, y, alt):
    sgr = r / y**(1.0/3)
    shob = alt / y**(1.0/3)
    alpha = np.arctan(shob / sgr)
    s = (alpha - _regular_mach_merge_angle(r, y, alt)) / _merge_region_width(r, y, alt)
    s0 = max(min (s, 1), -1)
    return 0.5 * (np.sin(0.5 * np.pi * s0) + 1)

def _p_mach(r, y, alt):
    sgr = r / y**(1.0/3)
    shob = alt / y**(1.0/3)
    alpha = np.arctan(shob / sgr)
    a = min(3.7 - 0.94 * np.log(sgr), 0.7)
    b = 0.77 * np.log(sgr) - 3.8 - 18 / sgr
    c = max(a, b)
    return _DNA1kTfreeairop(sgr / 2**(1.0/3)) / (1 - c * np.sin(alpha))

def _p_reg(r, y, alt):
    sgr = r / y**(1.0/3)
    pfree = _DNA1kTfreeairop(_scale_slant_range(r, y, alt))
    shob = alt / y**(1.0/3)
    alpha = np.arctan(shob / sgr)
    r_n = 2 + ((_shock_gamma(pfree) + 1) * (_mass_density_ratio(pfree) - 1)) / 2
    f = pfree / 75842
    d = (f**6 * (1.2 + 0.07 * f**0.5) ) / (f**6 + 1)
    return pfree * ((r_n - 2) * np.sin(alpha)**d + 2)

# PAIR
def _DNAairburstpeakop(r, y, alt):
    sigma = _regular_mach_switching_parameter(r, y, alt)
    if sigma == 0:
        return _p_mach(r, y, alt)
    elif 0 < sigma < 1:
        return _p_reg(r, y, alt) * sigma + _p_mach(r, y, alt) * (1 - sigma)
    elif sigma == 1:
        return _p_reg(r, y, alt)

#QAIR
def _DNAairburstpeakdyn(r, y, alt):
    pair = _DNAairburstpeakop(r, y, alt)
    sigma = _regular_mach_switching_parameter(r, y, alt)
    sgr = r / y**(1.0/3)
    shob = alt / y**(1.0/3)
    alpha = np.arctan(shob / sgr)
    n_q = _mass_density_ratio(pair)
    return 0.5 * pair * (n_q - 1) * (1 - (sigma * np.sin(alpha)**2))

# Airburst blast wave time-of-arrival
def _scaledmachstemformationrange(y, alt):
    shob = alt / y**(1.0/3)
    return shob**2.5 / 5822 + 2.09 * shob**0.75

def _slantrangescalingfactor(r, y, alt):
    sgr = r / y**(1.0/3)
    x_m = _scaledmachstemformationrange(y, alt)
    if sgr <= x_m:
        return 1
    else:
        return 1.26 - 0.26 * (x_m / sgr)

# TAAIR
def _airburstblastwavetoa(r, y, alt):
    v = _slantrangescalingfactor(r, y, alt)
    r1 = _scale_slant_range(r, y, alt) / v
    ta_air = _scaledfreeairblastwavetoa(r1)
    return ta_air * y**(1.0/3) * v

# Overpressure total impulse

def _scaledopposphasedursurf(r, y, alt):
    v = _slantrangescalingfactor(r, y, alt)
    r1 = _scale_slant_range(r, y, alt) / v
    ta_air = _scaledfreeairblastwavetoa(r1)
    t_0 = np.log(1000 * ta_air) / 3.77
    return (155 * np.exp(-20.8 * ta_air) + np.exp(-t_0**2 + 4.86 * t_0 + 0.25)) / 1000

def _scaledopposphasedur(r, y, alt):
    shob = alt / y**(1.0/3)
    v = _slantrangescalingfactor(r, y, alt)
    r1 = _scale_slant_range(r, y, alt) / v
    ta_air = _scaledfreeairblastwavetoa(r1) * v
    t_0 = np.log(1000 * ta_air) / 3.77
    dp_surf = (155 * np.exp(-20.8 * ta_air) + np.exp(-t_0**2 + 4.86 * t_0 + 0.25)) / 1000
    dp_unmod = dp_surf * (1 - (1 - 1 / (1 + 4.5e-8 * shob**7)) * (0.04 + 0.61 / (1 + ta_air**1.5 / 0.027)))
    return dp_unmod * 1.16 * np.exp(-abs(shob / 0.3048 - 156) / 1062)

# Functions for double-peak overpressure waveforms

def _peaksequalityapprox(shob):
    return 138.3 / (1 + 45.5 / shob)

def _peakstimeseperationapprox(shob, sgr, x_m):
    return max(shob / 8.186e5 * (sgr - x_m)**1.25, 1e-12)

def _DNA_b(sgr, shob, ta_air, dp, t):
    """A lot of physically meaningless internal functions used in the DNA overpressure and dynamic pressure calculations."""
    s = 1 - 1 / (1 + (1 / 4.5e-8 * shob**7)) - ((5.958e-3 * shob**2) / (1 + 3.682e-7 * shob**7)) / (1 + sgr**10 / 3.052e14)
    f = s * ((2.627 * ta_air**0.75) / (1 + 5.836 * ta_air) + (2341 * ta_air**2.5) / (1 + 2.541e6 * ta_air**4.75  - 0.216)) + 0.7076 - 3.077 / (1e-4 * ta_air**-3 + 4.367)
    g = 10 + s * (77.58 - 154 * ta_air**0.125 / (1 + 1.375 * np.sqrt(ta_air)))
    h = s * ((17.69 * ta_air) / (1 + 1803 * ta_air**4.25) - (180.5 * ta_air**1.25) / (1 + 99140 * ta_air**4) - 1.6) + 2.753 + 56 * ta_air /(1 + 1.473e6 * ta_air**5)
    return (f * (ta_air / t)**g + (1 - f) * (ta_air / t)**h) * (1 - (t - ta_air) / dp)

def _opatscaledtime(r, y, alt, sgr, shob, x_m, ta_air, dp, t):
    s = 1 - 1 / (1 + (1 / 4.5e-8 * shob**7)) - ((5.958e-3 * shob**2) / (1 + 3.682e-7 * shob**7)) / (1 + sgr**10 / 3.052e14)
    f = s * ((2.627 * ta_air**0.75) / (1 + 5.836 * ta_air) + (2341 * ta_air**2.5) / (1 + 2.541e6 * ta_air**4.75  - 0.216)) + 0.7076 - 3.077 / (1e-4 * ta_air**-3 + 4.367)
    g = 10 + s * (77.58 - 154 * ta_air**0.125 / (1 + 1.375 * np.sqrt(ta_air)))
    h = s * ((17.69 * ta_air) / (1 + 1803 * ta_air**4.25) - (180.5 * ta_air**1.25) / (1 + 99140 * ta_air**4) - 1.6) + 2.753 + 56 * ta_air /(1 + 1.473e6 * ta_air**5)
    b = (f * (ta_air / t)**g + (1 - f) * (ta_air / t)**h) * (1 - (t - ta_air) / dp)
    if x_m > sgr or shob > 116:
        return _DNAairburstpeakop(r, y, alt) * b
    else:
        x_e = _peaksequalityapprox(shob)
        e = max(min(abs((sgr - x_m) / (x_e - sgr)), 50), 0.02)
        w = 0.583 / (1 + 2477 / shob**2)
        d = 0.23 + w + 0.27 * e + e**5 * (0.5 - w)
        a = (d - 1) * (1 - 1 / (1 + e**-20))
        dt = _peakstimeseperationapprox(shob, sgr, x_m)
        v_0 = shob**6 / (2445 * (1 + shob**6.75 / 3.9e4) * (1 + 9.23 * e**2))
        c_0 = (1.04 - 1.04 / (1 + 3.725e7 / sgr**4)) / ((a + 1) * (1 + 9.872e8 / shob**9))
        g_a = max(min((t - ta_air) / dt, 400), 0.0001)
        v = 1 + v_0 * g_a**3 / (g_a**3 + 6.13)
        c = c_0 * (1 / (g_a**-7 + 0.923 * g_a**1.5)) * (1 - ((t - ta_air) / dp)**8)
        return _DNAairburstpeakop(r, y, alt) * (1 + a) * (b * v + c)

def _overpressureatscaledtime(r, y, alt, t):
    sgr = r / y**(1.0/3)
    shob = alt / y**(1.0/3)
    x_m = _scaledmachstemformationrange(y, alt)
    v = _slantrangescalingfactor(r, y, alt)
    r1 = _scale_slant_range(r, y, alt) / v
    ta_air = _scaledfreeairblastwavetoa(r1) * v
    dp = _scaledopposphasedur(r, y, alt)
    return _opatscaledtime(r, y, alt, sgr, shob, x_m, ta_air, dp, t)

# In lieu of the 20-point Gauss-Legendre quadrature used in the original
# BLAST.EXE, this fuction uses scipy.integrate.quad to call the venerable FORTRAN
# library QUADPACK and perform a Gauss-Kronod quadrature. This appears
# to be more accurate than the BLAST.EXE help file claims for the original
# approach, which is not surprising as it uses an adaptive algorithm that
# attempts to reduce error to within a particular tolerance.

# IPTOTAL
def _overpressuretotalimpulse(r, y, alt):
    sgr = r / y**(1.0/3)
    shob = alt / y**(1.0/3)
    x_m = _scaledmachstemformationrange(y, alt)
    v = _slantrangescalingfactor(r, y, alt)
    r1 = _scale_slant_range(r, y, alt) / v
    ta_air = _scaledfreeairblastwavetoa(r1) * v
    dp = _scaledopposphasedur(r, y, alt)
    t_p = 13 * (ta_air + dp) / 14
    scaled_impulse, _ = quad(lambda t: _opatscaledtime(r, y, alt, sgr, shob, x_m, ta_air, dp, t), ta_air, ta_air + dp)
    return scaled_impulse * y**(1.0/3)

def _dpt(shob, sgr, x_m, ta_air, dp_q, pair, b, t):
    if x_m > sgr or shob > 116:
        return pair * b
    else:
        dt = _peakstimeseperationapprox(shob, sgr, x_m)
        g_a = max(min((t - ta_air) / dt, 400), 0.0001)
        x_e = _peaksequalityapprox(shob)
        e = max(min(abs((sgr - x_m) / (x_e - sgr)), 50), 0.02)
        w = 0.583 / (1 + 2477 / shob**2)
        d = 0.23 + w + 0.27 * e + e**5 * (0.5 - w)
        a = (d - 1) * (1 - 1 / (1 + e**-20))
        v_0 = shob**6 / (2445 * (1 + shob**6.75 / 3.9e4) * (1 + 9.23 * e**2))
        c_0 = (1.04 - 1.04 / (1 + 3.725e7 / sgr**4)) / ((a + 1) * (1 + 9.872e8 / shob**9))
        v = 1 + v_0 * g_a**3 / (g_a**3 + 6.13)
        c = c_0 * (1 / (g_a**-7 + 0.923 * g_a**1.5)) * (1 - ((t- ta_air) / dp_q)**8)
        return pair * (1 + a) * (b * v + c)
        
# Dynamic pressure total impulse

def _dynamicpressureatscaledtime(r, y, alt, t):
    pair = _DNAairburstpeakop(r, y, alt)
    sgr = r / y**(1.0/3)
    shob = alt / y**(1.0/3)
    x_m = _scaledmachstemformationrange(y, alt)
    v = _slantrangescalingfactor(r, y, alt)
    sr = _scale_slant_range(r, y, alt)
    ta_air = _scaledfreeairblastwavetoa(sr / v) * v
    shob_0 = shob / 0.3048
    sgr_0 = sgr / 0.3048
    shob_x = abs(shob_0 - 200) + 200
    sgr_x = sgr_0 - 200
    dp_0 = 0.3 + 0.42 * np.exp(-shob_x / 131)
    if sgr_x > 0:
        dp_x = dp_0 + 4.4e-5 * sgr_x
    else:
        dp_x = dp_0 + sgr_x * (1.0 / 2361 - abs(shob_x -533)**2 / 7.88e7)
    if shob_0 >= 200:
        dp_q = dp_x
    else:
        dp_q = dp_x * (1 + 0.2 * np.sin(shob_0 * np.pi /200))
    delta_0 = max(shob_0**1.52 / 16330 - 0.29, 0)
    delta = 2.38 * np.exp(-7e-7 * abs(shob_0 - 750)**2.7 - 4e-7 * sgr_0**2) + delta_0
    b = _DNA_b(sgr, shob, ta_air, dp_q, t)
    dpt = _dpt(shob, sgr, x_m, ta_air, dp_q, pair, b, t)
    n_q = _mass_density_ratio(dpt)
    return 0.5 * dpt * (n_q - 1) * (dpt / pair)**delta

def DNA_static_overpressure(y, r, h, yunits='kT', dunits='m', opunits='kg/cm^2'):
    """Estimate peak static overpressure at range r from a burst of yield y using the
the Defense Nuclear Agency 1kT standard free airburst overpressure. This assumes a
thermally ideal surface."""
    yld = convert_units(y, yunits, 'kT')
    gr = convert_units(r, dunits, 'm')
    height = convert_units(h, dunits, 'm')
    op = _DNAairburstpeakop(gr, yld, height)
    return convert_units(op, 'Pa', opunits)

def DNA_dynamic_pressure(y, r, h, yunits='kT', dunits='m', opunits='kg/cm^2'):
    """Estimate peak pynamic overpressure at range r from a burst of yield y using the
the Defense Nuclear Agency 1kT standard free airburst overpressure, assuming an ideal
surface. Many real-world surfaces are not ideal (most, in the opinion of Soviet 
analysts), meaning that this function has only limited predictove capability."""
    yld = convert_units(y, yunits, 'kT')
    gr = convert_units(r, dunits, 'm')
    height = convert_units(h, dunits, 'm')
    dyn = _DNAairburstpeakdyn(gr, yld, height)
    return convert_units(dyn, 'Pa', opunits)

# Soviet overpressure functions from _Iadernoe oruzhie_. The aim here is to stick as
# close to the original manual as possible, hence the use of these raw values
# extracted from the graphs using GraphClick. Attempts to fit these points to a
# bivariate spline produced more elegant code but unacceptable results. The spline
# overfit to the 0.0, 70.0, 120.0, and 200.0m height-of-burst curves, with deep
# valleys in between producing physically unrealistic values. While this code is
# aesthetically quationable (much of it was generated programmatically) it gives a
# plausible reading of the Soviet plots for all scaled HOBs to 200m out to scale range
# 5000m.

_soviet_mach_sh20x = [1.331, 19.104, 41.226, 63.169, 80.125, 101.245, 120.583, 139.581, 159.644, 179.461, 201.321, 219.973, 241.426, 262.938, 283.481, 302.758, 325.245, 344.411, 366.451, 385.868, 406.185, 421.308, 438.554, 461.104, 481.968, 502.29, 521.382, 539.832, 560.252, 581.467, 602.997, 624.007, 644.188, 664.264, 685.195, 701.97, 719.388, 742.154, 759.153, 782.1, 801.371, 821.661, 838.498, 859.418, 881.444, 900.283, 918.763, 937.479, 959.438, 981.26, 998.157, 1018.784, 1036.762, 1058.714, 1076.708, 1096.315, 1115.931, 1135.326, 1155.808, 1176.399, 1195.328, 1215.69, 1238.752, 1257.313, 1277.394, 1297.996]

_soviet_mach_sh20y = [0.5523031, 0.54294986, 0.52608067, 0.49927458, 0.46671936, 0.43120286, 0.38863397, 0.3422252, 0.29534715, 0.24551266, 0.19700474, 0.14952701, 0.10209052, 0.06145246, 0.015359761, -0.022733787, -0.06198091, -0.09582564, -0.13548893, -0.16877033, -0.20134935, -0.22329882, -0.2502637, -0.28066874, -0.3080349, -0.3362991, -0.35951856, -0.38090667, -0.4089354, -0.43179828, -0.45717457, -0.48412615, -0.5044557, -0.5228787, -0.54363394, -0.5606673, -0.5783961, -0.5968795, -0.61261016, -0.6326441, -0.64589155, -0.6635403, -0.67366415, -0.6946486, -0.71219826, -0.7235382, -0.74232143, -0.75696194, -0.7670039, -0.78515613, -0.7986029, -0.8153086, -0.8268137, -0.838632, -0.8507809, -0.86012095, -0.8728952, -0.8827287, -0.8894103, -0.89619625, -0.91009486, -0.92081875, -0.928118, -0.93554205, -0.94309515, -0.95078194]

_soviet_mach_sh12x = [0.185, 18.063, 40.565, 62.433, 81.175, 100.851, 118.577, 140.272, 158.86, 179.04, 201.837, 220.533, 240.193, 261.484, 283.718, 303.67, 324.586, 343.316, 364.135, 385.013, 406.369, 420.938, 439.419, 460.575, 481.611, 503.282, 521.93, 539.092, 560.148, 580.711, 603.393, 625.366, 645.57, 664.924, 682.531, 702.317, 721.647, 741.642, 761.445, 780.592, 801.867, 822.177, 840.157, 859.306, 880.964, 899.578, 919.86, 939.25, 959.39, 981.444, 997.316, 1018.656, 1037.42, 1059.352, 1077.797, 1096.364, 1115.968, 1135.366, 1155.98, 1177.267, 1195.852, 1215.355, 1237.331, 1258.569, 1277.98, 1299.745]

_soviet_mach_sh12y = [1.1908078, 1.1579099, 1.0961798, 1.0184091, 0.941064, 0.83569056, 0.7261565, 0.59128726, 0.4835873, 0.38453263, 0.29336256, 0.22193561, 0.15198241, 0.08457629, 0.01870051, -0.029188387, -0.07623804, -0.11690664, -0.15926675, -0.19859628, -0.23507701, -0.2588484, -0.28903687, -0.3187588, -0.35164, -0.38510278, -0.40782323, -0.4341522, -0.46344155, -0.48811665, -0.51712644, -0.53910214, -0.5622495, -0.5783961, -0.59859943, -0.61618465, -0.6326441, -0.65169513, -0.66554624, -0.6798537, -0.6968039, -0.7144427, -0.72815835, -0.7447275, -0.75696194, -0.7695511, -0.78515613, -0.7986029, -0.8096683, -0.82390875, -0.83268267, -0.84163755, -0.8507809, -0.86327946, -0.8728952, -0.8827287, -0.89279, -0.90309, -0.91009486, -0.924453, -0.928118, -0.93930215, -0.9469216, -0.95467705, -0.9665762, -0.97469413]

_soviet_mach_sh7x = [43.498, 62.016, 80.472, 102.385, 118.651, 137.123, 159.124, 178.447, 199.181, 220.241, 240.446, 261.623, 282.019, 303.128, 324.82, 343.582, 365.69, 386.883, 407.316, 419.945, 438.6, 459.636, 482.27, 504.212, 521.759, 541.152, 558.706, 579.852, 604.245, 624.521, 644.971, 665.257, 683.297, 702.656, 721.246, 740.825, 758.882, 780.978, 801.944, 821.375, 840.749, 860.018, 882.071, 901.053, 919.378, 938.86, 960.238, 981.877, 997.851, 1019.741, 1038.082, 1057.774, 1079.01, 1096.82, 1116.995, 1135.905, 1156.171, 1178.157, 1197.581, 1216.761, 1237.352, 1259.419, 1277.991]

_soviet_mach_sh7y = [1.4558951, 1.3268068, 1.1763808, 0.9898501, 0.86093664, 0.72525805, 0.58782315, 0.46419135, 0.35755375, 0.24944296, 0.1547282, 0.08884456, 0.026124531, -0.03526908, -0.09691001, -0.13489604, -0.1811146, -0.22257319, -0.2620127, -0.28819278, -0.32239303, -0.3555614, -0.3882767, -0.4156688, -0.44129145, -0.46597388, -0.49214414, -0.5243288, -0.55284196, -0.5718652, -0.5968795, -0.61618465, -0.634512, -0.64975196, -0.66756153, -0.6819367, -0.6968039, -0.71669877, -0.73282826, -0.75202674, -0.7670039, -0.7798919, -0.79317415, -0.80410033, -0.8153086, -0.8268137, -0.84163755, -0.85387194, -0.86327946, -0.8696662, -0.87942606, -0.89279, -0.90309, -0.91364014, -0.92081875, -0.93181413, -0.94309515, -0.95078194, -0.9586073, -0.9706162, -0.9788107, -0.98716277, -0.9956786]

_soviet_nomach_sh20x = [0.776, 17.542, 39.194, 59.919, 80.395, 98.234, 120.384, 140.066, 160.045, 180.514, 200.089, 220.68, 241.361, 262.881, 282.052, 302.272, 319.874, 342.36, 361.619, 380.321, 401.146, 418.947, 440.827, 460.902, 480.084, 501.124, 520.191, 541.805, 558.649, 581.106, 600.363, 618.769, 640.37, 660.45, 680.663, 699.943, 717.806, 739.608, 759.72, 778.782, 799.109, 819.068, 838.982, 859.111, 878.461, 898.427, 918.219, 936.807, 958.894, 978.923, 997.298, 1017.674, 1037.8, 1056.481, 1077.19, 1098.258, 1118.988, 1137.451, 1158.16, 1178.221, 1200.057, 1220.432, 1238.064, 1259.942, 1280.013, 1298.179]

_soviet_nomach_sh20y = [0.47348696, 0.47085133, 0.44916973, 0.40636984, 0.33183205, 0.23426414, 0.12319806, 0.04257553, -0.026410384, -0.08777796, -0.13018179, -0.17783193, -0.21467015, -0.25103715, -0.27736607, -0.30451834, -0.32790214, -0.35066512, -0.36653155, -0.38721615, -0.40560743, -0.41793662, -0.4353339, -0.44249278, -0.45345733, -0.4596705, -0.47366074, -0.4867824, -0.49894074, -0.5128616, -0.5287083, -0.54363394, -0.5590909, -0.57675415, -0.59176004, -0.6055483, -0.6216021, -0.63827217, -0.65169513, -0.66554624, -0.6798537, -0.68824613, -0.7011469, -0.7144427, -0.73048705, -0.7375489, -0.75448734, -0.7619539, -0.7798919, -0.79048496, -0.7986029, -0.8096683, -0.82390875, -0.82973826, -0.838632, -0.8507809, -0.86012095, -0.8696662, -0.87614834, -0.88605666, -0.89619625, -0.90309, -0.91364014, -0.91721463, -0.924453, -0.93181413]

_soviet_nomach_sh12x = [0.631, 17.577, 39.951, 28.472, 58.118, 78.658, 97.327, 119.925, 139.845, 160.758, 180.155, 201.174, 220.079, 240.022, 262.906, 282.311, 300.743, 320.333, 341.52, 361.346, 381.298, 400.317, 418.635, 439.661, 459.697, 479.828, 501.458, 520.715, 541.523, 559.59, 581.485, 600.797, 619.104, 639.911, 660.035, 679.174, 698.348, 720.035, 740.337, 760.349, 779.427, 800.302, 818.658, 839.435, 859.027, 878.489, 899.365, 917.382, 938.439, 957.56, 977.898, 997.32, 1017.29, 1038.069, 1058.037, 1078.744, 1099.227, 1118.674, 1138.738, 1157.212, 1178.332, 1200.738, 1219.925, 1239.044, 1259.221, 1279.509, 1298.87]

_soviet_nomach_sh12y = [1.1146777, 1.0617163, 0.89641595, 0.9935685, 0.7189167, 0.5682017, 0.4556061, 0.35506824, 0.27669153, 0.20330492, 0.14113607, 0.08314413, 0.024895966, -0.024108874, -0.07520399, -0.114073664, -0.1518109, -0.19246496, -0.23358716, -0.2700257, -0.3053948, -0.33724216, -0.3635121, -0.38933983, -0.41453928, -0.43889862, -0.46344155, -0.47886193, -0.49894074, -0.52143353, -0.537602, -0.5590909, -0.57511836, -0.59516627, -0.6090649, -0.627088, -0.6401645, -0.653647, -0.66958624, -0.68402964, -0.6968039, -0.7099654, -0.71896666, -0.73048705, -0.74714696, -0.7619539, -0.77469075, -0.78515613, -0.8013429, -0.8068754, -0.82102305, -0.8268137, -0.83564717, -0.8477116, -0.85387194, -0.86012095, -0.87614834, -0.87942606, -0.88605666, -0.89619625, -0.90309, -0.91009486, -0.92081875, -0.928118, -0.93554205, -0.94309515, -0.9469216]

_soviet_nomach_sh7x = [39.016, 49.999, 58.378, 67.937, 79.137, 98.665, 107.882, 119.779, 141.147, 153.237, 180.792, 201.757, 220.338, 241.914, 262.073, 282.716, 303.068, 318.817, 339.29, 359.974, 381.646, 399.743, 420.145, 440.688, 460.453, 480.501, 501.541, 519.266, 541.776, 559.521, 581.203, 601.3, 618.193, 639.057, 660.737, 678.284, 699.991, 719.077, 740.641, 759.83, 778.266, 799.218, 820.055, 839.256, 858.756, 878.976, 899.119, 917.695, 936.99, 958.196, 978.491, 997.668, 1017.739, 1036.856, 1057.669, 1077.738, 1098.06, 1118.041, 1138.872, 1156.899, 1179.739, 1199.16, 1220.081, 1239.626, 1260.337, 1281.059, 1299.434]

_soviet_nomach_sh7y = [1.4701016, 1.3752978, 1.2849042, 1.0660276, 0.8939836, 0.64157325, 0.5370631, 0.43743342, 0.30297995, 0.24004978, 0.11193429, 0.044539776, -0.015922973, -0.068542145, -0.11804502, -0.15926675, -0.19722629, -0.22767828, -0.26440108, -0.3001623, -0.33161408, -0.36151075, -0.3936186, -0.4213608, -0.45345733, -0.47495517, -0.49349496, -0.5142786, -0.5346171, -0.55284196, -0.5702477, -0.5867002, -0.6055483, -0.61978877, -0.63827217, -0.6556077, -0.66958624, -0.6798537, -0.69897, -0.71219826, -0.7235382, -0.7399286, -0.75696194, -0.7670039, -0.77728355, -0.79048496, -0.80410033, -0.8096683, -0.82102305, -0.83268267, -0.838632, -0.8507809, -0.86012095, -0.86646104, -0.8728952, -0.8827287, -0.89619625, -0.90309, -0.9065783, -0.91721463, -0.924453, -0.93181413, -0.93930215, -0.94309515, -0.95078194, -0.9586073, -0.9665762]

_soviet_groundx = [66.257, 68.349, 72.247, 72.145, 74.648, 78.78, 82.873, 90.652, 98.896, 100.0, 130.0, 205.295, 306.881, 407.898, 507.448, 608.431, 704.076, 807.753, 907.192, 1007.669, 1109.542, 1202.889, 1308.551, 1405.661, 1504.885, 1599.907, 1704.178, 1800.224, 1899.995, 1997.408, 2095.903, 2193.049, 2273.675, 2420.979, 2818.851, 3207.894, 3609.952, 3992.678, 4402.044, 4807.183, 5205.226]

_soviet_groundy = [2.0277002, 1.9637926, 1.9058229, 1.847085, 1.7720796, 1.6970462, 1.5995556, 1.4737351, 1.312135, 1.0265741, 0.7101174, 0.31069332, -0.057991948, -0.29929632, -0.4609239, -0.5850267, -0.692504, -0.7798919, -0.8569852, -0.924453, -0.9788107, -1.031517, -1.0655016, -1.1023729, -1.1487416, -1.1739252, -1.2146702, -1.2441251, -1.2757242, -1.2924298, -1.3187587, -1.3467875, -1.3565474, -1.3872161, -1.4685211, -1.5528419, -1.6197888, -1.6777807, -1.7212464, -1.769551, -1.79588]

# These local functions interpolate along the lines found in the original graphs and
# raise a ValueOutsideGraphError if the requested value falls outside those in the
# Soviet sources

def _soviet_mach_sh20(range):
    if 0 <= range <= _soviet_mach_sh20x[-1]:
        return np.interp(range, _soviet_mach_sh20x, _soviet_mach_sh20y)
    else:
        raise ValueOutsideGraphError(range)

def _soviet_mach_sh12(range):
    if 0 <= range <= _soviet_mach_sh12x[-1]:
        return np.interp(range, _soviet_mach_sh12x, _soviet_mach_sh12y)
    else:
        raise ValueOutsideGraphError(range)

def _soviet_mach_sh7(range):
    if _soviet_mach_sh7x[0] <= range <= _soviet_mach_sh7x[-1]:
        return np.interp(range, _soviet_mach_sh7x, _soviet_mach_sh7y)
    else:
        raise ValueOutsideGraphError(range)

def _soviet_nomach_sh20(range):
    if 0 <= range <= _soviet_nomach_sh20x[-1]:
        return np.interp(range, _soviet_nomach_sh20x, _soviet_nomach_sh20y)
    else:
        raise ValueOutsideGraphError(range)

def _soviet_nomach_sh12(range):
    if 0 <= range <= _soviet_nomach_sh12x[-1]:
        return np.interp(range, _soviet_nomach_sh12x, _soviet_nomach_sh12y)
    else:
        raise ValueOutsideGraphError(range)

def _soviet_nomach_sh7(range):
    if _soviet_nomach_sh7x[0] <= range <= _soviet_nomach_sh7x[-1]:
        return np.interp(range, _soviet_nomach_sh7x, _soviet_nomach_sh7y)
    else:
        raise ValueOutsideGraphError(range)

def _soviet_ground(range):
    if _soviet_groundx[0] <= range <= _soviet_groundx[-1]:
        return np.interp(range, _soviet_groundx, _soviet_groundy)
    else:
        raise ValueOutsideGraphError(range)

def lerp10(h, h1, h2, o1, o2):
    """Returns 10**o, where o is the linear interpolation of value h between (h1, o1) 
and (h2, o2)."""
    return 10**np.interp(h, [h1, h2], [o1, o2])

def _sovietmach(scale_height, ground_range):
    if 120 <= scale_height <= 200:
        return lerp10(scale_height, 120, 200, _soviet_mach_sh12(ground_range), _soviet_mach_sh20(ground_range))
    elif 70 <= scale_height < 120:
        return lerp10(scale_height, 70, 120, _soviet_mach_sh7(ground_range), _soviet_mach_sh12(ground_range))
    elif 0 <= scale_height < 70:
        return lerp10(scale_height, 0, 70, _soviet_ground(ground_range), _soviet_mach_sh7(ground_range))
    else:
        raise ValueOutsideGraphError(scale_height)
        
def _sovietnomach(scale_height, ground_range):
    if 120 < scale_height <= 200:
        return lerp10(scale_height, 120, 200, _soviet_nomach_sh12(ground_range), _soviet_nomach_sh20(ground_range))
    elif 70 <= scale_height <= 120:
        return lerp10(scale_height, 70, 120, _soviet_nomach_sh7(ground_range), _soviet_nomach_sh12(ground_range))
    elif 0 <= scale_height < 70:
        return lerp10(scale_height, 0, 70, _soviet_ground(ground_range), _soviet_nomach_sh7(ground_range))
    else:
        raise ValueOutsideGraphError(scale_height)

_rsoviet_mach_sh20x = _soviet_mach_sh20x[::-1]

_rsoviet_mach_sh20y = _soviet_mach_sh20y[::-1]

_rsoviet_mach_sh12x = _soviet_mach_sh12x[::-1]

_rsoviet_mach_sh12y = _soviet_mach_sh12y[::-1]

_rsoviet_mach_sh7x = _soviet_mach_sh7x[::-1]

_rsoviet_mach_sh7y = _soviet_mach_sh7y[::-1]

_rsoviet_nomach_sh20x = _soviet_nomach_sh20x[::-1]

_rsoviet_nomach_sh20y = _soviet_nomach_sh20y[::-1]

_rsoviet_nomach_sh12x = _soviet_nomach_sh12x[::-1]

_rsoviet_nomach_sh12y = _soviet_nomach_sh12y[::-1]

_rsoviet_nomach_sh7x = _soviet_nomach_sh7x[::-1]

_rsoviet_nomach_sh7y = _soviet_nomach_sh7y[::-1]

_rsoviet_groundx = _soviet_groundx[::-1]

_rsoviet_groundy = _soviet_groundy[::-1]

def _rsoviet_mach_sh20(overpressure):
    if _rsoviet_mach_sh20y[0] <= overpressure <= _rsoviet_mach_sh20y[-1]:
        return np.interp(overpressure, _rsoviet_mach_sh20y, _rsoviet_mach_sh20x)
    else:
        raise ValueOutsideGraphError(overpressure)

def _rsoviet_mach_sh12(overpressure):
    if _rsoviet_mach_sh12y[0] <= overpressure <= _rsoviet_mach_sh12y[-1]:
        return np.interp(overpressure, _rsoviet_mach_sh12y, _rsoviet_mach_sh12x)
    else:
        raise ValueOutsideGraphError(overpressure)

def _rsoviet_mach_sh7(overpressure):
    if _rsoviet_mach_sh7y[0] <= overpressure <= _rsoviet_mach_sh7y[-1]:
        return np.interp(overpressure, _rsoviet_mach_sh7y, _rsoviet_mach_sh7x)
    else:
        raise ValueOutsideGraphError(overpressure)

def _rsoviet_nomach_sh20(overpressure):
    if _rsoviet_nomach_sh20y[0] <= overpressure <= _rsoviet_nomach_sh20y[-1]:
        return np.interp(overpressure, _rsoviet_nomach_sh20y, _rsoviet_nomach_sh20x)
    else:
        raise ValueOutsideGraphError(overpressure)

def _rsoviet_nomach_sh12(overpressure):
    if _rsoviet_nomach_sh12y[0] <= overpressure <= _rsoviet_nomach_sh12y[-1]:
        return np.interp(overpressure, _rsoviet_nomach_sh12y, _rsoviet_nomach_sh12x)
    else:
        raise ValueOutsideGraphError(overpressure)

def _rsoviet_nomach_sh7(overpressure):
    if _rsoviet_nomach_sh7y[0] <= overpressure <= _rsoviet_nomach_sh7y[-1]:
        return np.interp(overpressure, _rsoviet_nomach_sh7y, _rsoviet_nomach_sh7x)
    else:
        raise ValueOutsideGraphError(overpressure)

def _rsoviet_ground(overpressure):
    if _rsoviet_groundy[0] <= overpressure <= _rsoviet_groundy[-1]:
        return np.interp(overpressure, _rsoviet_groundy, _rsoviet_groundx)
    else:
        raise ValueOutsideGraphError(overpressure)

# Something of a kludge, but this turned out to be necessary to get sensible
# output when scale_height > 120 and overpressures were higher than the maximum for
# scale_height == 200 on the original graph
def _rsovietnomach(scale_height, overpressure):
    logop = np.log10(overpressure)
    if scale_height >= 120 and overpressure > 2.975:
        l = lambda x: np.log10(_sovietnomach(scale_height, x))
        distances = [x*10 for x in range(11)][::-1]
        return np.interp(logop, map(l, distances), distances)
    elif 120 <= scale_height <= 200:
        return np.interp(scale_height, [120, 200], [_rsoviet_nomach_sh12(logop), _rsoviet_nomach_sh20(logop)])
    elif 70 <= scale_height < 120:
        return np.interp(scale_height, [70, 120], [_rsoviet_nomach_sh7(logop), _rsoviet_nomach_sh12(logop)])
    elif 0 <= scale_height < 70:
        return np.interp(scale_height, [0, 70], [_rsoviet_ground(logop), _rsoviet_nomach_sh7(logop)])
    else:
        raise ValueOutsideGraphError(scale_height)

def _rsovietmach(scale_height, overpressure):
    logop = np.log10(overpressure)
    if scale_height >= 120 and overpressure > 2.2336:
        l = lambda x: np.log10(_sovietmach(scale_height, x))
        distances = [x*10 for x in range(18)][::-1]
        return np.interp(logop, map(l, distances), distances)
    elif 120 <= scale_height <= 200:
        return np.interp(scale_height, [120, 200], [_rsoviet_mach_sh12(logop), _rsoviet_mach_sh20(logop)])
    elif 70 <= scale_height < 120:
        return np.interp(scale_height, [70, 120], [_rsoviet_mach_sh7(logop), _rsoviet_mach_sh12(logop)])
    elif 0 <= scale_height < 70:
        return np.interp(scale_height, [0, 70], [_rsoviet_ground(logop), _rsoviet_mach_sh7(logop)])
    else:
        raise ValueOutsideGraphError(scale_height)

def soviet_overpressure(y, r, h, thermal_layer=True, yunits='kT', dunits='m', opunits='kg/cm^2'):
    """Estimate peak static overpressure at radius r from the epicenter based on the
graphs in the 1987 Soviet military publication _Iadernoe oruzhie: Posbie dlia 
ofitserov_. 

The most interesting feature of this model is that it provides for cases
in which the Mach stem is suppressed by a thermal layer. This phenomenon was 
considered a largely theoretical 'second-order effect' among most US NWE researchers,
but was observed in extreme forms in the USSR's atmospheric nuclear tests, leading
them to conclude it would occur in many real-world military scenarios. To use a 
Soviet model with a Mach stem present, set the parameter thermal_layer to False."""
    yld = convert_units(y, yunits, 'kT')
    gr = convert_units(r, dunits, 'm')
    height = convert_units(h, dunits, 'm')
    sr = scale_range(yld, gr)
    sh = scale_height(yld, height)
    if thermal_layer:
        return convert_units(_sovietnomach(sh, sr), 'kg/cm^2', opunits)
    else:
        return convert_units(_sovietmach(sh, sr), 'kg/cm^2', opunits)

def r_soviet_overpressure(y, op, h, thermal_layer=True, yunits='kT', dunits='m', opunits='kg/cm^2'):
    """Estimate the radius from the epicenter at which peak static overpressure op 
will be experienced based graphs in the 1987 Soviet military publication _Iadernoe 
oruzhie: Posbie dlia ofitserov_. 

The most interesting feature of this model is that it provides for cases
in which the Mach stem is suppressed by a thermal layer. This phenomenon was 
considered a largely theoretical 'second-order effect' among most US NWE researchers,
but was observed in extreme forms in the USSR's atmospheric nuclear tests, leading
them to conclude it would occur in many real-world military scenarios. To use a 
Soviet model with a Mach stem present, set the parameter thermal_layer to False."""
    yld = convert_units(y, yunits, 'kT')
    height = convert_units(h, dunits, 'm')
    sh = scale_height(yld, height)
    overp = convert_units(op, opunits, 'kg/cm^2')
    if thermal_layer:
        return convert_units(_rsovietnomach(sh, overp), 'm', dunits) * y**0.3333333333333333
    else:
        return convert_units(_rsovietmach(sh, overp), 'm', dunits) * y**0.3333333333333333
