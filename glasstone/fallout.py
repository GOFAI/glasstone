# WSEG-10 fallout model as described in Dan W. Hanifen, "Documentation and Analysis
# of the WSEG-10 Fallout Prediction Model," Thesis, Air Force Institute of Technology,
# March 1980.

# WSEG-10 was originally developed by the Weapons Systems Evaluation Group in 1959
# with the goal of providing "an inexpensive, easy to use, analytical fallout 
# prediction code." According to Hanifen, in 1980 it was "the most popular analytical 
# fallout prediction model in use over the past twenty years." In light of its
# historical application for damage assessment modeling, as well as its comparative
# simplicity, Vincent Jodoin of Oak Ridge National Laboratory recommended its
# inclusion in the glasstone nuclear weapons effects library.

# This implementation of WSEG-10 aims to provide a portable Python version of
# the model as described in Hanifen's 1980 AFIT thesis.

# As with all fallout models, WSEG-10 is not actually predictive. Instead it
# provides a plausible prediction of a "mean case" fallout field, in this case
# an elliptical region following the "hotline" downwind from the burst point, with
# a Gaussian distribution crosswind and an upwind correction factor. While
# simplistic, these assumptions are not necessarily "worse" than those used in
# more sophisticated fallout models. While "disk-tosser" models such as DELFIC
# and SEER produce more complicted fallout fields than an "empirical" model
# such as WSEG-10, the additional complexity of these models does not translate
# into increased predictive accuracy except with accurate input data, and
# given uncertainties in the model design (cloud rise, particle size/activity
# distributions, etc.) perhaps not even then. For basic damage assessment
# studies, WSEG-10 is still adequate when employed with a correct understanding of
# its strengths and limitations.

import numpy as np
from scipy.special import gamma
from scipy.stats import norm
from scipy.integrate import quad
from affine import Affine
from glasstone.utilities import convert_units

class WSEG10:
    def __init__(self, gzx, gzy, yld, ff, wind, wd, shear, tob=0, dunits='km', wunits='km/h', shearunits='m/s-km', yunits='kT'):
        self.translation = ~Affine.translation(convert_units(gzx, dunits, 'mi'), convert_units(gzy, dunits, 'mi')) # translate coordinates relative to GZ (st. mi)
        self.wd = wd # wind direction in degrees (0=N, 90=E, etc.)
        self.yld = convert_units(yld, yunits, 'MT') # yield (MT)
        self.ff = ff # fission fraction, 0 < ff <= 1.0
        self.wind = convert_units(wind, wunits, 'mph') # wind speed (mi/hr)
        self.shear = convert_units(shear, shearunits, 'mph/kilofoot') # wind shear in mi/hr-kilofoot
        self.tob = tob # time of burst (hrs)
        # FORTRAN is ugly in any language
        # Store these values in the WSEG10 object to avoid recalculating them
        d = np.log(self.yld) + 2.42 # physically meaningless, but occurs twice
        # According to Hanifen, "The cloud is initially formed because the nuclear
        # fireball vaporizes both the surface of the earth at ground zero and the
        # weapon itself. The activity contained in the cloud is both neutron induced
        # and fission. After formation, the fireball rises and begins to cool at its
        # outer edges faster than the center thereby creating the typical torroidal
        # currents associated with the nuclear cloud. WSEG arbitrarily assumes that
        # the cloud will rise to a maximum center height within fifteen minutes and
        # then stabilize." 
        self.H_c =  44 + 6.1 * np.log(yld) - 0.205 * abs(d) * d # cloud center height
        lnyield = np.log(self.yld)
        self.s_0 = np.exp(0.7 + lnyield / 3 - 3.25 / (4.0 + (lnyield + 5.4)**2)) #sigma_0
        self.s_02 = self.s_0**2
        self.s_h = 0.18 * self.H_c # sigma_h
        self.T_c = 1.0573203 * (12 * (self.H_c / 60) - 2.5 * (self.H_c / 60)**2) * (1 - 0.5 * np.exp(-1 * (self.H_c / 25)**2)) # time constant
        self.L_0 = wind * self.T_c # L_0, used by g(x)
        self.L_02 = self.L_0**2
        self.s_x2 = self.s_02 * (self.L_02 + 8 * self.s_02) / (self.L_02 + 2 * self.s_02)
        self.s_x = np.sqrt(self.s_x2) # sigma_x
        self.L_2 = self.L_02 + 2 * self.s_x2
        self.L = np.sqrt(self.L_2) # L
        self.n = (ff * self.L_02 + self.s_x2) / (self.L_02 + 0.5 * self.s_x2) # n
        self.a_1 = 1 / (1 + ((0.001 * self.H_c * wind) / self.s_0)) # alpha_1

    def g(self, x):
        """Fallout Deposition Distribution Function.

Throughout the growth and transport of the radioactive cloud there is a 
continual fall of particles back to the ground. WSEG states that there must be some 
function "g(t)" which describes the fractional rate of activity arrival on the ground
everywhere at some time t. The integral of this function, G(t), represents the
fraction of activity down at time t. This g(t) function will be independent of the
horizontal activity distribution and therefore independent of the growth of a with
time. On the other hand g(t) will be dependent on the initial vertical distribution
and the activity/size distribution which determines particle fall rate. This
arbitrary choice of g(t) is based on Rand calculations which assume an activity/size
distribution given by activity_size_distribution(). These calculations are neither
shown nor referenced in the original 1959 WSEG model. If the activity/size
distribution for a given set of initial conditions is different
than that given by activity_size_distribution(), the form of g(t) should change.
This is not possible under the WSEG model where the function g(t) is fixed. The only
possible compensation for various activity/size distributions results because T_c
varies with yield."""
        return np.exp(-(np.abs(x) / self.L)**self.n) / (self.L * gamma(1 + 1 / self.n))

    def phi(self, x):
        """Normalized Downwind and Upwind Distribution.

In order to predict upwind fallout and at the same time preserve normalization, a
function phi is empirically inserted."""
        w = (self.L_0 / self.L) * (x / (self.s_x * self.a_1))
        return norm.cdf(w)

    def D_Hplus1(self, x, y, dunits='km', doseunits='Sv'):
        """Returns dose rate at x, y at 1 hour after burst. This value includes dose rate from all activity that WILL be deposited at that location, not just that that has arrived by H+1 hr."""
        rx, ry = self.translation * (convert_units(x, dunits, 'mi'), convert_units(y, dunits, 'mi')) * ~Affine.rotation(-self.wd + 270)
        f_x = self.yld * 2e6 * self.phi(rx) * self.g(rx) * self.ff
        s_y = np.sqrt(self.s_02 + ((8 * abs(rx + 2 * self.s_x) * self.s_02) / self.L) + (2 * (self.s_x * self.T_c * self.s_h * self.shear)**2 / self.L_2) + (((rx + 2 * self.s_x) * self.L_0 * self.T_c * self.s_h * self.shear)**2 / self.L**4))
        a_2 = 1 / (1 + ((0.001 * self.H_c * self.wind) / self.s_0) * (1 - norm.cdf(2 * x / self.wind)))
        f_y = np.exp(-0.5 * (ry / (a_2 * s_y))**2) / (2.5066282746310002 * s_y)
        return convert_units(f_x * f_y, 'Roentgen', doseunits)
        
    def fallouttoa(self, x):
        """Average time-of-arrival for fallout along hotline at x. Minimum is 0.5hr for any location."""
        T_1 = 1.0
        return np.sqrt(0.25 + (self.L_02 * (x + 2 * self.s_x2) * self.T_c**2) / (self.L_2 * (self.L_02 + 0.5 * self.s_x2)) + ((2 * self.s_x2 * T_1**2) / (self.L_02 + 0.5 * self.s_x)))

    def dose(self, x, y, dunits='km', doseunits='Sv'):
        """Estimate of total "Equivalent Residual Dose" (ERD) at location x, y from time of fallout arrival to 30 days, including a 90% recovery factor. 

"""
        rx, _ = self.translation * (convert_units(x, dunits, 'mi'), convert_units(y, dunits, 'mi')) * ~Affine.rotation(-self.wd + 270)
        t_a = self.fallouttoa(rx)
        # To obtain a measure of dose to humans, "Biological Dose" was defined as the
        # product of the DH+1 and a conversion factor, called Bio. Bio is an
        # empirical function depending on fallout arrival time and length of
        # exposure. Ten percent of the dose received is assumed irreparable and
        # ninety percent is assumed reparable with a thirty day time constant. This
        # was solved numerically and plotted as Dose vs. Time. Bio was then estimated
        # as Bio = (t / 19)**0.33, so that the dose at some time after activity
        # arrival is defined as Dose = DH+1 * Bio. Further refinements in the model
        # resulted in a second order approximation for Bio used here:
        bio = np.exp(-(0.287 + 0.52 * np.log(t_a / 31.6) + 0.04475 * np.log((t_a / 31.6)**2)))
        return self.D_Hplus1(x, y, dunits=dunits, doseunits=doseunits) * bio
