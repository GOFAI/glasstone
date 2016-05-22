import numpy as np
from glasstone.utilities import convert_units, ValueOutsideGraphError

# Thermal impulse functions from Ministerstvo Oborony SSSR, _Iadernoe oruzhie_ 4th ed.
# (Moscow: Izd-vo MO SSSR, 1987), pp. 49-50.

# This plot is a combination of 9 curves on two scales against a diagonal nomogram!

_sovietthermal1x = [-0.80410033, -0.6968039, -0.6143937, -0.5257837, -0.4213608, -0.326058, -0.23284414, -0.14691047, -0.041914154, 0.050766293, 0.14426279, 0.23704077, 0.33725953, 0.43328968, 0.51930285, 0.61267793, 0.7173376, 0.8061121, 0.8985058, 0.9845723, 1.0632958, 1.1452274, 1.2207355, 1.291724, 1.364926, 1.4221793, 1.4872091, 1.5449605, 1.6022445]

_sovietthermal1y= [1.988, 3.116, 3.94, 4.938, 6.006, 7.028, 8.016, 8.987, 10.014, 11.009, 12.034, 13.025, 13.995, 15.017, 15.985, 16.95, 17.983, 18.97, 19.991, 21.005, 21.968, 22.985, 24.034, 25.019, 26.019, 27.07, 28.1, 29.098, 30.099]

_sovietthermal2x = [-0.80410033, -0.6968039, -0.6143937, -0.5257837, -0.4213608, -0.326058, -0.23284414, -0.14691047, -0.041914154, 0.050766293, 0.14426279, 0.23704077, 0.33725953, 0.43328968, 0.51772356, 0.58557355, 0.6696887, 0.75572246, 0.844042, 0.92272544, 1.0051376, 1.0729113, 1.1370374, 1.2089517, 1.2720737, 1.3293775, 1.3838692, 1.4339138, 1.4843141]

_sovietthermal2y = [1.988, 3.116, 3.94, 4.938, 6.006, 7.028, 8.016, 8.987, 10.014, 11.009, 12.034, 13.025, 13.995, 15.017, 16.11, 16.921, 17.944, 19.004, 19.991, 21.011, 22.005, 23.027, 24.022, 24.988, 26.021, 27.016, 28.1, 29.041, 30.103]

_sovietthermal3x = [-0.80410033, -0.6968039, -0.6143937, -0.5257837, -0.4213608, -0.326058, -0.23284414, -0.14691047, -0.041914154, 0.03941413, 0.12613142, 0.22271647, 0.3126004, 0.39811367, 0.4801507, 0.55473137, 0.6339731, 0.7180863, 0.79000354, 0.8755243, 0.93921965, 1.0043213, 1.0757294, 1.1379868, 1.1966183, 1.2489536, 1.302569, 1.3489276, 1.3939435]

_sovietthermal3y = [1.988, 3.116, 3.94, 4.938, 6.006, 7.028, 8.016, 8.987, 10.014, 10.986, 12.018, 13.021, 13.985, 15.032, 15.966, 16.925, 17.922, 18.973, 19.95, 21.035, 22.026, 22.969, 23.999, 25.008, 26.041, 27.013, 28.074, 29.071, 30.098]

_sovietthermal4x = [-0.80410033, -0.6968039, -0.6143937, -0.5257837, -0.4213608, -0.326058, -0.23284414, -0.14691047, -0.041914154, 0.031004282, 0.11025293, 0.20221579, 0.2846563, 0.3767594, 0.44731313, 0.52009034, 0.59106463, 0.67108023, 0.7429607, 0.81117266, 0.86165386, 0.92262167, 0.9802761, 1.0408001, 1.0939817, 1.1440446, 1.1944588, 1.237594, 1.2848366]

_sovietthermal4y = [1.988, 3.116, 3.94, 4.938, 6.006, 7.028, 8.016, 8.987, 10.014, 11.034, 12.035, 13.004, 13.968, 15.007, 15.996, 16.966, 17.974, 18.975, 19.971, 21.038, 21.949, 22.947, 24.04, 25.038, 26.07, 26.996, 28.057, 29.058, 30.085]

_sovietthermal5x = [-0.80410033, -0.6968039, -0.6143937, -0.5257837, -0.4213608, -0.326058, -0.23284414, -0.14630179, -0.06752623, 0.017450733, 0.09201845, 0.17435059, 0.26316246, 0.33825722, 0.41329974, 0.48883262, 0.56026536, 0.62438524, 0.6999244, 0.75762373, 0.8071967, 0.8713394, 0.91041094, 0.9636462, 1.0141003, 1.0668474, 1.1168401, 1.1490343, 1.1913112]

_sovietthermal5y = [1.988, 3.116, 3.94, 4.938, 6.006, 7.028, 8.016, 9.011, 9.97, 11.017, 12.056, 12.989, 14.007, 14.996, 15.959, 16.963, 17.959, 18.981, 20.0, 20.994, 22.014, 23.01, 24.002, 24.993, 26.016, 27.018, 28.102, 29.082, 30.112]

_sovietthermal6x = [-0.80410033, -0.6968039, -0.64206517, -0.5512937, -0.4596705, -0.3585259, -0.26841125, -0.18243463, -0.10347379, -0.012780763, 0.065952994, 0.14144976, 0.22762965, 0.30254737, 0.37125266, 0.44043675, 0.5018805, 0.56690866, 0.6277754, 0.6809697, 0.7383048, 0.78146815, 0.8309734, 0.8705795, 0.92012334, 0.9660478, 1.0087279, 1.0514227, 1.0836459]

_sovietthermal6y = [1.988, 3.116, 3.924, 4.935, 6.006, 6.987, 7.948, 8.995, 9.959, 10.976, 12.004, 12.977, 14.006, 15.055, 15.998, 16.948, 17.94, 18.931, 19.983, 21.034, 22.002, 22.965, 24.053, 24.965, 25.989, 27.032, 28.058, 29.085, 30.111]

_sovietthermal7x = [-0.80410033, -0.6968039, -0.64206517, -0.5512937, -0.4596705, -0.3736596, -0.29157913, -0.21467015, -0.13312219, -0.05403929, 0.019946702, 0.09447114, 0.16643012, 0.23779498, 0.2955671, 0.35812527, 0.42226145, 0.4798631, 0.5340261, 0.5884958, 0.6385891, 0.6880637, 0.72762257, 0.7707784, 0.8143142, 0.84978783, 0.89292896, 0.9215304, 0.9565046]

_sovietthermal7y = [1.988, 3.116, 3.924, 4.935, 6.006, 6.973, 7.956, 8.991, 9.985, 11.016, 12.028, 13.044, 14.003, 15.003, 16.027, 16.967, 17.966, 18.971, 19.973, 21.012, 21.972, 23.011, 24.003, 25.004, 26.087, 27.015, 28.095, 29.056, 30.051]

_sovietthermal8x = [-0.80410033, -0.6968039, -0.64206517, -0.55595523, -0.4723701, -0.3936186, -0.31515464, -0.24260396, -0.16685289, -0.10182351, -0.02365001, 0.035029277, 0.08849046, 0.15289961, 0.21378331, 0.26339933, 0.31659928, 0.36436334, 0.40790054, 0.44746813, 0.48685536, 0.5291736, 0.5614592, 0.6005373, 0.63316536, 0.65810686, 0.68295693, 0.70389295, 0.71416205]

_sovietthermal8y = [1.988, 3.116, 3.924, 4.915, 5.996, 6.982, 7.972, 8.975, 9.972, 10.968, 12.025, 13.05, 13.987, 14.986, 16.034, 16.968, 17.962, 19.008, 20.005, 20.967, 21.997, 23.042, 23.974, 24.963, 25.992, 26.99, 28.037, 29.024, 30.089]

_sovietthermal9x = [-0.75696194, -0.6819367, -0.5968795, -0.52143353, -0.45842078, -0.38510278, -0.32330638, -0.2620127, -0.20134935, -0.1408617, -0.08830985, -0.04143613, 0.008174207, 0.06145246, 0.10822666, 0.15106326, 0.18977095, 0.22865695, 0.26054838, 0.303628, 0.3304138, 0.36267093, 0.3941013, 0.42308193, 0.44435713, 0.4656802, 0.4864305, 0.4931791]

_sovietthermal9y = [2.963, 3.922, 4.984, 6.037, 7.002, 7.976, 8.997, 9.992, 11.018, 12.069, 13.025, 13.992, 14.975, 15.973, 16.932, 17.989, 18.974, 19.963, 21.011, 21.997, 22.979, 23.963, 24.985, 26.01, 27.031, 28.028, 29.022, 30.067]


def _soviet_thermal1(scale_range):
    logrange = np.log10(scale_range)
    if _sovietthermal1x[0] <= logrange <= _sovietthermal1x[-1]:
        return np.interp(logrange, _sovietthermal1x, _sovietthermal1y)
    else:
        raise ValueOutsideGraphError(scale_range)

def _soviet_thermal2(scale_range):
    logrange = np.log10(scale_range)
    if _sovietthermal2x[0] <= logrange <= _sovietthermal2x[-1]:
        return np.interp(logrange, _sovietthermal2x, _sovietthermal2y)
    else:
        raise ValueOutsideGraphError(scale_range)
    
def _soviet_thermal3(scale_range):
    logrange = np.log10(scale_range)
    if _sovietthermal3x[0] <= logrange <= _sovietthermal3x[-1]:
        return np.interp(logrange, _sovietthermal3x, _sovietthermal3y)
    else:
        raise ValueOutsideGraphError(scale_range)

def _soviet_thermal4(scale_range):
    logrange = np.log10(scale_range)
    if _sovietthermal4x[0] <= logrange <= _sovietthermal4x[-1]:
        return np.interp(logrange, _sovietthermal4x, _sovietthermal4y)
    else:
        raise ValueOutsideGraphError(scale_range)

def _soviet_thermal5(scale_range):
    logrange = np.log10(scale_range)
    if _sovietthermal5x[0] <= logrange <= _sovietthermal5x[-1]:
        return np.interp(logrange, _sovietthermal5x, _sovietthermal5y)
    else:
        raise ValueOutsideGraphError(scale_range)
    
def _soviet_thermal6(scale_range):
    logrange = np.log10(scale_range)
    if _sovietthermal6x[0] <= logrange <= _sovietthermal6x[-1]:
        return np.interp(logrange, _sovietthermal6x, _sovietthermal6y)
    else:
        raise ValueOutsideGraphError(scale_range)
    
def _soviet_thermal7(scale_range):
    logrange = np.log10(scale_range)
    if _sovietthermal7x[0] <= logrange <= _sovietthermal7x[-1]:
        return np.interp(logrange, _sovietthermal7x, _sovietthermal7y)
    else:
        raise ValueOutsideGraphError(scale_range)

def _soviet_thermal8(scale_range):
    logrange = np.log10(scale_range)
    if _sovietthermal8x[0] <= logrange <= _sovietthermal8x[-1]:
        return np.interp(logrange, _sovietthermal8x, _sovietthermal8y)
    else:
        raise ValueOutsideGraphError(scale_range)

def _soviet_thermal9(scale_range):
    logrange = np.log10(scale_range)
    if _sovietthermal9x[0] <= logrange <= _sovietthermal9x[-1]:
        return np.interp(logrange, _sovietthermal9x, _sovietthermal9y)
    else:
        raise ValueOutsideGraphError(scale_range)

def _thermal_slope(impulse, actual_yield):
    """This function returns the value of the diagonal lines on the thermal impulse nomogram. Note that it uses the actual yield, not the scale yield."""
    return 12.83 - 4.93 * np.log10(impulse) + 5.15 * np.log10(actual_yield)

def _reverse_thermal_slope(y, actual_yield):
    """Reverse of _thermal_slope which returns the y-value of the non-unitized scale used in the original Soviet nomogram."""
    return 10**((y - 12.83 - 5.15 * np.log10(actual_yield)) / -4.93)

def _soviet_air_thermal(scale_range, actual_yield, visibility):
    if visibility == 1:
        return _reverse_thermal_slope(_soviet_thermal1(scale_range), actual_yield)
    elif visibility == 2:
        return _reverse_thermal_slope(_soviet_thermal2(scale_range), actual_yield)
    elif visibility == 3:
        return _reverse_thermal_slope(_soviet_thermal3(scale_range), actual_yield)
    elif visibility == 4:
        return _reverse_thermal_slope(_soviet_thermal4(scale_range), actual_yield)
    elif visibility == 5:
        return _reverse_thermal_slope(_soviet_thermal5(scale_range), actual_yield)
    elif visibility == 6:
        return _reverse_thermal_slope(_soviet_thermal6(scale_range), actual_yield)
    elif visibility == 7:
        return _reverse_thermal_slope(_soviet_thermal7(scale_range), actual_yield)
    elif visibility == 8:
        return _reverse_thermal_slope(_soviet_thermal8(scale_range), actual_yield)
    elif visibility == 9:
        return _reverse_thermal_slope(_soviet_thermal9(scale_range), actual_yield)
    else:
        raise ValueOutsideGraphError(visibility)

# These functions convert between the airburst and groundburst scales on the nomogram
def _air_to_ground(impulse):
    return 10**(np.log10(impulse) - 0.48287)

def _ground_to_air(impulse):
    return 10**(np.log10(impulse) + 0.48287)

def _soviet_ground_thermal(actual_range, actual_yield, visibility):
    return _air_to_ground(_soviet_air_thermal(actual_range, actual_yield, visibility))

def _rsoviet_thermal1(y):
    if _sovietthermal1y[0] <= y <= _sovietthermal1y[-1]:
        return 10**np.interp(y, _sovietthermal1y, _sovietthermal1x)
    else:
        raise ValueOutsideGraphError(y)

def _rsoviet_thermal2(y):
    if _sovietthermal2y[0] <= y <= _sovietthermal2y[-1]:
        return 10**np.interp(y, _sovietthermal2y, _sovietthermal2x)
    else:
        raise ValueOutsideGraphError(y)

def _rsoviet_thermal3(y):
    if _sovietthermal3y[0] <= y <= _sovietthermal3y[-1]:
        return 10**np.interp(y, _sovietthermal3y, _sovietthermal3x)
    else:
        raise ValueOutsideGraphError(y)

def _rsoviet_thermal4(y):
    if _sovietthermal4y[0] <= y <= _sovietthermal4y[-1]:
        return 10**np.interp(y, _sovietthermal4y, _sovietthermal4x)
    else:
        raise ValueOutsideGraphError(y)

def _rsoviet_thermal5(y):
    if _sovietthermal5y[0] <= y <= _sovietthermal5y[-1]:
        return 10**np.interp(y, _sovietthermal5y, _sovietthermal5x)
    else:
        raise ValueOutsideGraphError(y)

def _rsoviet_thermal6(y):
    if _sovietthermal6y[0] <= y <= _sovietthermal6y[-1]:
        return 10**np.interp(y, _sovietthermal6y, _sovietthermal6x)
    else:
        raise ValueOutsideGraphError(y)

def _rsoviet_thermal7(y):
    if _sovietthermal7y[0] <= y <= _sovietthermal7y[-1]:
        return 10**np.interp(y, _sovietthermal7y, _sovietthermal7x)
    else:
        raise ValueOutsideGraphError(y)

def _rsoviet_thermal8(y):
    if _sovietthermal8y[0] <= y <= _sovietthermal8y[-1]:
        return 10**np.interp(y, _sovietthermal8y, _sovietthermal8x)
    else:
        raise ValueOutsideGraphError(y)

def _rsoviet_thermal9(y):
    if _sovietthermal9y[0] <= y <= _sovietthermal9y[-1]:
        return 10**np.interp(y, _sovietthermal9y, _sovietthermal9x)
    else:
        raise ValueOutsideGraphError(y)

def _reverse_soviet_air_thermal(impulse, actual_yield, visibility):
    y = _thermal_slope(impulse, actual_yield)
    if visibility == 1:
        return _rsoviet_thermal1(y)
    elif visibility == 2:
        return _rsoviet_thermal2(y)
    elif visibility == 3:
        return _rsoviet_thermal3(y)
    elif visibility == 4:
        return _rsoviet_thermal4(y)
    elif visibility == 5:
        return _rsoviet_thermal5(y)
    elif visibility == 6:
        return _rsoviet_thermal6(y)
    elif visibility == 7:
        return _rsoviet_thermal7(y)
    elif visibility == 8:
        return _rsoviet_thermal8(y)
    elif visibility == 9:
        return _rsoviet_thermal9(y)
    else:
        raise ValueOutsideGraphError(visibility)

def _reverse_soviet_ground_thermal(impulse, actual_yield, visibility):
    return _reverse_soviet_air_thermal(_ground_to_air(impulse), actual_yield, visibility)

# Confusingly, the numbers [1-9] found in the original nomogram and used in the
# functions above are actually the _opposite_ of the International Visibility Code
# -----------------+------------------+---------------------+-----------------------
# Description      | International    |Model Input (< 100kT | Model Input (airbursts
#                  | Visibility Code  |and groundbursts)    | >100kT)
# -----------------+------------------+---------------------+-----------------------
# "Clear air"      | 9                | 1                   | 1
# "Very light haze"| 7                | 3                   | 2
# "Light haze"     | 5                | 5                   | 4
# "Smoggy air"     | 4                | 6                   | 5
# "Light fog"      | 3                | 8                   | 7
# "Thick fog"      | 1-2              | 9                   | 8

def _ivc_to_model_input(ivc, actual_yield, burst_height):
    """This function converts an International Visibility Code [1-9] into the numbers used in the Soviet thermal impulse model."""
    if ivc == 9:
        return 1
    else:
        groundburst = 10 - ivc
        if ivc == 3 or ivc == 2:
            groundburst = groundburst + 1
        if burst_height == 0 or actual_yield <= 100:
            return groundburst
        else:
            return groundburst - 1

# Didn't include unit conversions for fluence--does anyone use something other than
# Cal/cm^2?

def soviet_air_thermal(y, r, h, visibility, yunits='kT', dunits='m'):
    """Estimate thermal fluence at range r from an air burst of yield y using the 
methodology described in _Iadernoe oruzhie_ 4th ed. (1987).

visibility is given as International Visibility Code (IVC) numbers (1=thick fog,
9=clear)."""
    slant_range = convert_units(r, dunits, 'km')
    yld = convert_units(y, yunits, 'kT')
    # height only used to determine visibility code
    v = _ivc_to_model_input(visibility, yld, h)
    fluence = _soviet_air_thermal(slant_range, yld, v)
    return fluence

def soviet_ground_thermal(y, r, h, visibility, yunits='kT', dunits='m'):
    """Estimate thermal fluence at range r from a ground burst of yield y using the 
methodology described in _Iadernoe oruzhie_ 4th ed. (1987).

visibility is given as International Visibility Code (IVC) numbers (1=thick fog,
9=clear)."""
    slant_range = convert_units(r, dunits, 'km')
    yld = convert_units(y, yunits, 'kT')
    # height only used to determine visibility code
    v = _ivc_to_model_input(visibility, yld, h)
    fluence = _soviet_ground_thermal(slant_range, yld, v)
    return fluence

def r_soviet_air_thermal(y, fluence, h, visibility, yunits='kT', dunits='m'):
    """Estimate the range from an air burst of yield y at which fluence will occur
using the methodology described in _Iadernoe oruzhie_ 4th ed. (1987).

visibility is given as International Visibility Code (IVC) numbers (1=thick fog,
9=clear)."""
    yld = convert_units(y, yunits, 'kT')
    v = _ivc_to_model_input(visibility, yld, h)
    sr = _reverse_soviet_air_thermal(fluence, yld, v)
    return convert_units(sr, 'km', dunits)

def r_soviet_ground_thermal(y, fluence, h, visibility, yunits='kT', dunits='m'):
    """Estimate the range from a ground burst of yield y at which fluence will occur
using the methodology described in _Iadernoe oruzhie_ 4th ed. (1987).

visibility is given as International Visibility Code (IVC) numbers (1=thick fog,
9=clear)."""
    yld = convert_units(y, yunits, 'kT')
    v = _ivc_to_model_input(visibility, yld, h)
    sr = _reverse_soviet_ground_thermal(fluence, yld, v)
    return convert_units(sr, 'km', dunits)
