import numpy as np

# functions reused throughout glasstone module
class ValueOutsideGraphError(Exception):
    """This exception indicates that the requested input value falls outside the
graphs found in the original source."""
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# Unfortunately, historical NWE models emply a bewildering array of non-Si units.
# Even worse, many of the models utilize arbitrary combinations of units. This unit
# conversion function is provided to provide standard inputs in SI units for all
# parts of glasstone.

class UnknownUnitError(Exception):
    """This exception indicates that the requested unit to convert is unknown."""
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
    
def convert_units(v, unitsfrom, unitsto):
    if unitsfrom == unitsto:
        return v
    # yield
    elif unitsfrom == 'kT' and unitsto== 'MT':
        return v / 1000.0
    elif unitsfrom == 'MT' and unitsto== 'kT':
        return v * 1000.0
    # distance
    elif unitsfrom == 'm' and unitsto== 'kilofeet':
        return v / 304.8
    elif unitsfrom == 'm' and unitsto== 'km':
        return v / 1000.0
    elif unitsfrom == 'km' and unitsto== 'm':
        return v * 1000.0
    elif unitsfrom == 'kilofeet' and unitsto== 'm':
        return 304.8 * v
    elif unitsfrom == 'yards' and unitsto== 'm':
        return v / 1.09361
    elif unitsfrom == 'm' and unitsto== 'yards':
        return v * 1.09361
    elif unitsfrom == 'ft' and unitsto== 'm':
        return v * 0.3048
    elif unitsfrom == 'm' and unitsto== 'ft':
        return v / 0.3048
    elif unitsfrom == 'kilofeet' and unitsto== 'km':
        return convert_units(v, 'kilofeet', 'm') / 1000.0
    elif unitsfrom == 'kilofeet' and unitsto== 'mi':
        return v / 5.28
    elif unitsfrom== 'mi' and unitsto== 'km':
        return v * 1.60934
    elif unitsfrom== 'km' and unitsto== 'mi':
        return v / 1.60934
    elif unitsfrom== 'km' and unitsto== 'kilofeet':
        return v / 0.3048
    elif unitsfrom== 'yards' and unitsto== 'meters':
        return v * 0.9144
    elif unitsfrom== 'yards' and unitsto== 'km':
        return v * 0.0009144
    elif unitsfrom== 'meters' and unitsto== 'yards':
        return v / 0.9144
    elif unitsfrom== 'km' and unitsto== 'yards':
        return v / 0.0009144
    #pressure
    elif unitsfrom == 'psi' and unitsto== 'kg/cm^2':
        return v * 0.070307
    elif unitsfrom == 'kg/cm^2' and unitsto== 'psi':
        return v / 0.070307
    elif unitsfrom == 'MPa' and unitsto== 'psi':
        return v * 145.037738
    elif unitsfrom == 'psi' and unitsto== 'MPa':
        return v / 145.037738
    elif unitsfrom == 'kg/cm^2' and unitsto== 'MPa':
        return convert_units(convert_units(v, 'kg/cm^2', 'psi'), 'psi', 'MPa')
    elif unitsfrom == 'MPa' and unitsto== 'kg/cm^2':
        return convert_units(convert_units(v, 'psi', 'kg/cm^2'), 'MPa', 'psi')
    elif unitsfrom =='Pa':
        return convert_units(v, 'MPa', unitsto) / 1e6
    elif unitsto == 'Pa':
        return convert_units(v, unitsfrom, 'MPa') * 1e6
    # speed
    elif unitsfrom == 'm/s' and unitsto== 'mph':
        return v * 2.23694
    elif unitsfrom == 'mph' and unitsto== 'm/s':
        return v / 2.23694
    elif unitsfrom == 'm/s' and unitsto== 'km/h':
        return v * 3.6
    elif unitsfrom == 'km/h' and unitsto== 'm/s':
        return v / 3.6
    elif unitsfrom == 'mph' and unitsto== 'km/h':
        return v * 1.60934
    elif unitsfrom == 'km/h' and unitsto== 'mph':
        return v / 1.60934
    # wind shear
    elif unitsfrom == 'm/s-km' and unitsto == 'mph/kilofoot':
        return v * 0.13625756613945836
    # dose
    # under normal circumstances this isn't quite right, as Roentgens were
    # usually employed as a unit of exposure rather than dose. However, WSEG-10
    # used an unusual unit, the Equivilent Residual Dose, which does convert
    # directly into Sv:
    elif unitsfrom == 'Roentgen' and unitsto == 'Sv':
        return v / 100.0
    else:
        raise UnknownUnitError((unitsfrom, unitsto))

def dict_reverse(d):
    new_dict = {}
    for k in d:
        new_dict[k] = d[k][::-1]
    return new_dict
