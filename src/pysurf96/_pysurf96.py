from typing import Literal

import os
import glob
import platform
import numpy
import numpy.ctypeslib
import ctypes

MAXLAYER = 100
MAXPERIODS = 60


WaveType = Literal["love", "rayleigh"]
Velocity = Literal["group", "phase"]

libsurf96=ctypes.cdll.LoadLibrary(glob.glob(os.path.dirname(__file__)+"/surf96*.so")[0])

class Surf96Error(Exception):
    pass


def surf96(
    thickness: numpy.ndarray,
    vp: numpy.ndarray,
    vs: numpy.ndarray,
    rho: numpy.ndarray,
    periods: numpy.ndarray,
    wave: WaveType = "love",
    mode: int = 1,
    velocity: Velocity = "group",
    flat_earth: bool = True,
) -> numpy.ndarray:
    """Calculate synthetic surface wave dispersion curves.

    Calculate synthetic surface wave dispersion curves for a given earth model, wave
    type and periods.

    This is a slim Fortran wrapper around surf96 from Computer Programs in Seismology
    from R. Hermann (2013)

    Args:
        thickness (numpy.ndarray): Layer thickness in [km].
        vp (numpy.ndarray): Layer Vp velocity in [km/s].
        vs (numpy.ndarray): Layer Vs velocity in [km/s].
        rho (numpy.ndarray): Layer density in [g/m^3].
        periods (numpy.ndarray): The periods in seconds, where wave velocity is calculated
        wave (WaveType, optional): The wave type, "love" or "rayleigh".
            Defaults to "love".
        mode (int, optional): Mode of the wave, 1: fundamental, 2: second-mode, etc...
            Minimum is fundamental mode (1). Defaults to 1.
        velocity (Velocity, optional): "group" or "phase" velocity. Defaults to "group".
        flat_earth (bool, optional): Assume a flat earth. Defaults to True.

    Raises:
        ValueError: Raised when inumpyut values are unexpected.
        Surf96Error: If surf96 fortran code raises an error,
            this may be due to low velocity zone.

    Returns:
        numpy.ndarray: The surface wave velocities at defined periods.
    """
    if not (thickness.size == vp.size == vs.size == rho.size):
        raise ValueError("Thickness, vp/vs velocities and rho have different sizes.")
    if not (thickness.ndim == vp.ndim == vs.ndim == rho.ndim == 1):
        "Thickness, vp/vs velocities or rho have more than one dimension"
    if thickness.size > MAXLAYER:
        raise ValueError(f"Maximum number of layers is {MAXLAYER}")
    if periods.size > MAXPERIODS:
        raise ValueError(f"Maximum number of periods is {MAXPERIODS}")
    if wave not in ("love", "rayleigh"):
        raise ValueError("Wave type has to be either love or rayleigh")
    if velocity not in ("group", "phase"):
        raise ValueError("Velocity has to be group or phase")
    if mode <= 0:
        raise ValueError("Mode has to be at least 1 (fundamental mode)")

    nlayers = thickness.size
    kmax = periods.size

    _thk = numpy.empty(MAXLAYER)
    _vp = numpy.empty(MAXLAYER)
    _vs = numpy.empty(MAXLAYER)
    _rho = numpy.empty(MAXLAYER)

    _thk[:nlayers] = thickness
    _vp[:nlayers] = vp
    _vs[:nlayers] = vs
    _rho[:nlayers] = rho

    iflsph = 1 if flat_earth else 0
    iwave = 1 if wave == "love" else 2
    igr = 0 if velocity == "phase" else 1
    mode = int(mode)

    t = numpy.empty(MAXPERIODS)
    t[:kmax] = periods
    
    result = numpy.zeros(MAXPERIODS)

	# the conversion vp -> _vp -> vp_ could possibly be done in one step

    thk_=numpy.asfortranarray(_thk,dtype=numpy.float32)
    vp_=numpy.asfortranarray(_vp,dtype=numpy.float32)
    vs_=numpy.asfortranarray(_vs,dtype=numpy.float32)

    rho_=numpy.asfortranarray(_rho,dtype=numpy.float32)
    nlayers_=ctypes.c_int(nlayers)
    iflsph_=ctypes.c_int(iflsph)
    iwave_=ctypes.c_int(iwave)
    mode_=ctypes.c_int(mode)
    igr_=ctypes.c_int(igr)
    kmax_=ctypes.c_int(kmax)
    t_=numpy.asfortranarray(t,dtype=numpy.float32)
    result_=numpy.asfortranarray(result,dtype=numpy.float32)
    error=ctypes.c_int(0)

    libsurf96.surfdisp96(
        thk_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        vp_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        vs_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
        rho_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
        ctypes.byref(nlayers_), 
        ctypes.byref(iflsph_), 
        ctypes.byref(iwave_), 
        ctypes.byref(mode_), 
        ctypes.byref(igr_), 
    	ctypes.byref(kmax_), 
        t_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        result_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(error)
    )
    
    result=result_    

    if error:
        raise Surf96Error(
            "surf96 threw an error! "
            "This may be due to low velocity zone causing"
            " reverse phase velocity dispersion,"
            " and mode jumping. Due to looking for Love waves in a halfspace"
            " which is OK if there are Rayleigh data."
        )

    return result[:kmax]


__all__ = ["surf96", "Surf96Error"]
