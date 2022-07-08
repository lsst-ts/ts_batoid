# This file is part of ts_phosim.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import re
import warnings
from enum import Enum

from lsst.utils import getPackageDir

from lsst.obs.lsst import LsstComCam, LsstCam


class CamDistType(Enum):
    L1S1zer = 1
    L1S2zer = 2
    L2S1zer = 3
    L2S2zer = 4
    L3S1zer = 5
    L3S2zer = 6


class SurfaceType(Enum):
    M1 = 1
    M2 = 2
    M3 = 3
    L1F = 4
    L1B = 5
    L2F = 6
    L2B = 7
    FilterF = 8
    FilterB = 9
    L3F = 10
    L3B = 11
    FP = 12
    Chip = 13


def getModulePath():
    """Get the path of module.

    Returns
    -------
    str
        Directory path of module.
    """

    return getPackageDir("ts_phosim")


def getConfigDir():
    """Get the directory of configuration files.

    Returns
    -------
    str
        Directory of configuration files.
    """

    return os.path.join(getModulePath(), "policy")


def opt2ZemaxCoorTrans(x, y, z):
    """Do the coordinate transformation from optics to Zemax.

    Parameters
    ----------
    x : float
        x coordinate in optics.
    y : float
        y coordinate in optics.
    z : float
        z coordinate in optics.

    Returns
    -------
    float
        x coordinate in Zemax.
    float
        y coordinate in Zemax.
    float
        z coordinate in Zemax.
    """

    return -x, y, -z


def zemax2optCoorTrans(x, y, z):
    """Do the coordinate transformation from Zemax to optics.

    Parameters
    ----------
    x : float
        x coordinate in Zemax.
    y : float
        y coordinate in Zemax.
    z : float
        z coordinate in Zemax.

    Returns
    -------
    float
        x coordinate in optics.
    float
        y coordinate in optics.
    float
        z coordinate in optics.
    """

    return -x, y, -z


def mapSurfNameToEnum(surfName):
    """Map the surface name to Enum: SurfaceType.

    Parameters
    ----------
    surfName : str
        Surface name.

    Returns
    -------
    enum 'SurfaceType'
        SurfaceType.

    Raises
    ------
    ValueError
        No surface type is found.
    """

    nameList = [surfaceType.name for surfaceType in SurfaceType]
    try:
        idx = nameList.index(surfName.strip())
        return SurfaceType(idx + 1)
    except ValueError:
        raise ValueError("No surface type is found.")


def getPhoSimPath(phosimPathVar="PHOSIMPATH"):
    """Ge the PhoSim path from the environment variables.

    Parameters
    ----------
    phosimPathVar : str, optional
        PhoSim path variable name. (the default is "PHOSIMPATH".)

    Returns
    -------
    str
        PhoSim path.

    Raises
    ------
    RuntimeError
        Please set the 'PHOSIMPATH' environment variable.
    """

    try:
        return os.environ[phosimPathVar]
    except KeyError:
        raise RuntimeError("Please set the '%s' environment variable." % phosimPathVar)


def getAoclcOutputPath(aoclcOutputPathVar="AOCLCOUTPUTPATH"):
    """Get the AOCLC output path.

    AOCLC: Active optics closed loop control

    Parameters
    ----------
    aoclcOutputPathVar : str, optional
        AOCLC output path variable. (the default is "AOCLCOUTPUTPATH".)

    Returns
    -------
    str
        AOCLC output path.
    """

    try:
        outputPath = os.environ[aoclcOutputPathVar]
    except KeyError:
        outputPath = os.path.join(getModulePath(), "output")
        warnings.warn(
            "No 'AOCLCOUTPUTPATH' assigned. Use %s instead." % outputPath,
            category=UserWarning,
        )

    return outputPath


def sortOpdFileList(opdFileList):
    """Sort the OPD file list.

    OPD: Optical path difference.

    Parameters
    ----------
    opdFileList : list[str]
        OPD file list.

    Returns
    -------
    list[str]
        Sorted OPD file list.

    Raises
    ------
    ValueError
        Unmatched file name found.
    """

    # Get the sorted index based on the OPD number
    baseNameList = [os.path.basename(filePath) for filePath in opdFileList]
    opdNumList = []
    for basename in baseNameList:
        m = re.match(r"\Aopd_\d+_(\d+).fits", basename)
        if m is None:
            raise ValueError("Unmatched file name (%s) found." % basename)
        else:
            opdNumList.append(int(m.groups()[0]))

    sortedIdxList = sorted(range(len(opdNumList)), key=lambda idx: opdNumList[idx])

    # Get the rearranged file list
    sortedOpdFileList = [opdFileList[idx] for idx in sortedIdxList]

    return sortedOpdFileList


def getCamera(instName):
    """Returns a lsst instrument for a given instrument name.

    Parameters
    ----------
    instName : `str`
        Instrument name. Valid options are 'comcam or 'lsstfam'.

    Returns
    -------
    camera : `lsst.afw.cameraGeom.Camera`

    Raises
    ------
    ValueError
        If input `instName` is not valid.
    """
    # Check the input
    if (instName == "lsstfam") or (instName == "lsst"):
        return LsstCam().getCamera()
    elif instName == "comcam":
        return LsstComCam().getCamera()
    else:
        raise ValueError(
            f"This instrument name ({instName}) is not supported. Must be 'comcam' or 'lsstfam' or 'lsst'."
        )


if __name__ == "__main__":
    pass
