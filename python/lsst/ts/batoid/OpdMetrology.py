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
import warnings
import numpy as np
from astropy.io import fits

import lsst.geom
from lsst.afw.cameraGeom import PIXELS, FIELD_ANGLE

from lsst.ts.wep.cwfs.Tool import ZernikeAnnularFit, ZernikeEval
from lsst.ts.wep.ParamReader import ParamReader

from lsst.ts.ofc.utils import get_config_dir as getConfigDirOfc

from lsst.ts.phosim.utils.MetroTool import calc_pssn, psf2eAtmW
from lsst.ts.phosim.utils.Utility import getConfigDir, getCamera


class OpdMetrology(object):
    def __init__(self):
        """Initialization of OPD metrology class.

        OPD: Optical path difference.
        """

        self.wt = np.array([])
        self.fieldX = np.array([])
        self.fieldY = np.array([])
        self._camera = None

    def setCamera(self, instName):
        """Set the camera.

        Parameters
        ----------
        instName : `str`
            Instrument name. Valid options are 'comcam or 'lsstfam'.
        """

        self._camera = getCamera(instName)

    def getFieldXY(self):
        """Get the field X, Y in degree.

        Returns
        -------
        numpy.ndarray
            Field X in degree.
        numpy.ndarray
            Field Y in degree.
        """

        return self.fieldX, self.fieldY

    def getWeightingRatio(self):
        """Get the weighting ratio used in Gaussian quadrature.

        Returns
        -------
        numpy.ndarray
            Weighting ratio.
        """

        return self.wt

    def setWeightingRatio(self, wt):
        """Set the weighting ratio used in Gaussian quadrature.

        Parameters
        ----------
        wt : list or numpy.ndarray
            Weighting ratio.
        """

        wtArray = np.array(wt, dtype=float)
        self._setNormalizedWeightingRatio(wtArray)

    def _setNormalizedWeightingRatio(self, wtArray):
        """Normalize the weighting ratios.

        Parameters
        ----------
        wtArray : numpy.ndarray
            Weighting ratio.

        Raises
        ------
        ValueError
            All weighting ratios should be >=0.
        """

        if np.all(wtArray >= 0):
            self.wt = wtArray / np.sum(wtArray)
        else:
            raise ValueError("All weighting ratios should be >=0.")

    def setFieldXYinDeg(self, fieldXInDegree, fieldYInDegree):
        """Set the field X, Y in degree.

        Parameters
        ----------
        fieldXInDegree : list or numpy.ndarray
            Field X in degree.
        fieldYInDegree : list or numpy.ndarray
            Field Y in degree.
        """

        self.fieldX = np.array(fieldXInDegree, dtype=float)
        self.fieldY = np.array(fieldYInDegree, dtype=float)

    def addFieldXYbyDeg(self, fieldXInDegree, fieldYInDegree):
        """Add the new field X, Y in degree.

        Parameters
        ----------
        fieldXInDegree : float, list, or numpy.ndarray
            New field X in degree.
        fieldYInDegree : float, list, or numpy.ndarray
            New field Y in degree.
        """

        self.fieldX = np.append(self.fieldX, fieldXInDegree)
        self.fieldY = np.append(self.fieldY, fieldYInDegree)

    def setWgtAndFieldXyOfGQ(self, instName):
        """Set the GQ weighting ratio and field X, Y.

        GQ: Gaussian quadrature.

        Parameters
        ----------
        instName : `str`
            Instrument name.

        Raises
        ------
        RuntimeError
            If the instrument path does not exists.
            If fieldXy.yaml file does not exists in the instrument
            configuration directory.
        """

        instrumentPath = getConfigDirOfc() / instName

        if not instrumentPath.exists():
            raise RuntimeError(f"OFC instrument path does not exist: {instrumentPath}")

        # Set the weighting ratio
        pathWgtFile = instrumentPath / "imgQualWgt.yaml"
        wgtFile = ParamReader(filePath=pathWgtFile)
        self.setWeightingRatio(list(wgtFile.getContent().values()))

        # Set the field (x, y)
        pathFieldXyFile = os.path.join(
            getConfigDir(), "instrument", instName, "fieldXy.yaml"
        )

        if not os.path.exists(pathFieldXyFile):
            raise RuntimeError(f"Field xy file does not exists: {pathFieldXyFile}.")

        paramReader = ParamReader(filePath=pathFieldXyFile)
        fieldXY = paramReader.getMatContent()
        self.setFieldXYinDeg(fieldXY[:, 0], fieldXY[:, 1])

    def setDefaultLsstGQ(self):
        """Set the default LSST GQ field X, Y and weighting ratio.

        GQ: Gaussian quadrature
        """

        warnings.warn(
            "Use setWgtAndFieldXyOfGQ() instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.setWgtAndFieldXyOfGQ("lsst")

    def setDefaultLsstWfsGQ(self):
        """Set default values for LSST WFS field X, Y
        and weighting ratio.
        """

        # Set equal full weights for each of the
        # four corner wavefront sensor pairs.
        self.setWeightingRatio([1.0, 1.0, 1.0, 1.0])
        wfsFieldX, wfsFieldY = self.getDefaultLsstWfsGQ()
        self.setFieldXYinDeg(wfsFieldX, wfsFieldY)

    def getDefaultLsstWfsGQ(self):
        """Get the default field X, Y of LSST WFS on GQ.

        WFS: Wavefront sensor.
        GQ: Gaussian quadrature

        Returns
        -------
        list
            Field X in degree.
        list
            Field Y in degree.
        """

        # Field x, y for 4 WFS
        fieldWFSx = [1.176, -1.176, -1.176, 1.176]
        fieldWFSy = [1.176, 1.176, -1.176, -1.176]

        return fieldWFSx, fieldWFSy

    def setDefaultComcamGQ(self):
        """Set the default ComCam GQ field X, Y and weighting ratio.

        GQ: Gaussian quadrature
        """

        warnings.warn(
            "Use setWgtAndFieldXyOfGQ() instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.setWgtAndFieldXyOfGQ("comcam")

    def getZkFromOpd(self, opdFitsFile=None, opdMap=None, znTerms=22, obscuration=0.61):
        """Get the wavefront error of OPD in the basis of annular Zernike
        polynomials.

        OPD: Optical path difference.

        Parameters
        ----------
        opdFitsFile : str, optional
            OPD FITS file. (the default is None.)
        opdMap : numpy.ndarray, optional
            OPD map data. (the default is None.)
        znTerms : int, optional
            Number of terms of annular Zk (z1-z22 by default). (the default
            is 22.)
        obscuration : float, optional
            Obscuration of annular Zernike polynomial. (the default is 0.61.)

        Returns
        -------
        numpy.ndarray
            Annular Zernike polynomials. For PhoSim OPD, the unit is um.
        numpy.ndarray
            OPD map.
        numpy.ndarray
            Meshgrid x in OPD map.
        numpy.ndarray
            Meshgrid y in OPD map.

        Raises
        ------
        ValueError
            The x, y dimensions of OPD are different.
        """

        # Get the OPD data (PhoSim OPD unit: um)
        if opdFitsFile is not None:
            opd = fits.getdata(opdFitsFile)
        elif opdMap is not None:
            opd = opdMap.copy()

        # Check the x, y dimensions of OPD are the same
        if np.unique(opd.shape).size != 1:
            raise ValueError("The x, y dimensions of OPD are different.")

        # x-, y-coordinate in the OPD image
        opdSize = opd.shape[0]
        opdGrid1d = np.linspace(-1, 1, opdSize)
        opdx, opdy = np.meshgrid(opdGrid1d, opdGrid1d)

        # Fit the OPD map with Zk and write into the file
        idx = opd != 0
        zk = ZernikeAnnularFit(opd[idx], opdx[idx], opdy[idx], znTerms, obscuration)

        return zk, opd, opdx, opdy

    def rmPTTfromOPD(self, opdFitsFile=None, opdMap=None):
        """Remove the afftection of piston (z1), x-tilt (z2), and y-tilt (z3)
        from the OPD map.

        OPD: Optical path difference.

        Parameters
        ----------
        opdFitsFile : str, optional
            OPD FITS file. (the default is None.)
        opdMap : numpy.ndarray, optional
            OPD map data. (the default is None.)

        Returns
        -------
        numpy.ndarray
            OPD map after removing the affection of z1-z3.
        numpy.ndarray
            Meshgrid x in OPD map.
        numpy.ndarray
            Meshgrid y in OPD map.
        """

        # Do the spherical Zernike fitting for the OPD map
        # Only fit the first three terms (z1-z3): piston, x-tilt, y-tilt
        zk, opd, opdx, opdy = self.getZkFromOpd(
            opdFitsFile=opdFitsFile, opdMap=opdMap, znTerms=3, obscuration=0
        )

        # Find the index that the value of OPD is not 0
        idx = opd != 0

        # Remove the PTT
        opd[idx] -= ZernikeEval(zk, opdx[idx], opdy[idx])

        return opd, opdx, opdy

    def addFieldXYbyCamPos(self, sensorName, xInpixel, yInPixel, folderPath2FocalPlane):
        """Add the new field X, Y in degree by the camera pixel positions.

        Parameters
        ----------
        sensorName : str
            Canera sensor name (e.g. "R22_S11", "R40_S02_C0").
        xInpixel : float
            Pixel x on camera coordinate.
        yInPixel : float
            Pixel y on camera coordinate.
        folderPath2FocalPlane : str
            Path to the directory of focal plane data ("focalplanelayout.txt").
        """

        # Get the sensor
        sensor = self._camera[sensorName]

        # Do the coordinate transformation. DM Coordinates
        # expected from the center of the pixel (0.5 pixel offset)
        fieldYInRad, fieldXInRad = sensor.transform(
            lsst.geom.Point2D(yInPixel - 0.5, xInpixel - 0.5), PIXELS, FIELD_ANGLE
        )
        fieldXInDegree = np.degrees(fieldXInRad)
        fieldYInDegree = np.degrees(fieldYInRad)

        # Add to listed field x, y
        self.addFieldXYbyDeg(fieldXInDegree, fieldYInDegree)

    def calcPSSN(
        self, wavelengthInUm, opdFitsFile=None, opdMap=None, zen=0, debugLevel=0
    ):
        """Calculate the PSSN based on OPD map.

        PSSN: Normalized point source sensitivity.
        OPD: Optical path difference.

        Parameters
        ----------
        wavelengthInUm : float
            Wavelength in microns.
        opdFitsFile : str, optional
            OPD FITS file. (the default is None.)
        opdMap : numpy.ndarray, optional
            OPD map data. (the default is None.)
        zen : float, optional
            elescope zenith angle in degree. (the default is 0.)
        debugLevel : int, optional
            Debug level. The higher value gives more information. (the default
            is 0.)

        Returns
        -------
        float
            Calculated PSSN.
        """

        # Before calc_pssn,
        # (1) Remove PTT (piston, x-tilt, y-tilt),
        # (2) Make sure outside of pupil are all zeros
        opdRmPTT = self.rmPTTfromOPD(opdFitsFile=opdFitsFile, opdMap=opdMap)[0]

        # Calculate the normalized point source sensitivity (PSSN)
        pssn = calc_pssn(opdRmPTT, wavelengthInUm, zen=zen, debugLevel=debugLevel)

        return pssn

    def calcFWHMeff(self, pssn):
        """Calculate the effective FWHM.

        FWHM: Full width at half maximum.
        PSSN: Normalized point source sensitivity.

        Parameters
        ----------
        pssn : float
            PSSN value.

        Returns
        -------
        float
            Effective FWHM.
        """

        # FWHMeff_sys = FWHMeff_atm * sqrt(1/PSSN - 1).
        # FWHMeff_atm = 0.6 arcsec.
        # Another correction factor (eta = 1.086) is used to account for the
        # difference between the
        # simple RSS and the more proper convolution.
        # Follow page 7 (section 7.2 FWHM) in document-17242 for more
        # information.
        eta = 1.086
        FWHMatm = 0.6
        FWHMeff = eta * FWHMatm * np.sqrt(1 / pssn - 1)

        return FWHMeff

    def calcDm5(self, pssn):
        """Calculate the loss of limiting depth (dm5).

        PSSN: Normalized point source sensitivity.

        Parameters
        ----------
        pssn : float
            PSSN value.

        Returns
        -------
        float
            dm5.
        """

        # Calculate dm5 (the loss of limiting depth)
        # Check eq. (4.1) in page 4 in document-17242 for more information.
        dm5 = -1.25 * np.log10(pssn)

        return dm5

    def calcEllip(
        self, wavelengthInUm, opdFitsFile=None, opdMap=None, zen=0, debugLevel=0
    ):
        """Calculate the ellipticity.

        Parameters
        ----------
        wavelengthInUm : float
            Wavelength in microns.
        opdFitsFile : str, optional
            OPD FITS file. (the default is None.)
        opdMap : numpy.ndarray, optional
            OPD map data. (the default is None.)
        zen : float, optional
            elescope zenith angle in degree. (the default is 0.)
        debugLevel : int, optional
            Debug level. The higher value gives more information. (the default
            is 0.)

        Returns
        -------
        float
            Ellipticity.
        """

        # Remove the affection of piston (z1), x-tilt (z2), and y-tilt (z3)
        # from OPD map.
        opdRmPTT = self.rmPTTfromOPD(opdFitsFile=opdFitsFile, opdMap=opdMap)[0]

        # Calculate the ellipticity
        elli = psf2eAtmW(opdRmPTT, wavelengthInUm, zen=zen, debugLevel=debugLevel)[0]

        return elli

    def calcGQvalue(self, valueList):
        """Calculate the GQ value.

        GQ: Gaussian quadrature

        Parameters
        ----------
        valueList : list or numpy.ndarray
            List of value (PSSN, effective FWHM, dm5, ellipticity).

        Returns
        -------
        float
            GQ value.

        Raises
        ------
        ValueError
            Length of wt ratio != length of value list.
        """

        # Check the lengths of weighting ratio and value list are the same
        if len(self.wt) != len(valueList):
            raise ValueError("Length of wt ratio != length of value list.")

        # Calculate the effective value on Gaussain quardure plane
        valueArray = np.array(valueList, dtype=float)
        GQvalue = np.sum(self.wt * valueArray)

        return GQvalue
