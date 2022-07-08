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
import numpy as np

from lsst.ts.phosim.telescope.MirrorSim import MirrorSim

from lsst.ts.phosim.utils.Utility import getConfigDir, opt2ZemaxCoorTrans
from lsst.ts.phosim.utils.PlotUtil import plotResMap

from lsst.ts.wep.cwfs.Tool import ZernikeAnnularFit, ZernikeAnnularEval
from lsst.ts.wep.ParamReader import ParamReader


class M1M3Sim(MirrorSim):
    def __init__(self):
        """Initiate the M1M3 simulator class."""

        # M2 setting file
        configDir = os.path.join(getConfigDir(), "M1M3")
        settingFilePath = os.path.join(configDir, "m1m3Setting.yaml")
        self._m1m3SettingFile = ParamReader(filePath=settingFilePath)

        # Inner and outer radius of M1 mirror in m
        radiusM1Inner = self._m1m3SettingFile.getSetting("radiusM1Inner")
        radiusM1Outer = self._m1m3SettingFile.getSetting("radiusM1Outer")

        # Inner and outer radius of M3 mirror in m
        radiusM3Inner = self._m1m3SettingFile.getSetting("radiusM3Inner")
        radiusM3Outer = self._m1m3SettingFile.getSetting("radiusM3Outer")

        super(M1M3Sim, self).__init__(
            (radiusM1Inner, radiusM3Inner), (radiusM1Outer, radiusM3Outer), configDir
        )

        # Mirror surface bending mode grid file
        self._gridFile = ParamReader()

        # FEA model file
        self._feaFile = ParamReader()

        # FEA model data in zenith angle
        self._feaZenFile = ParamReader()

        # FEA model data in horizontal angle
        self._feaHorFile = ParamReader()

        # Actuator forces along zenith direction
        self._forceZenFile = ParamReader()

        # Actuator forces along horizon direction
        self._forceHorFile = ParamReader()

        # Influence matrix of actuator forces
        self._forceInflFile = ParamReader()

        self._config(
            "M1M3_LUT.yaml",
            "M1M3_1um_156_grid.yaml",
            "M1M3_thermal_FEA.yaml",
            "M1M3_dxdydz_zenith.yaml",
            "M1M3_dxdydz_horizon.yaml",
            "M1M3_force_zenith.yaml",
            "M1M3_force_horizon.yaml",
            "M1M3_influence_256.yaml",
        )

    def _config(
        self,
        lutFileName,
        gridFileName,
        feaFileName,
        feaZenFileName,
        feaHorFileName,
        forceZenFileName,
        forceHorFileName,
        forceInflFileName,
    ):
        """Do the configuration.

        LUT: Look-up table.
        FEA: Finite element analysis.

        Parameters
        ----------
        lutFileName : str
            LUT file name.
        gridFileName : str
            File name of bending mode data.
        feaFileName : str
            FEA model data file name.
        feaZenFileName : str
            FEA model data file name in zenith angle.
        feaHorFileName : str
            FEA model data file name in horizontal angle.
        forceZenFileName : str
            File name of actuator forces along zenith direction.
        forceHorFileName : str
            File name of actuator forces along horizon direction.
        forceInflFileName : str
            Influence matrix of actuator forces.
        """

        numTerms = self._m1m3SettingFile.getSetting("numTerms")

        super(M1M3Sim, self).config(
            numTerms=numTerms,
            lutFileName=lutFileName,
        )

        mirrorDataDir = self.getMirrorDataDir()

        gridFilePath = os.path.join(mirrorDataDir, gridFileName)
        self._gridFile.setFilePath(gridFilePath)

        feaFilePath = os.path.join(mirrorDataDir, feaFileName)
        self._feaFile.setFilePath(feaFilePath)

        feaZenFilePath = os.path.join(mirrorDataDir, feaZenFileName)
        self._feaZenFile.setFilePath(feaZenFilePath)

        feaHorFilePath = os.path.join(mirrorDataDir, feaHorFileName)
        self._feaHorFile.setFilePath(feaHorFilePath)

        forceZenFilePath = os.path.join(mirrorDataDir, forceZenFileName)
        self._forceZenFile.setFilePath(forceZenFilePath)

        forceHorFilePath = os.path.join(mirrorDataDir, forceHorFileName)
        self._forceHorFile.setFilePath(forceHorFilePath)

        forceInflFilePath = os.path.join(mirrorDataDir, forceInflFileName)
        self._forceInflFile.setFilePath(forceInflFilePath)

    def getPrintthz(self, zAngleInRadian, preCompElevInRadian=0):
        """Get the mirror print in m along z direction in specific zenith
        angle.

        FEA: Finite element analysis.

        Parameters
        ----------
        zAngleInRadian : float
            Zenith angle in radian.
        preCompElevInRadian : float, optional
            Pre-compensation elevation angle in radian. (the default is 0.)

        Returns
        -------
        numpy.ndarray
            Corrected projection in m along z direction.
        """

        # Data needed to determine gravitational print through
        data = self._feaZenFile.getMatContent()
        zdx = data[:, 0]
        zdy = data[:, 1]
        zdz = data[:, 2]

        data = self._feaHorFile.getMatContent()
        hdx = data[:, 0]
        hdy = data[:, 1]
        hdz = data[:, 2]

        # Do the M1M3 gravitational correction.
        # Map the changes of dx, dy, and dz on a plane for certain zenith angle
        printthxInM = zdx * np.cos(zAngleInRadian) + hdx * np.sin(zAngleInRadian)
        printthyInM = zdy * np.cos(zAngleInRadian) + hdy * np.sin(zAngleInRadian)
        printthzInM = zdz * np.cos(zAngleInRadian) + hdz * np.sin(zAngleInRadian)

        # Get the bending mode information
        idx1, idx3, bx, by, bz = self._getMirCoor()

        # Calcualte the mirror ideal shape
        zRef = self._calcIdealShape(bx * 1000, by * 1000, idx1, idx3) / 1000

        # Calcualte the mirror ideal shape with the displacement
        zpRef = (
            self._calcIdealShape(
                (bx + printthxInM) * 1000, (by + printthyInM) * 1000, idx1, idx3
            )
            / 1000
        )

        # Convert printthz into surface sag to get the estimated wavefront
        # error.
        # Do the zenith angle correction by the linear approximation with the
        # ideal shape.
        printthzInM = printthzInM - (zpRef - zRef)

        # Normalize the coordinate
        Ri = self.getInnerRinM()[0]
        R = self.getOuterRinM()[0]

        normX = bx / R
        normY = by / R
        obs = Ri / R

        # Fit the annular Zernike polynomials z0-z2 (piton, x-tilt, y-tilt)
        zc = ZernikeAnnularFit(printthzInM, normX, normY, 3, obs)

        # Do the estimated wavefront error correction for the mirror projection
        printthzInM -= ZernikeAnnularEval(zc, normX, normY, obs)

        return printthzInM

    def _getMirCoor(self):
        """Get the mirror coordinate and node.

        Returns
        -------
        numpy.ndarray[int]
            M1 node.
        numpy.ndarray[int]
            M3 node.
        numpy.ndarray
            x coordinate.
        numpy.ndarray
            y coordinate.
        numpy.ndarray
            z coordinate.
        """

        # Get the bending mode information
        data = self._gridFile.getMatContent()

        nodeID = data[:, 0].astype("int")
        nodeM1 = nodeID == 1
        nodeM3 = nodeID == 3

        bx = data[:, 1]
        by = data[:, 2]
        bz = data[:, 3:]

        return nodeM1, nodeM3, bx, by, bz

    def _calcIdealShape(self, xInMm, yInMm, idxM1, idxM3, dr1=0, dr3=0, dk1=0, dk3=0):
        """Calculate the ideal shape of mirror along z direction.

        This is described by a series of cylindrically-symmetric aspheric
        surfaces.

        Parameters
        ----------
        xInMm : numpy.ndarray
            Coordinate x in 1D array in mm.
        yInMm : numpy.ndarray
            Coordinate y in 1D array in mm.
        idxM1 : numpy.ndarray [int]
            M1 node.
        idxM3 : numpy.ndarray [int]
            M3 node.
        dr1 : float, optional
            Displacement of r in mirror 1. (the default is 0.)
        dr3 : float, optional
            Displacement of r in mirror 3. (the default is 0.)
        dk1 : float, optional
            Displacement of kappa (k) in mirror 1. (the default is 0.)
        dk3 : float, optional
            Displacement of kappa (k) in mirror 3. (the default is 0.)

        Returns
        -------
        numpy.ndarray
            Ideal mirror surface along z direction.

        Raises
        ------
        ValueError
            X is unequal to y.
        """

        # M1 optical design
        r1 = -1.9835e4
        k1 = -1.215
        alpha1 = np.zeros((8, 1))
        alpha1[2] = 1.38e-24

        # M3 optical design
        r3 = -8344.5
        k3 = 0.155
        alpha3 = np.zeros((8, 1))
        alpha3[2] = -4.5e-22
        alpha3[3] = -8.15e-30

        # Get the dimension of input xInMm, yInMm
        nr = xInMm.shape
        mr = yInMm.shape
        if nr != mr:
            raise ValueError("X[%d] is unequal to y[%d]." % (nr, mr))

        # Calculation the curvature (c) and conic constant (kappa)

        # Mirror 1 (M1)
        c1 = 1 / (r1 + dr1)
        k1 = k1 + dk1

        # Mirror 3 (M3)
        c3 = 1 / (r3 + dr3)
        k3 = k3 + dk3

        # Construct the curvature, kappa, and alpha matrixes for the ideal
        # shape calculation
        cMat = np.zeros(nr)
        cMat[idxM1] = c1
        cMat[idxM3] = c3

        kMat = np.zeros(nr)
        kMat[idxM1] = k1
        kMat[idxM3] = k3

        alphaMat = np.tile(np.zeros(nr), (8, 1))
        for ii in range(8):
            alphaMat[ii, idxM1] = alpha1[ii]
            alphaMat[ii, idxM3] = alpha3[ii]

        # Calculate the radius
        r2 = xInMm**2 + yInMm**2

        # Calculate the ideal surface

        # The optical elements of telescopes can often be described by a
        # series of cylindrically-symmetric aspheric surfaces:
        # z(r) = c * r^2/[ 1 + sqrt( 1-(1+k) * c^2 * r^2 ) ] +
        # sum(ai * r^(2*i)) + sum(Aj * Zj)
        # where i = 1-8, j = 1-N

        z0 = cMat * r2 / (1 + np.sqrt(1 - (1 + kMat) * cMat**2 * r2))
        for ii in range(8):
            z0 += alphaMat[ii, :] * r2 ** (ii + 1)

        # M3 vertex offset from M1 vertex, values from Zemax model
        # M3voffset = (233.8 - 233.8 - 900 - 3910.701 - 1345.500 + 1725.701
        # + 3530.500 + 900 + 233.800)
        M3voffset = 233.8

        # Add the M3 offset (sum(Aj * Zj), j = 1 - N)
        z0[idxM3] = z0[idxM3] + M3voffset

        # In Zemax, z axis points from M1M3 to M2. the reversed direction
        # (z0>0) is needed. That means the direction of M2 to M1M3.
        return -z0

    def getTempCorr(self, m1m3TBulk, m1m3TxGrad, m1m3TyGrad, m1m3TzGrad, m1m3TrGrad):
        """Get the mirror print correction along z direction for certain
        temperature gradient.

        FEA: Finite element analysis.

        Parameters
        ----------
        m1m3TBulk : float
            Bulk temperature in degree C (+/-2sigma spans +/-0.8C).
        m1m3TxGrad : float
            Temperature gradient along x direction in degree C (+/-2sigma
            spans 0.4C).
        m1m3TyGrad : float
            Temperature gradient along y direction in degree C (+/-2sigma
            spans 0.4C).
        m1m3TzGrad : float
            Temperature gradient along z direction in degree C (+/-2sigma
            spans 0.1C).
        m1m3TrGrad : float
            Temperature gradient along r direction in degree C (+/-2sigma
            spans 0.1C).

        Returns
        -------
        numpy.ndarray
            Corrected projection in um along z direction.
        """

        # Data needed to determine thermal deformation
        data = self._feaFile.getMatContent()

        # These are the normalized coordinates

        # In the original XLS file (thermal FEA model), max(x)=164.6060 in,
        # while 4.18m = 164.5669 in for real mirror. These two numbers do
        # not match.
        # The data here has normalized the dimension (x, y) already.
        # n.b. these may not have been normalized correctly, b/c max(tx)=1.0
        tx = data[:, 0]
        ty = data[:, 1]

        # Below are in M1M3 coordinate system, and in micron

        # Do the fitting in the normalized coordinate
        bx, by = self._getMirCoor()[2:4]
        R = self.getOuterRinM()[0]
        normX = bx / R
        normY = by / R

        # Fit the bulk
        tbdz = self.fitData(tx, ty, data[:, 2], normX, normY)

        # Fit the x-grad
        txdz = self.fitData(tx, ty, data[:, 3], normX, normY)

        # Fit the y-grad
        tydz = self.fitData(tx, ty, data[:, 4], normX, normY)

        # Fit the z-grad
        tzdz = self.fitData(tx, ty, data[:, 5], normX, normY)

        # Fit the r-gradß
        trdz = self.fitData(tx, ty, data[:, 6], normX, normY)

        # Get the temprature correction
        tempCorrInUm = (
            m1m3TBulk * tbdz
            + m1m3TxGrad * txdz
            + m1m3TyGrad * tydz
            + m1m3TzGrad * tzdz
            + m1m3TrGrad * trdz
        )

        return tempCorrInUm

    def getMirrorResInMmInZemax(self, writeZcInMnToFilePath=None):
        """Get the residue of surface (mirror print along z-axis) in mm under
        the Zemax coordinate.

        This value is after the fitting with spherical Zernike polynomials
        (zk).

        Parameters
        ----------
        writeZcInMnToFilePath : str, optional
            File path to write the fitted zk in mm. (the default is None.)

        Returns
        ------
        numpy.ndarray
            Fitted residue in mm after removing the fitted zk terms in Zemax
            coordinate.
        numpy.ndarray
            X position in mm in Zemax coordinate.
        numpy.ndarray
            Y position in mm in Zemax coordinate.
        numpy.ndarray
            Fitted zk in mm in Zemax coordinate.
        """

        # Get the bending mode information
        bx, by, bz = self._getMirCoor()[2:5]

        # Transform the M1M3 coordinate to Zemax coordinate
        bxInZemax, byInZemax, surfInZemax = opt2ZemaxCoorTrans(
            bx, by, self.getSurfAlongZ()
        )

        # Get the mirror residue and zk in um
        RinM = self.getOuterRinM()[0]
        resInUmInZemax, zcInUmInZemax = self._getMirrorResInNormalizedCoor(
            surfInZemax, bxInZemax / RinM, byInZemax / RinM
        )

        # Change the unit to mm
        resInMmInZemax = resInUmInZemax * 1e-3
        bxInMmInZemax = bxInZemax * 1e3
        byInMmInZemax = byInZemax * 1e3
        zcInMmInZemax = zcInUmInZemax * 1e-3

        # Save the file of fitted Zk
        if writeZcInMnToFilePath is not None:
            np.savetxt(writeZcInMnToFilePath, zcInMmInZemax)

        return resInMmInZemax, bxInMmInZemax, byInMmInZemax, zcInMmInZemax

    def writeMirZkAndGridResInZemax(
        self, resFile=[], surfaceGridN=200, writeZcInMnToFilePath=None
    ):
        """Write the grid residue in mm of mirror surface after the fitting
        with Zk under the Zemax coordinate.

        Parameters
        ----------
        resFile : list, optional
            File path to save the grid surface residue map. (the default
            is [].)
        surfaceGridN : {number}, optional
            Surface grid number. (the default is 200.)
        writeZcInMnToFilePath : str, optional
            File path to write the fitted zk in mm. (the default is None.)

        Returns
        -------
        str
            Grid residue map related data of M1.
        str
            Grid residue map related data of M3.
        """

        # Get the residure map
        resInMmInZemax, bxInMmInZemax, byInMmInZemax = self.getMirrorResInMmInZemax(
            writeZcInMnToFilePath=writeZcInMnToFilePath
        )[0:3]

        # Get the mirror node
        idx1, idx3 = self._getMirCoor()[0:2]

        # Grid sample map for M1 and M3
        for ii, idx in zip((0, 1), (idx1, idx3)):

            # Change the unit from m to mm
            innerRinMm = self.getInnerRinM()[ii] * 1e3
            outerRinMm = self.getOuterRinM()[ii] * 1e3

            # Get the residue map used in Zemax
            # Content header: (NUM_X_PIXELS, NUM_Y_PIXELS, delta x, delta y)
            # Content: (z, dx, dy, dxdy)
            content = self._gridSampInMnInZemax(
                resInMmInZemax[idx],
                bxInMmInZemax[idx],
                byInMmInZemax[idx],
                innerRinMm,
                outerRinMm,
                surfaceGridN,
                surfaceGridN,
                resFile=resFile[ii],
            )
            if ii == 0:
                contentM1 = content
            elif ii == 1:
                contentM3 = content

        return contentM1, contentM3

    def showMirResMap(self, resFile, writeToResMapFilePath=[]):
        """Show the mirror residue map.

        Parameters
        ----------
        resFile : list
            File path of the grid surface residue map.
        writeToResMapFilePath : list, optional
            File path to save the residue map. (the default is [].)
        """

        # Get the residure map
        resInMmInZemax, bxInMmInZemax, byInMmInZemax = self.getMirrorResInMmInZemax()[
            0:3
        ]

        # Get the mirror node
        idx1, idx3 = self._getMirCoor()[0:2]

        # Show the mirror maps
        RinMtuple = self.getOuterRinM()

        for ii, RinM, idx in zip((0, 1), RinMtuple, (idx1, idx3)):
            outerRinMm = RinM * 1e3
            plotResMap(
                resInMmInZemax[idx],
                bxInMmInZemax[idx],
                byInMmInZemax[idx],
                outerRinMm,
                resFile=resFile[ii],
                writeToResMapFilePath=writeToResMapFilePath[ii],
            )

    def genMirSurfRandErr(self, zAngleInRadian, m1m3ForceError=0.05, seedNum=0):
        """Generate the mirror surface random error.

        LUT: Loop-up table.

        Parameters
        ----------
        zAngleInRadian : float
            Zenith angle in radian.
        m1m3ForceError : float, optional
            Ratio of actuator force error. (the default is 0.05.)
        seedNum : int, optional
            Random seed number. (the default is 0.)

        Returns
        -------
        numpy.ndarray
            Generated mirror surface random error in m.
        """

        # Get the actuator forces in N of M1M3 based on the look-up table (LUT)
        zangleInDeg = np.rad2deg(zAngleInRadian)
        LUTforce = self.getLUTforce(zangleInDeg)

        # Assume the m1m3ForceError=0.05
        # Add 5% force error to the original actuator forces
        # This means from -5% to +5% of original actuator's force.
        np.random.seed(int(seedNum))
        nActuator = len(LUTforce)
        myu = (1 + 2 * (np.random.rand(nActuator) - 0.5) * m1m3ForceError) * LUTforce

        # Balance forces along z-axis
        # This statement is intentionally to make the force balance.
        nzActuator = int(self._m1m3SettingFile.getSetting("numActuatorInZ"))
        myu[nzActuator - 1] = np.sum(LUTforce[:nzActuator]) - np.sum(
            myu[: nzActuator - 1]
        )

        # Balance forces along y-axis
        # This statement is intentionally to make the force balance.
        myu[nActuator - 1] = np.sum(LUTforce[nzActuator:]) - np.sum(myu[nzActuator:-1])

        # Get the net force along the z-axis
        zf = self._forceZenFile.getMatContent()
        hf = self._forceHorFile.getMatContent()
        u0 = zf * np.cos(zAngleInRadian) + hf * np.sin(zAngleInRadian)

        # Calculate the random surface
        G = self._forceInflFile.getMatContent()
        randSurfInM = G.dot(myu - u0)

        return randSurfInM


if __name__ == "__main__":
    pass
