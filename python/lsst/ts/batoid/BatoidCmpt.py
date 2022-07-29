# This file is part of ts_batoid.
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
import batoid
import re
import shutil
import warnings
import numpy as np
from scipy import ndimage
from astropy.io import fits

from lsst.ts.wep.Utility import runProgram
from lsst.ts.wep.ParamReader import ParamReader

from lsst.ts.batoid.OpdMetrology import OpdMetrology
from lsst.ts.batoid.utils.Utility import getConfigDir, sortOpdFileList
from lsst.ts.batoid.utils.SensorWavefrontError import SensorWavefrontError
from lsst.ts.batoid.wfsim.wfsim import SSTBuilder


class BatoidCmpt(object):
    def __init__(self, instName):
        """Initialization of Batoid component class.

        WEP: wavefront estimation pipeline.

        Parameters
        ----------
        tele : TeleFacade
            Telescope instance.
        """

        # Configuration directory
        self.configDir = getConfigDir()

        # Telescope setting file
        settingFilePath = os.path.join(self.configDir, "batoidCmptSetting.yaml")
        self._batoidCmptSettingFile = ParamReader(filePath=settingFilePath)

        settingFilePath = os.path.join(getConfigDir(), "teleSetting.yaml")
        self._teleSettingFile = ParamReader(filePath=settingFilePath)
        
        # Batoid optic instance
        self.optic = None
        self.builder = None

        # OPD metrology
        self.metr = OpdMetrology()
        self.metr.setCamera(instName)

        # Output directory of data
        self.outputDir = ""

        # Output directory of image
        self.outputImgDir = ""

        # Seed number
        self.seedNum = 0

        # M1M3 force error
        self.m1m3ForceError = 0.05

        self.refWavelength = self._teleSettingFile.getSetting("wavelengthInNm")

    def setM1M3ForceError(self, m1m3ForceError):
        """Set the M1M3 force error.

        Parameters
        ----------
        m1m3ForceError : float
            Ratio of actuator force error between 0 and 1.
        """

        self.m1m3ForceError = m1m3ForceError

    def getM1M3ForceError(self):
        """Get the M1M3 force error.

        Returns
        -------
        float
            Ratio of actuator force error.
        """

        return self.m1m3ForceError

    def getOptic(self):
        """Get the optic object.

        Returns
        -------
        Optic
            Optic batoid object.
        """

        return self.optic

    def setOptic(
        self,
        addCam=True, 
        addM1M3=True, 
        addM2=True
    ):

        builder = SSTBuilder(batoid.Optic.fromYaml("LSST_g_500.yaml"))

        if addM1M3:
            builder = (
                builder
                .with_m1m3_gravity(np.deg2rad(self.zAngleInDeg))
                .with_m1m3_temperature(
                    m1m3_TBulk=self._teleSettingFile.getSetting("m1m3TBulk"),
                    m1m3_TxGrad=self._teleSettingFile.getSetting("m1m3TxGrad"),
                    m1m3_TyGrad=self._teleSettingFile.getSetting("m1m3TyGrad"),
                    m1m3_TzGrad=self._teleSettingFile.getSetting("m1m3TzGrad"),
                    m1m3_TrGrad=self._teleSettingFile.getSetting("m1m3TrGrad")
                )
                .with_m1m3_lut(
                    np.deg2rad(self.zAngleInDeg), 
                    error=self.m1m3ForceError, 
                    seed=self.seedNum
                )
            )
        if addM2:
            builder = (
                builder
                .with_m2_gravity(np.deg2rad(self.zAngleInDeg))
                .with_m2_temperature(
                    m2_TrGrad=self._teleSettingFile.getSetting("m2TrGrad"),
                    m2_TzGrad=self._teleSettingFile.getSetting("m2TzGrad")
                )
            )
        if addCam:
            builder = (
                builder
                .with_camera_gravity(
                    zenith_angle=np.deg2rad(self.zAngleInDeg),
                    rotation_angle=np.deg2rad(self.rotAngInDeg)
                )
                .with_camera_temperature(
                    camera_TBulk=self._teleSettingFile.getSetting("camTB")
                )
            )
        
        self.builder = builder
        self.optic =  builder.build()

    def getTeleSettingFile(self):
        """Get the setting file.

        Returns
        -------
        lsst.ts.wep.ParamReader
            Setting file.
        """

        return self._teleSettingFile


    def getSettingFile(self):
        """Get the setting file.

        Returns
        -------
        lsst.ts.wep.ParamReader
            Setting file.
        """

        return self._batoidCmptSettingFile

    def getNumOfZk(self):
        """Get the number of Zk (annular Zernike polynomial).

        Returns
        -------
        int
            Number of Zk.
        """

        return int(self._batoidCmptSettingFile.getSetting("numOfZk"))

    def getIntraFocalDirName(self):
        """Get the intra-focal directory name.

        Returns
        -------
        str
            Intra-focal directory name.
        """

        return self._batoidCmptSettingFile.getSetting("intraDirName")

    def getExtraFocalDirName(self):
        """Get the extra-focal directory name.

        Returns
        -------
        str
            Extra-focal directory name.
        """

        return self._batoidCmptSettingFile.getSetting("extraDirName")

    def getWfsDirName(self):
        """Get the WFS directory name.

        Returns
        -------
        str
            WFS directory name.
        """

        return self._batoidCmptSettingFile.getSetting("wfsDirName")

    def getOpdMetr(self):
        """Get the OPD metrology object.

        OPD: optical path difference.

        Returns
        -------
        OpdMetrology
            OPD metrology object.
        """

        return self.metr

    def setOutputDir(self, outputDir):
        """Set the output directory.

        The output directory will be constructed if there is no existed one.

        Parameters
        ----------
        outputDir : str
            Output directory.
        """

        self._makeDir(outputDir)
        self.outputDir = outputDir

    def _makeDir(self, newDir, exist_ok=True):
        """Make the new directory.

        Super-mkdir; create a leaf directory and all intermediate ones. Works
        like mkdir, except that any intermediate path segment (not just the
        rightmost) will be created if it does not exist.

        Parameters
        ----------
        newDir : str
            New directory.
        exist_ok : bool, optional
            If the target directory already exists, raise an OSError if
            exist_ok is False. Otherwise no exception is raised. (the default
            is True.)
        """

        os.makedirs(newDir, exist_ok=exist_ok)

    def getOutputDir(self):
        """Get the output directory.

        Returns
        -------
        str
            Output directory.
        """

        return self.outputDir

    def setOutputImgDir(self, outputImgDir):
        """Set the output image directory.

        The output image directory will be constructed if there is no existed
        one.

        Parameters
        ----------
        outputImgDir : str
            Output image directory
        """

        self._makeDir(outputImgDir)
        self.outputImgDir = outputImgDir

    def getOutputImgDir(self):
        """Get the output image directory.

        Returns
        -------
        str
            Output image directory
        """

        return self.outputImgDir

    def setSeedNum(self, seedNum):
        """Set the seed number for the M1M3 mirror surface purturbation.

        Parameters
        ----------
        seedNum : int
            Seed number.
        """

        self.seedNum = int(seedNum)

    def getSeedNum(self):
        """Get the seed number for the M1M3 random surface purturbation.

        Returns
        -------
        int or None
            Seed number. None means there is no random purturbation.
        """

        return self.seedNum

    def setSurveyParam(
        self,
        obsId=None,
        filterType=None,
        boresight=None,
        zAngleInDeg=None,
        rotAngInDeg=None,
    ):
        """Set the survey parameters.

        Parameters
        ----------
        obsId : int, optional
            Observation Id. (the default is None.)
        filterType : enum 'FilterType' in lsst.ts.wep.Utility, optional
            Active filter type. (the default is None.)
        boresight : tuple, optional
            Telescope boresight in (ra, decl). (the default is None.)
        zAngleInDeg : float, optional
            Zenith angle in degree. (the default is None.)
        rotAngInDeg : float, optional
            Camera rotation angle in degree between -90 and 90 degrees. (the
            default is None.)
        """

        self.filterType = filterType
        self.obsId = obsId
        self.boresight = boresight
        self.zAngleInDeg = zAngleInDeg
        self.rotAngInDeg = rotAngInDeg


    def addOpdFieldXYbyDeg(self, fieldXInDegree, fieldYInDegree):
        """Add the OPD new field X, Y in degree.

        OPD: optical path difference.

        Parameters
        ----------
        fieldXInDegree : float, list, or numpy.ndarray
            New field X in degree.
        fieldYInDegree : float, list, or numpy.ndarray
            New field Y in degree.
        """

        self.metr.addFieldXYbyDeg(fieldXInDegree, fieldYInDegree)

    def accDofInUm(self, dofInUm):
        """Accumulate the aggregated degree of freedom (DOF) in um.

        idx 0-4: M2 dz, dx, dy, rx, ry
        idx 5-9: Cam dz, dx, dy, rx, ry
        idx 10-29: M1M3 20 bending modes
        idx 30-49: M2 20 bending modes

        Parameters
        ----------
        dofInUm : list or numpy.ndarray
            DOF in um.
        """

        self.builder = self.builder.with_aos_dof(dofInUm + self.builder.dof)
        self.optic = self.builder.build()

    def setDofInUm(self, dofInUm):
        """Set the accumulated degree of freedom (DOF) in um.

        idx 0-4: M2 dz, dx, dy, rx, ry
        idx 5-9: Cam dz, dx, dy, rx, ry
        idx 10-29: M1M3 20 bending modes
        idx 30-49: M2 20 bending modes

        Parameters
        ----------
        dofInUm : list or numpy.ndarray
            DOF in um.
        """

        self.builder = self.builder.with_aos_dof(dofInUm)
        self.optic = self.builder.build()

    def getDofInUm(self):
        """Get the accumulated degree of freedom (DOF) in um.

        idx 0-4: M2 dz, dx, dy, rx, ry
        idx 5-9: Cam dz, dx, dy, rx, ry
        idx 10-29: M1M3 20 bending modes
        idx 30-49: M2 20 bending modes

        Returns
        -------
        numpy.ndarray
            DOF in um.
        """

        return self.builder.dof

    def saveDofInUmFileForNextIter(
        self, dofInUm, dofInUmFileName="dofPertInNextIter.mat"
    ):
        """Save the DOF in um data to file for the next iteration.

        DOF: degree of freedom.

        Parameters
        ----------
        dofInUm : list or numpy.ndarray
            DOF in um.
        dofInUmFileName : str, optional
            File name to save the DOF in um. (the default is
            "dofPertInNextIter.mat".)
        """

        filePath = os.path.join(self.outputDir, dofInUmFileName)
        header = "The followings are the DOF in um:"
        np.savetxt(filePath, np.transpose(dofInUm), header=header)

    def runBatoid(
        self, obsId, sensorIdList, sensorLocationList
    ):
        """Run the Batoid program.

        Parameters
        ----------
        argString : str
            Arguments for Batoid.
        """

        listOfWfErr = []
        numOfZk = self.getNumOfZk()

        for sensorId, sensorLocation in zip(sensorIdList, sensorLocationList):

            zk = batoid.zernike(
                self.optic,
                sensorLocation[0], sensorLocation[1],
                self.refWavelength, 
                eps=0.61, 
                jmax = numOfZk + 3, 
                nx=25
            ) * self.refWavelength * 1e6

            sensorWavefrontData = SensorWavefrontError(numOfZk=numOfZk)
            sensorWavefrontData.setSensorId(sensorId)
            #Batoid returns null zk[0] which is unused
            sensorWavefrontData.setAnnularZernikePoly(zk[4:])

            listOfWfErr.append(sensorWavefrontData)

            opd = batoid.wavefront(
                self.optic,
                sensorLocation[0], sensorLocation[1],
                wavelength=self.refWavelength, 
                nx=255
            ).array * self.refWavelength * 1e6
            
            opddata = opd.data.astype(np.float32)
            opddata[opd.mask] = 0.0
            ofn = os.path.join(self.outputDir, f"opd_{obsId}_{sensorId}.fits.gz")
            fits.writeto(
                ofn,
                opddata,
                overwrite=True
            )

        return listOfWfErr


    def _writePertAndCmdFiles(self, cmdSettingFileName, cmdFileName):
        """Write the physical perturbation and command files.

        Parameters
        ----------
        cmdSettingFileName : str
            Physical command setting file name.
        cmdFileName : str
            Physical command file name.

        Returns
        -------
        str
            Command file path.
        """

        # Write the perturbation file
        pertCmdFileName = "pert.cmd"
        pertCmdFilePath = os.path.join(self.outputDir, pertCmdFileName)
        if not os.path.exists(pertCmdFilePath):
            self.tele.writePertBaseOnConfigFile(
                self.outputDir,
                seedNum=self.seedNum,
                m1m3ForceError=self.m1m3ForceError,
                saveResMapFig=True,
                pertCmdFileName=pertCmdFileName,
            )

        # Write the physical command file
        cmdSettingFile = os.path.join(self.configDir, "cmdFile", cmdSettingFileName)
        cmdFilePath = os.path.join(self.outputDir, cmdFileName)
        if not os.path.exists(cmdFilePath):
            self.tele.writeCmdFile(
                self.outputDir,
                cmdSettingFile=cmdSettingFile,
                pertFilePath=pertCmdFilePath,
                cmdFileName=cmdFileName,
            )

        return cmdFilePath

    def _getInstSettingFilePath(self, instSettingFileName):
        """Get the instance setting file path.

        Parameters
        ----------
        instSettingFileName : str
            Instance setting file name.

        Returns
        -------
        str
            Instance setting file path.
        """

        instSettingFile = os.path.join(self.configDir, "instFile", instSettingFileName)

        return instSettingFile

    def analyzeComCamOpdData(
        self, zkFileName="opd.zer", rotOpdInDeg=0.0, pssnFileName="PSSN.txt"
    ):
        """Analyze the ComCam OPD data.

        Rotate OPD to simulate the output by rotated camera. When anaylzing the
        PSSN, the unrotated OPD is used.

        ComCam: Commissioning camera.
        OPD: Optical path difference.
        PSSN: Normalized point source sensitivity.

        Parameters
        ----------
        zkFileName : str, optional
            OPD in zk file name. (the default is "opd.zer".)
        rotOpdInDeg : float, optional
            Rotate OPD in degree in the counter-clockwise direction. (the
            default is 0.0.)
        pssnFileName : str, optional
            PSSN file name. (the default is "PSSN.txt".)
        """

        warnings.warn(
            "Use analyzeOpdData() instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.analyzeOpdData(
            "comcam",
            zkFileName=zkFileName,
            rotOpdInDeg=rotOpdInDeg,
            pssnFileName=pssnFileName,
        )

    def analyzeOpdData(
        self, instName, zkFileName="opd.zer", rotOpdInDeg=0.0, pssnFileName="PSSN.txt"
    ):
        """Analyze the OPD data.

        Rotate OPD to simulate the output by rotated camera. When anaylzing the
        PSSN, the unrotated OPD is used.

        OPD: Optical path difference.
        PSSN: Normalized point source sensitivity.

        Parameters
        ----------
        instName : `str`
            Instrument name.
        zkFileName : str, optional
            OPD in zk file name. (the default is "opd.zer".)
        rotOpdInDeg : float, optional
            Rotate OPD in degree in the counter-clockwise direction. (the
            default is 0.0.)
        pssnFileName : str, optional
            PSSN file name. (the default is "PSSN.txt".)
        """

        self._writeOpdZkFile(zkFileName, rotOpdInDeg)
        self._writeOpdPssnFile(instName, pssnFileName)

    def _writeOpdZkFile(self, zkFileName, rotOpdInDeg):
        """Write the OPD in zk file.

        OPD: optical path difference.

        Parameters
        ----------
        zkFileName : str
            OPD in zk file name.
        rotOpdInDeg : float
            Rotate OPD in degree in the counter-clockwise direction.
        """

        filePath = os.path.join(self.outputImgDir, zkFileName)
        opdData = self._mapOpdToZk(rotOpdInDeg)
        header = (
            "The followings are OPD in rotation angle of %.2f degree in um from z4 to z22:"
            % (rotOpdInDeg)
        )
        np.savetxt(filePath, opdData, header=header)

    def _mapOpdToZk(self, rotOpdInDeg):
        """Map the OPD to the basis of annular Zernike polynomial (Zk).

        OPD: optical path difference.

        Parameters
        ----------
        rotOpdInDeg : float
            Rotate OPD in degree in the counter-clockwise direction.

        Returns
        -------
        numpy.ndarray
            Zk data from OPD. This is a 2D array. The row is the OPD index and
            the column is z4 to z22 in um. The order of OPD index is based on
            the file name.
        """

        # Get the sorted OPD file list
        opdFileList = self._getOpdFileInDir(self.outputImgDir)

        # Map the OPD to the Zk basis and do the collection
        numOfZk = self.getNumOfZk()
        opdData = np.zeros((len(opdFileList), numOfZk))
        for idx, opdFile in enumerate(opdFileList):
            opd = fits.getdata(opdFile)

            # Rotate OPD if needed
            if rotOpdInDeg != 0:
                opdRot = ndimage.rotate(opd, rotOpdInDeg, reshape=False)
                opdRot[opd == 0] = 0
            else:
                opdRot = opd

            # z1 to z22 (22 terms)
            zk = self.metr.getZkFromOpd(opdMap=opdRot)[0]

            # Only need to collect z4 to z22
            initIdx = 3
            opdData[idx, :] = zk[initIdx : initIdx + numOfZk]

        return opdData

    def _getOpdFileInDir(self, opdDir):
        """Get the sorted OPD files in the directory.

        OPD: Optical path difference.

        Parameters
        ----------
        opdDir : str
            OPD file directory.

        Returns
        -------
        list
            List of sorted OPD files.
        """

        # Get the files
        opdFileList = []
        fileList = self._getFileInDir(opdDir)
        for file in fileList:
            fileName = os.path.basename(file)
            m = re.match(r"\Aopd_\d+_(\d+).fits.gz", fileName)
            if m is not None:
                opdFileList.append(file)

        # Do the sorting of file name
        sortedOpdFileList = sortOpdFileList(opdFileList)

        return sortedOpdFileList

    def _getFileInDir(self, fileDir):
        """Get the files in the directory.

        Parameters
        ----------
        fileDir : str
            File directory.

        Returns
        -------
        list
            List of files.
        """

        fileList = []
        for name in os.listdir(fileDir):
            filePath = os.path.join(fileDir, name)
            if os.path.isfile(filePath):
                fileList.append(filePath)

        return fileList

    def _writeOpdPssnFile(self, instName, pssnFileName):
        """Write the OPD PSSN in file.

        OPD: Optical path difference.
        PSSN: Normalized point source sensitivity.

        Parameters
        ----------
        instName : `str`
            Instrument name.
        pssnFileName : str
            PSSN file name.
        """

        # Set the weighting ratio and field positions of OPD
        if instName == "lsst":
            self.metr.setDefaultLsstWfsGQ()
        else:
            self.metr.setWgtAndFieldXyOfGQ(instName)

        # Calculate the PSSN
        pssnList, gqEffPssn = self._calcPssnOpd()

        # Calculate the FWHM
        effFwhmList, gqEffFwhm = self._calcEffFwhmOpd(pssnList)

        # Append the list to write the data into file
        pssnList.append(gqEffPssn)
        effFwhmList.append(gqEffFwhm)

        # Stack the data
        data = np.vstack((pssnList, effFwhmList))

        # Write to file
        filePath = os.path.join(self.outputImgDir, pssnFileName)
        header = "The followings are PSSN and FWHM (in arcsec) data. The final number is the GQ value."
        np.savetxt(filePath, data, header=header)

    def _calcPssnOpd(self):
        """Calculate the PSSN of OPD.

        OPD: Optical path difference.
        PSSN: Normalized point source sensitivity.
        GQ: Gaussian quadrature.

        Returns
        -------
        list
            PSSN list.
        float
            GQ effective PSSN.
        """

        opdFileList = self._getOpdFileInDir(self.outputImgDir)

        wavelengthInUm = self.refWavelength * 1e6
        pssnList = []
        for opdFile in opdFileList:
            pssn = self.metr.calcPSSN(wavelengthInUm, opdFitsFile=opdFile)
            pssnList.append(pssn)

        # Calculate the GQ effectice PSSN
        gqEffPssn = self.metr.calcGQvalue(pssnList)

        return pssnList, gqEffPssn

    def _calcEffFwhmOpd(self, pssnList):
        """Calculate the effective FWHM of OPD.

        FWHM: Full width and half maximum.
        PSSN: Normalized point source sensitivity.
        GQ: Gaussian quadrature.

        Parameters
        ----------
        pssnList : list
            List of PSSN.

        Returns
        -------
        list
            Effective FWHM list.
        float
            GQ effective FWHM.
        """

        # Calculate the list of effective FWHM
        effFwhmList = []
        for pssn in pssnList:
            effFwhm = self.metr.calcFWHMeff(pssn)
            effFwhmList.append(effFwhm)

        # Calculate the GQ effectice FWHM
        gqEffFwhm = self.metr.calcGQvalue(effFwhmList)

        return effFwhmList, gqEffFwhm

    def mapOpdDataToListOfWfErr(self, opdZkFileName, sensorIdList):
        """Map the OPD data to the list of wavefront error.

        OPD: Optical path difference.

        Parameters
        ----------
        opdZkFileName : str
            OPD zk file name.
        sensorIdList : list
            Reference sensor name list.

        Returns
        -------
        list [lsst.ts.wep.ctrlIntf.SensorWavefrontError]
            List of SensorWavefrontError object.
        """

        opdZk = self._getZkFromFile(opdZkFileName)

        listOfWfErr = []
        for sensorId, zk in zip(sensorIdList, opdZk):

            sensorWavefrontData = SensorWavefrontError(numOfZk=self.getNumOfZk())
            sensorWavefrontData.setSensorId(sensorId)
            sensorWavefrontData.setAnnularZernikePoly(zk)

            listOfWfErr.append(sensorWavefrontData)

        return listOfWfErr

    def _getZkFromFile(self, zkFileName):
        """Get the zk (z4-z22) from file.

        Parameters
        ----------
        zkFileName : str
            Zk file name.

        Returns
        -------
        numpy.ndarray
            zk matrix. The colunm is z4-z22. The raw is each data point.
        """

        filePath = os.path.join(self.outputImgDir, zkFileName)
        zk = np.loadtxt(filePath)

        return zk

    def getOpdPssnFromFile(self, pssnFileName):
        """Get the OPD PSSN from file.

        OPD: Optical path difference.
        PSSN: Normalized point source sensitivity.

        Parameters
        ----------
        pssnFileName : str
            PSSN file name.

        Returns
        -------
        numpy.ndarray
            PSSN.
        """

        data = self._getDataOfPssnFile(pssnFileName)
        pssn = data[0, :-1]

        return pssn

    def _getDataOfPssnFile(self, pssnFileName):
        """Get the data of the PSSN file.

        PSSN: Normalized point source sensitivity.

        Parameters
        ----------
        pssnFileName : str
            PSSN file name.

        Returns
        -------
        numpy.ndarray
            Data of the PSSN file.
        """

        filePath = os.path.join(self.outputImgDir, pssnFileName)
        data = np.loadtxt(filePath)

        return data

    def getOpdGqEffFwhmFromFile(self, pssnFileName):
        """Get the OPD GQ effective FWHM from file.

        OPD: Optical path difference.
        GQ: Gaussian quadrature.
        FWHM: Full width at half maximum.
        PSSN: Normalized point source sensitivity.

        Parameters
        ----------
        pssnFileName : str
            PSSN file name.

        Returns
        -------
        float
            OPD GQ effective FWHM.
        """

        data = self._getDataOfPssnFile(pssnFileName)
        gqEffFwhm = data[1, -1]

        return gqEffFwhm

    def getListOfFwhmSensorData(self, pssnFileName, sensorIdList):
        """Get the list of FWHM sensor data based on the OPD PSSN file.

        FWHM: Full width at half maximum.
        OPD: Optical path difference.
        PSSN: Normalized point source sensitivity.

        Parameters
        ----------
        pssnFileName : str
            PSSN file name.
        sensorIdList : list
            Reference sensor id list.

        Returns
        -------
        fwhmCollection : `np.ndarray [object]`
            Numpy array with fwhm data. This is a numpy array of arrays. The
            data type is `object` because each element may have different
            number of elements.
        sensor_id: `np.ndarray`
            Numpy array with sensor ids.
        """

        # Get the FWHM data from the PSSN file
        # The first row is the PSSN and the second one is the FWHM
        # The final element in each row is the GQ value
        data = self._getDataOfPssnFile(pssnFileName)
        fwhmData = data[1, :-1]

        sensor_id = np.array(sensorIdList, dtype=int)

        fwhmCollection = np.array([], dtype=object)
        for fwhm in fwhmData:
            fwhmCollection = np.append(fwhmCollection, fwhm)

        return fwhmCollection, sensor_id

