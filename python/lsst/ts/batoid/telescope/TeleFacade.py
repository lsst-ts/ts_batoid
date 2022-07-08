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
import numpy as np

from lsst.ts.phosim.telescope.CamSim import CamSim
from lsst.ts.phosim.telescope.M1M3Sim import M1M3Sim
from lsst.ts.phosim.telescope.M2Sim import M2Sim
from lsst.ts.phosim.telescope.PhosimCommu import PhosimCommu

from lsst.ts.phosim.utils.Utility import (
    SurfaceType,
    CamDistType,
    getConfigDir,
    mapSurfNameToEnum,
)

from lsst.ts.wep.Utility import FilterType, CamType, mapFilterRefToG
from lsst.ts.wep.ParamReader import ParamReader


class TeleFacade(object):
    def __init__(self):
        """Initialization of telescope facade class.

        This class uses the facade pattern that the high level class telescope
        helps to write the perturbations of camera, M1M3, and M2 into the
        PhoSim by the interface to PhoSim.
        """

        # Telescope setting file
        settingFilePath = os.path.join(getConfigDir(), "teleSetting.yaml")
        self._teleSettingFile = ParamReader(filePath=settingFilePath)

        # Camera subsystem
        self.cam = None

        # M1M3 subsystem
        self.m1m3 = None

        # M2 subsystem
        self.m2 = None

        # PhoSim communication
        self.phoSimCommu = PhosimCommu()

        # Telescope aggregated DOF in um
        # This value will be put back to PhoSim to do the perturbation
        numOfDof = self.getNumOfDof()
        self.dofInUm = np.zeros(numOfDof)

        # Telescopt survery parameters
        defocalDist = self.getDefaultDefocalDist()
        self.surveyParam = {
            "instName": "lsst",
            "defocalDistInMm": defocalDist,
            "obsId": 9999,
            "filterType": FilterType.REF,
            "boresight": (0, 0),
            "zAngleInDeg": 0,
            "rotAngInDeg": 0,
        }

        # Sensors on the telescope
        self.sensorOn = {"sciSensorOn": True, "wfSensorOn": True, "guidSensorOn": False}

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

        return self.dofInUm

    def getSurveyParam(self):
        """Get the survey parameters of telescope.

        Returns
        -------
        dict
            Survey parameters of telescope.
        """

        return self.surveyParam

    def getNumOfDof(self):
        """Get the number of DOF in the setting file.

        DOF: Degree of freedom.

        Returns
        -------
        int
            Number of DOF.
        """

        return int(self._teleSettingFile.getSetting("numOfDof"))

    def getDefaultDefocalDist(self):
        """Get the default defocal distance in mm.

        Returns
        -------
        float
            Default defocal distance in mm.
        """

        return self._teleSettingFile.getSetting("defocalDist")

    def getSurfGridN(self):
        """Get the number of grid of surface

        Returns
        -------
        int
            Number of grid of surface.
        """

        return int(self._teleSettingFile.getSetting("surfaceGridN"))

    def getCamMjd(self):
        """Get the camera MJD.

        Returns
        -------
        float
            Camera MJD.
        """

        return self._teleSettingFile.getSetting("cameraMJD")

    def getRefWaveLength(self):
        """Get the reference wavelength in nm.

        Returns
        -------
        int
            Reference wavelength in nm.
        """

        return self._teleSettingFile.getSetting("wavelengthInNm")

    def getDefocalDistInMm(self):
        """Get the defocal distance in mm.

        Returns
        -------
        float
            defocal distance in mm.
        """

        return self.surveyParam["defocalDistInMm"]

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

        self._setSurveyParamItem("obsId", obsId, int)
        self._setSurveyParamItem("filterType", filterType, FilterType)
        self._setSurveyParamItem("boresight", boresight, tuple)
        self._setSurveyParamItem("zAngleInDeg", zAngleInDeg, (int, float))
        self._setSurveyParamItem("rotAngInDeg", rotAngInDeg, (int, float))

    def _setSurveyParamItem(self, dictKeyName, varValue, varType):
        """Set the item in the dictionary of survey parameters.

        Parameters
        ----------
        dictKeyName : str
            Name of dictionary key.
        varValue : int, tuple, float, or enum 'FilterType'
            Variable value.
        varType: int, tuple, float, or enum 'FilterType'
            Variable type.
        """

        if isinstance(varValue, varType):
            self.surveyParam[dictKeyName] = varValue

    def setSensorOn(self, sciSensorOn=True, wfSensorOn=True, guidSensorOn=False):
        """Set the sensor on.

        Parameters
        ----------
        sciSensorOn : bool, optional
            Scientific sensors are on. (the default is True.)
        wfSensorOn : bool, optional
            Wavefront sensors are on. (the default is True.)
        guidSensorOn : bool, optional
            Guider sensors are on. (the default is False.)
        """

        self.sensorOn["sciSensorOn"] = sciSensorOn
        self.sensorOn["wfSensorOn"] = wfSensorOn
        self.sensorOn["guidSensorOn"] = guidSensorOn

    def runPhoSim(self, argString):
        """

        Run the PhoSim program.

        Arguments:
            argString {[str]} -- Arguments for PhoSim.
        """

        self.phoSimCommu.runPhoSim(argstring=argString)

    def getPhoSimArgs(
        self,
        instFilePath,
        extraCommandFile=None,
        numPro=1,
        numThread=1,
        outputDir=None,
        sensorName=None,
        e2ADC=1,
        logFilePath=None,
    ):
        """Get the arguments needed to run the PhoSim.

        Parameters
        ----------
        instanceFile : str
            Instance catalog file.
        extraCommandFile : str, optional
            Command file to modify the default physics. (the default is None.)
        numProc : int, optional
             Number of processors. (the default is 1.)
        numThread : int, optional
            Number of threads. (the default is 1.)
        outputDir : str, optional
            Output image directory. (the default is None.)
        sensorName : str, optional
            Sensor chip specification (e.g., all, R22_S11, "R22_S11|R22_S12").
            (the default is None.)
        e2ADC : int, optional
            Whether to generate amplifier images (1 = true, 0 = false). (the
            default is 1.)
        logFilePath : str, optional
            Log file path of PhoSim calculation. (the default is None.)

        Returns
        -------
        str
            Arguments to run the PhoSim.
        """

        instName = self.surveyParam["instName"]
        argString = self.phoSimCommu.getPhoSimArgs(
            instFilePath,
            extraCommandFile=extraCommandFile,
            numProc=numPro,
            numThread=numThread,
            outputDir=outputDir,
            instrument=instName,
            sensorName=sensorName,
            e2ADC=e2ADC,
            logFilePath=logFilePath,
        )

        return argString

    def setInstName(self, camType, defocalDist=None):
        """Set the instrument name.

        Parameters
        ----------
        camType : enum 'CamType' in lsst.ts.wep.Utility
            Camera type.
        defocalDist : float, optional
            Defocal distance in mm. If None, the default value will be used.
            (the default is None.)

        Raises
        ------
        ValueError
            Defocal distance can only be = 0 with LsstCam.
        ValueError
            Defocal distance can not be < 0.
        """

        if defocalDist is None:
            defocalDist = self.getDefaultDefocalDist()

        if (defocalDist == 0) and (camType != CamType.LsstCam):
            raise ValueError(
                "Defocal distance can only be zero if working with ",
                "LSSTCam and corner wavefront sensors.",
            )

        if defocalDist < 0:
            raise ValueError("Defocal distance can not be < 0.")

        self.surveyParam["defocalDistInMm"] = defocalDist

        self._setInstNameBasedOnCamType(camType)

    def _setInstNameBasedOnCamType(self, camType):
        """Set the instrument name based on the camera type.

        Parameters
        ----------
        camType : enum 'CamType' in lsst.ts.wep.Utility
            Camera type.

        Raises
        ------
        ValueError
            This camera type is not supported yet.
        """

        if camType in (CamType.LsstCam, CamType.LsstFamCam):
            instName = "lsst"
        elif camType == CamType.ComCam:
            instName = "comcam"
        else:
            raise ValueError("This camera type (%s) is not supported yet." % camType)

        self.surveyParam["instName"] = instName

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

        self._checkNumOfDof(dofInUm)
        self.dofInUm = np.array(dofInUm, dtype=float)

    def _checkNumOfDof(self, dof):
        """Check the number of DOF:

        DOF: Degree of freedom.

        Parameters
        ----------
        dof : list or numpy.ndarray
            DOF.

        Raises
        ------
        ValueError
            The size of DOF should be 50.
        """

        numOfDof = self.getNumOfDof()

        if len(dof) != numOfDof:
            raise ValueError("The size of DOF should be %d." % numOfDof)

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

        self._checkNumOfDof(dofInUm)
        self.dofInUm += np.array(dofInUm, dtype=float)

    def addSubSys(self, addCam=False, addM1M3=False, addM2=False):
        """Add the sub systems to do the perturbation.

        Parameters
        ----------
        addCam : bool, optional
            Add the camera. (the default is False.)
        addM1M3 : bool, optional
            Add the M1M3 mirror. (the default is False.)
        addM2 : bool, optional
            Add the M2 mirror. (the default is False.)
        """

        if addCam:
            self.cam = CamSim()

        if addM1M3:
            self.m1m3 = M1M3Sim()

        if addM2:
            self.m2 = M2Sim()

    def setPhoSimDir(self, phosimDir):
        """Set the directory of PhoSim.

        Parameters
        ----------
        phosimDir : str
            Directory of PhoSim.
        """

        self.phoSimCommu.setPhoSimDir(phosimDir)

    def writeAccDofFile(self, outputFileDir, dofFileName="pert.mat"):
        """Write the accumulated degree of freedom (DOF) in um to file.

        Parameters
        ----------
        outputFileDir : str
            Output file directory.
        dofFileName : str, optional
            DOF file name. (the default is "pert.mat".)

        Returns
        -------
        str
            DOF file path.
        """

        dofFilePath = os.path.join(outputFileDir, dofFileName)
        np.savetxt(dofFilePath, self.dofInUm)

        return dofFilePath

    def writeCmdFile(
        self,
        cmdFileDir,
        cmdSettingFile=None,
        pertFilePath=None,
        cmdFileName="taskPert.cmd",
    ):
        """Write the physical command file.

        OPD: Optical path difference.

        Parameters
        ----------
        cmdFileDir : str
            Directory to the OPD command file.
        cmdSettingFile : str, optional
            Physical command setting file path. (the default is None.)
        pertFilePath : str, optional
            Subsystem perturbation command file path. (the default is None.)
        cmdFileName : str, optional
            Command file name. (the default is "taskPert.cmd".)

        Returns
        -------
        str
            Command file path.
        """

        # Command file path
        cmdFilePath = os.path.join(cmdFileDir, cmdFileName)
        self.phoSimCommu.writeToFile(cmdFilePath, content="", mode="w")

        # Write the physical setting
        if cmdSettingFile is not None:
            self.phoSimCommu.writeToFile(cmdFilePath, sourceFile=cmdSettingFile)

        # Add the subsystem perturbation
        if pertFilePath is not None:
            self.phoSimCommu.writeToFile(cmdFilePath, sourceFile=pertFilePath)

        return cmdFilePath

    def writeStarInstFile(
        self,
        instFileDir,
        skySim,
        simSeed=1000,
        sedName="sed_flat.txt",
        instSettingFile=None,
        instFileName="star.inst",
    ):
        """Write the star instance file.

        Parameters
        ----------
        instFileDir : str
            Directory to instance file.
        skySim : SkySim
            SkySim object.
        simSeed : int, optional
            Random number seed. (the default is 1000.)
        sedName : str, optional
            The name of the SED file with a file path that is relative to the
            data directory in PhoSim. (the default is "sed_flat.txt".)
        instSettingFile : str optional
            Instance setting file. (the default is None.)
        instFileName : str, optional
            Star instance file name. (the default is "star.inst".)

        Returns
        -------
        str
            Instance file path.
        """

        # Instance file path
        instFilePath = os.path.join(instFileDir, instFileName)

        # Get the filter ID in PhoSim
        aFilterId = self._getFilterIdInPhoSim()

        # Write the default instance setting
        obsId = self.surveyParam["obsId"]

        boresight = self.surveyParam["boresight"]
        ra = boresight[0]
        dec = boresight[1]

        rot = self.surveyParam["rotAngInDeg"]
        mjd = self.getCamMjd()

        self.phoSimCommu.getStarInstance(
            obsId,
            aFilterId,
            ra=ra,
            dec=dec,
            rot=rot,
            mjd=mjd,
            simSeed=simSeed,
            filePath=instFilePath,
        )

        if instFilePath is not None:
            self.phoSimCommu.writeToFile(instFilePath, sourceFile=instSettingFile)

        # Write the telescope accumulated degree of freedom (DOF)
        content = self.phoSimCommu.doDofPert(self.dofInUm)

        # Set the camera configuration
        sciSensorOn = self.sensorOn["sciSensorOn"]
        wfSensorOn = self.sensorOn["wfSensorOn"]
        guidSensorOn = self.sensorOn["guidSensorOn"]
        content += self.phoSimCommu.doCameraConfig(
            sciSensorOn=sciSensorOn, wfSensorOn=wfSensorOn, guidSensorOn=guidSensorOn
        )

        # Use the SED file of single wavelendth if the reference filter is
        # used.
        filterType = self.surveyParam["filterType"]
        if filterType == FilterType.REF:
            self._writeSedFileIfPhoSimDirSet()
            sedName = "sed_%s.txt" % int(self.getRefWaveLength())

        # Write the star source
        starId = skySim.getStarId()
        ra, decl = skySim.getRaDecInDeg()
        mag = skySim.getStarMag()
        for idx in range(len(starId)):
            content += self.phoSimCommu.generateStar(
                starId[idx], ra[idx], decl[idx], mag[idx], sedName
            )
        self.phoSimCommu.writeToFile(instFilePath, content=content)

        return instFilePath

    def _getFilterIdInPhoSim(self):
        """Get the active filter Id used in PhoSim.

        Returns
        -------
        int
            Active filter ID in PhoSim.
        """

        filterType = self.surveyParam["filterType"]
        mappedFilterType = mapFilterRefToG(filterType)

        return self.phoSimCommu.getFilterId(mappedFilterType)

    def _writeSedFileIfPhoSimDirSet(self):
        """Write the SED file if the PhoSim directory is set.

        SED: Spectral energy distribution.
        """

        if os.path.isdir(self.phoSimCommu.getPhoSimDir()):
            self.phoSimCommu.writeSedFile(self.getRefWaveLength())
        else:
            warnings.warn(
                "No inspection of SED file for no PhoSim path.", category=UserWarning
            )

    def writeOpdInstFile(
        self, instFileDir, opdMetr, instSettingFile=None, instFileName="opd.inst"
    ):
        """Write the optical path difference (OPD) instance file.

        Parameters
        ----------
        instFileDir : str
            Directory to instance file.
        opdMetr : OpdMetrology
            OpdMetrology object.
        instSettingFile : str, optional
            Instance setting file. (the default is None.)
        instFileName : str, optional
            OPD instance file name. (the default is "opd.inst".)

        Returns
        -------
        str
            Instance file path.
        """

        # Instance file path
        instFilePath = os.path.join(instFileDir, instFileName)

        # Get the observation ID
        obsId = self.surveyParam["obsId"]

        # Get the filter ID in PhoSim
        aFilterId = self._getFilterIdInPhoSim()

        # Add the sky information
        boresight = self.surveyParam["boresight"]
        ra = boresight[0]
        dec = boresight[1]

        rot = self.surveyParam["rotAngInDeg"]

        # Write the default instance setting
        self.phoSimCommu.getOpdInstance(
            obsId, aFilterId, ra=ra, dec=dec, rot=rot, filePath=instFilePath
        )
        if instSettingFile is not None:
            self.phoSimCommu.writeToFile(instFilePath, sourceFile=instSettingFile)

        # Write the telescope accumulated degree of freedom (DOF)
        content = self.phoSimCommu.doDofPert(self.dofInUm)

        # Write the OPD source
        fieldX, fieldY = opdMetr.getFieldXY()
        for idx in range(len(fieldX)):
            content += self.phoSimCommu.generateOpd(
                idx, fieldX[idx], fieldY[idx], self.getRefWaveLength()
            )
        self.phoSimCommu.writeToFile(instFilePath, content=content)

        # Write the OPD SED file if necessary
        self._writeSedFileIfPhoSimDirSet()

        return instFilePath

    def writePertBaseOnConfigFile(
        self,
        pertCmdFileDir,
        seedNum=None,
        m1m3ForceError=0.05,
        saveResMapFig=False,
        pertCmdFileName="pert.cmd",
    ):
        """Write the perturbation command file based on the telescope
        configuration file.

        Parameters
        ----------
        pertCmdFileDir : str
            Directory to the pertubation command file.
        seedNum : int, optional
            Random seed number. If the value is not None, the M1M3 mirror
            will generate a random surface error. (the default is None.)
        m1m3ForceError : float, optional
            Ratio of actuator force error. (the default is 0.05.)
        saveResMapFig : bool, optional
            Save the mirror surface residue map or not. (the default is False.)
        pertCmdFileName : str, optional
            Perturbation command file name. (the default is "pert.cmd".)

        Returns
        -------
        str
            Perturbation command file path.
        """

        # Get the perturbation of subsystems
        content = ""
        if self.m1m3 is not None:
            m1ResFileName = "M1res.txt"
            m3ResFileName = "M3res.txt"
            m1m3ZcFileName = "M1M3zlist.txt"

            m1ResFilePath = os.path.join(pertCmdFileDir, m1ResFileName)
            m3ResFilePath = os.path.join(pertCmdFileDir, m3ResFileName)
            m1m3ZcFilePath = os.path.join(pertCmdFileDir, m1m3ZcFileName)

            content = self._addPertM1M3(
                m1ResFilePath,
                m3ResFilePath,
                m1m3ZcFilePath,
                content,
                m1m3ForceError,
                seedNum=seedNum,
            )

        if self.m2 is not None:
            m2ResFileName = "M2res.txt"
            m2ZcFileName = "M2zlist.txt"

            m2ResFilePath = os.path.join(pertCmdFileDir, m2ResFileName)
            m2ZcFilePath = os.path.join(pertCmdFileDir, m2ZcFileName)

            content = self._addPertM2(m2ResFilePath, m2ZcFilePath, content)

        if self.cam is not None:
            content = self._addPertCam(content)

        # Write the perturbation command to file
        pertCmdFilePath = os.path.join(pertCmdFileDir, pertCmdFileName)
        self.phoSimCommu.writeToFile(pertCmdFilePath, content=content, mode="w")

        # Save the mirror residue map if necessary
        if saveResMapFig:

            if self.m1m3 is not None:
                self._saveM1M3ResMapFig(m1ResFilePath, m3ResFilePath)

            if self.m2 is not None:
                self._saveM2ResMapFig(m2ResFilePath)

        return pertCmdFilePath

    def _addPertM1M3(
        self,
        m1ResFilePath,
        m3ResFilePath,
        m1m3ZcFilePath,
        content,
        m1m3ForceError,
        seedNum=None,
    ):
        """Add the perturbation of M1M3.

        Parameters
        ----------
        m1ResFilePath : str
            M1 residue file path.
        m3ResFilePath : str
            M3 residue file path.
        m1m3ZcFilePath : str
            M1M3 fitted zk file path.
        content : str
            Perturbation without M1M3.
        m1m3ForceError : float
            Ratio of actuator force error.
        seedNum : int, optional
            Random seed number. If the value is not None, the M1M3 mirror
            will generate a random surface error. (the default is None.)

        Returns
        -------
        str
            Perturbation with M1M3.
        """

        # Do the gravity correction
        zAngleInRad = self._getZenAngleInRad()
        printthzInM = self.m1m3.getPrintthz(zAngleInRad)

        # Add the surface error if necessary
        randSurfInM = None
        if seedNum is not None:
            randSurfInM = self.m1m3.genMirSurfRandErr(
                zAngleInRad, m1m3ForceError=m1m3ForceError, seedNum=seedNum
            )

        # Do the temperature correction
        m1m3TBulk = self._teleSettingFile.getSetting("m1m3TBulk")
        m1m3TxGrad = self._teleSettingFile.getSetting("m1m3TxGrad")
        m1m3TyGrad = self._teleSettingFile.getSetting("m1m3TyGrad")
        m1m3TzGrad = self._teleSettingFile.getSetting("m1m3TzGrad")
        m1m3TrGrad = self._teleSettingFile.getSetting("m1m3TrGrad")
        tempCorrInUm = self.m1m3.getTempCorr(
            m1m3TBulk, m1m3TxGrad, m1m3TyGrad, m1m3TzGrad, m1m3TrGrad
        )

        # Set the mirror surface in mm
        if randSurfInM is not None:
            mirrorSurfInUm = (printthzInM + randSurfInM) * 1e6 + tempCorrInUm
        else:
            mirrorSurfInUm = printthzInM * 1e6 + tempCorrInUm
        self.m1m3.setSurfAlongZ(mirrorSurfInUm)

        resFile = [m1ResFilePath, m3ResFilePath]
        surfaceGridN = self.getSurfGridN()
        self.m1m3.writeMirZkAndGridResInZemax(
            resFile=resFile,
            surfaceGridN=surfaceGridN,
            writeZcInMnToFilePath=m1m3ZcFilePath,
        )

        # Get the Zk in mm
        zkInMm = np.loadtxt(m1m3ZcFilePath)

        # Do the surface perturbation
        surfList = [SurfaceType.M1, SurfaceType.M3]
        surfIdList = []

        contentWithPert = content
        for ii in range(len(surfList)):
            surf = surfList[ii]
            surfId = self.phoSimCommu.getSurfaceId(surf)
            contentWithPert += self.phoSimCommu.doSurfPert(surfId, zkInMm)

            # Collect the surface ID
            surfIdList.append(surfId)

            # Do the surface residue map perturbation
            contentWithPert += self.phoSimCommu.doSurfMapPert(surfId, resFile[ii], 1)

        # Do the surface linkage
        contentWithPert += self.phoSimCommu.doSurfLink(surfIdList[1], surfIdList[0])

        return contentWithPert

    def _getZenAngleInRad(self):
        """Get the zenith angle in radian.

        Returns
        -------
        float
            Zenith angle in radian.
        """

        zAngleInDeg = self.surveyParam["zAngleInDeg"]
        zAngleInRad = np.deg2rad(zAngleInDeg)

        return zAngleInRad

    def _addPertM2(self, m2ResFilePath, m2ZcFilePath, content):
        """Add the perturbation of M2.

        Parameters
        ----------
        m2ResFilePath : str
            M2 residue file path.
        m2ZcFilePath : str
            M2 fitted zk file path.
        content : str
            Perturbation without M2.

        Returns
        -------
        str
            Perturbation with M2.
        """

        # Do the gravity correction
        zAngleInRad = self._getZenAngleInRad()
        printthzInUm = self.m2.getPrintthz(zAngleInRad)

        # Do the temperature correction
        m2TzGrad = self._teleSettingFile.getSetting("m2TzGrad")
        m2TrGrad = self._teleSettingFile.getSetting("m2TrGrad")
        tempCorrInUm = self.m2.getTempCorr(m2TzGrad, m2TrGrad)

        # Set the mirror surface in mm
        mirrorSurfInUm = printthzInUm + tempCorrInUm
        self.m2.setSurfAlongZ(mirrorSurfInUm)

        surfaceGridN = self.getSurfGridN()
        self.m2.writeMirZkAndGridResInZemax(
            resFile=m2ResFilePath,
            surfaceGridN=surfaceGridN,
            writeZcInMnToFilePath=m2ZcFilePath,
        )

        # Get the Zk in mm
        zkInMm = np.loadtxt(m2ZcFilePath)

        # Do the surface perturbation
        surfId = self.phoSimCommu.getSurfaceId(SurfaceType.M2)

        contentWithPert = content
        contentWithPert += self.phoSimCommu.doSurfPert(surfId, zkInMm)

        # Do the surface residue map perturbation
        contentWithPert += self.phoSimCommu.doSurfMapPert(surfId, m2ResFilePath, 1)

        return contentWithPert

    def _addPertCam(self, content):
        """Add the perturbation of camera.

        Parameters
        ----------
        content : str
            Perturbation without camera.

        Returns
        -------
        str
            Perturbation with camera.
        """

        # Set the camera rotation angle
        rotAngInDeg = self.surveyParam["rotAngInDeg"]
        self.cam.setRotAngInDeg(rotAngInDeg)

        # Set the temperature information
        tempInDegC = self._teleSettingFile.getSetting("camTB")
        self.cam.setBodyTempInDegC(tempInDegC)

        # Add the perturbation of camera
        zAngleInRad = self._getZenAngleInRad()
        contentWithPert = content
        for distType in CamDistType:
            # Get the surface ID
            surfaceType = self._getPhoSimCamSurf(distType.name)
            surfId = self.phoSimCommu.getSurfaceId(surfaceType)

            # Do the perturbation
            zkInMm = self.cam.getCamDistortionInMm(zAngleInRad, distType)
            contentWithPert += self.phoSimCommu.doSurfPert(surfId, zkInMm)

        return contentWithPert

    def _saveM1M3ResMapFig(self, m1ResFilePath, m3ResFilePath):
        """Save the figure of M1M3 residue map.

        Parameters
        ----------
        m1ResFilePath : str
            M1 residue file path.
        m3ResFilePath : str
            M3 residue file path.
        """

        resFile = [m1ResFilePath, m3ResFilePath]

        writeToResMapFilePath1 = self._getImgPathFromDataFilePath(m1ResFilePath)
        writeToResMapFilePath3 = self._getImgPathFromDataFilePath(m3ResFilePath)
        writeToResMapFilePath = [writeToResMapFilePath1, writeToResMapFilePath3]

        self.m1m3.showMirResMap(
            resFile=resFile, writeToResMapFilePath=writeToResMapFilePath
        )

    def _getImgPathFromDataFilePath(self, dataFilePath, imgType=".png"):
        """Get the image path from the data file path.

        Parameters
        ----------
        dataFilePath : str
            Data file path.
        imgType : str, optional
            Image type (the default is ".png".)

        Returns
        -------
        str
            Image path.
        """

        imgPath = os.path.splitext(dataFilePath)[0] + imgType

        return imgPath

    def _saveM2ResMapFig(self, m2ResFilePath):
        """Save the figure of M2 residue map.

        Parameters
        ----------
        m2ResFilePath : str
            M2 residue file path.
        """

        writeToResMapFilePath = self._getImgPathFromDataFilePath(m2ResFilePath)

        self.m2.showMirResMap(
            resFile=m2ResFilePath, writeToResMapFilePath=writeToResMapFilePath
        )

    def _getPhoSimCamSurf(self, camSurfName):
        """Get the camera surface used in PhoSim.

        Parameters
        ----------
        camSurfName : str
            Camera surface name (e.g. L1S1zer).

        Returns
        -------
        enum 'SurfaceType'
            Camera surface type.

        Raises
        ------
        ValueError
            Can not get the camera surface name.
        """

        # Get the camera surface name
        m = re.match(r"(\AL\d)S(\d)zer", camSurfName)
        if m is None:
            raise ValueError("Cannot get the camera surface name: %s." % camSurfName)

        # Get the surface name used in PhoSim
        camPhoFaceDict = {"1": "F", "2": "B"}
        surfName = m.groups()[0] + camPhoFaceDict[m.groups()[1]]

        return mapSurfNameToEnum(surfName)


if __name__ == "__main__":
    pass
