#!/usr/bin/env python

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
import shutil
import logging

import astropy.io.ascii

import numpy as np

from lsst.afw.cameraGeom import DetectorType

from lsst.daf import butler as dafButler

from lsst.ts.wep.Utility import CamType, FilterType, runProgram

from lsst.ts.ofc import OFC, OFCData

from lsst.ts.phosim.Utility import getPhoSimPath, getAoclcOutputPath, getCamera
from lsst.ts.phosim.telescope.TeleFacade import TeleFacade
from lsst.ts.phosim.PhosimCmpt import PhosimCmpt
from lsst.ts.phosim.PlotUtil import plotFwhmOfIters
from lsst.ts.phosim.SkySim import SkySim
from lsst.ts.phosim.OpdMetrology import OpdMetrology
from lsst.ts.phosim.SensorWavefrontError import SensorWavefrontError


class CloseLoopTask(object):
    def __init__(self):
        """Initialization of the close-loop task class to run the simulation
        with PhoSim."""

        self.log = logging.getLogger(type(self).__name__)

        # Sky simulator
        self.skySim = None

        # OFC calculator
        self.ofcCalc = None

        # PhoSim component
        self.phosimCmpt = None

        # Use the amplifier image
        self.useAmp = False

        # Use the eimage
        self.useEimg = False

        # offset between the specified obsId and the assigned visitId, e.g.
        # obsId = 9006000, visitId = 4021123106000.
        self.visitIdOffset = 4021114100000

        # Ra/Dec/RotAng coordinates used in the simulation.
        self.boresightRa = None
        self.boresightDec = None
        self.boresightRotAng = None

    def configSkySim(self, instName, pathSkyFile="", starMag=15):
        """Configure the sky simulator.

        If the path of sky file is not provided, The defult OPD field positions
        will be used.

        OPD: Optical path difference.

        Parameters
        ----------
        instName : enum 'InstName' in lsst.ts.ofc.Utility
            Instrument name.
        pathSkyFile : str, optional
            Path to the sky file. (the default is "".)
        starMag : float, optional
            Default star magnitude if there is no sky file used. This is to
            pretend there are the stars at OPD field positions. (the default is
            15.)

        Raises
        ------
        ValueError
            This instrument name is not supported.
        """

        self.skySim = SkySim()
        if pathSkyFile == "":
            self._setSkySimBasedOnOpdFieldPos(instName, starMag)
        else:
            absSkyFilePath = os.path.abspath(pathSkyFile)
            self.skySim.addStarByFile(absSkyFilePath)

    def _setSkySimBasedOnOpdFieldPos(self, instName, starMag):
        """Set the sky simulator based on the OPD field positions.

        OPD: Optical path difference.

        Parameters
        ----------
        instName : enum 'InstName' in lsst.ts.ofc.Utility
            Instrument name.
        starMag : float
            Star magnitude. This is to pretend there are the stars at OPD field
            positions.

        Raises
        ------
        ValueError
            This instrument name is not supported.
        """

        self.log.info(
            "Use the default OPD field positions to be star positions."
            f"The star magnitude is chosen to be {starMag}."
        )

        opdMetr = OpdMetrology()
        if instName in ("comcam", "lsstfam"):
            opdMetr.setWgtAndFieldXyOfGQ(instName)
        elif instName == "lsst":
            fieldX, fieldY = opdMetr.getDefaultLsstWfsGQ()
            opdMetr.setFieldXYinDeg(fieldX, fieldY)
        else:
            raise ValueError(f"This instrument name ({instName}) is not supported.")

        starId = 0
        raInDegList, declInDegList = opdMetr.getFieldXY()
        for raInDeg, declInDeg in zip(raInDegList, declInDegList):
            # It is noted that the field position might be < 0. But it is
            # not the same case for ra (0 <= ra <= 360).
            if raInDeg < 0:
                raInDeg += 360.0
            self.skySim.addStarByRaDecInDeg(starId, raInDeg, declInDeg, starMag)
            starId += 1

    def configOfcCalc(self, instName):
        """Configure the OFC calculator.

        OFC: Optical feedback calculator.

        Parameters
        ----------
        instName : enum 'InstName' in lsst.ts.ofc.Utility
            Instrument name.
        """

        self.ofcCalc = OFC(OFCData(instName))

    def configPhosimCmpt(
        self,
        filterType,
        rotAngInDeg,
        m1m3ForceError,
        numPro,
        boresight=[0, 0],
        zAngleInDeg=27.0912,
        seedNum=6,
    ):
        """Configure the component to use PhoSim.

        Parameters
        ----------
        filterType : enum 'FilterType' in lsst.ts.wep.Utility
            Filter type.
        rotAngInDeg : float
            The camera rotation angle in degree (-90 to 90).
        m1m3ForceError : float
            Ratio of M1M3 actuator force error between 0 and 1.
        numPro : int
            Number of processor to run PhoSim.
        boresight : list[float], optional
            Boresight [ra, dec] in degree. (the default is [0, 0].)
        zAngleInDeg : float, optional
            Zenith angle in degree of telescope. (the default is 27.0912.)
        seedNum : int, optional
            Seed number for the M1M3 mirror surface purturbation.

        Returns
        -------
        PhosimCmpt
            PhoSim component.
        """

        self.boresightRa = boresight[0]
        self.boresightDec = boresight[1]
        self.boresightRotAng = rotAngInDeg

        # Set the Telescope facade class
        tele = TeleFacade()
        tele.addSubSys(addCam=True, addM1M3=True, addM2=True)

        phosimDir = getPhoSimPath()
        tele.setPhoSimDir(phosimDir)

        # Prepare the phosim component
        self.phosimCmpt = PhosimCmpt(tele)

        # Set the telescope survey parameters
        self.phosimCmpt.setSurveyParam(
            filterType=filterType,
            boresight=tuple(boresight),
            zAngleInDeg=zAngleInDeg,
            rotAngInDeg=rotAngInDeg,
        )

        # Set the M1M3 force error
        self.phosimCmpt.setM1M3ForceError(m1m3ForceError)

        # Update the number of processor if necessary
        if numPro > 1:
            settingFile = self.phosimCmpt.getSettingFile()
            settingFile.updateSetting("numPro", numPro)

        # Set the seed number for M1M3 surface
        self.phosimCmpt.setSeedNum(seedNum)

        return self.phosimCmpt

    def getSkySim(self):
        """Get the sky simulator.

        Returns
        -------
        SkySim or None
            Sky simulator. None if not configured yet.
        """

        return self.skySim

    def getOfcCalc(self):
        """Get the OFC calculator.

        OFC: Optical feedback control.

        Returns
        -------
        OFCCalculation child (e.g. OFCCalculationOfComCam) or None
            Concrete child class of OFCCalculation class. None if not
            configured yet.
        """

        return self.ofcCalc

    def getPhosimCmpt(self):
        """Get the PhoSim component.

        Returns
        -------
        PhosimCmpt or None
            PhoSim component. None if not configured yet.
        """

        return self.phosimCmpt

    def assignImgType(self, useEimg):
        """Assign the image type.

        OPD: Optical path difference.

        Parameters
        ----------
        useEimg : bool or None
            Use the eimage or not. Put None if only use the OPD.
        """

        if useEimg is None:
            self.useEimg = False
            self.useAmp = False
        elif useEimg is True:
            self.useEimg = True
            self.useAmp = False
        else:
            self.useEimg = False
            self.useAmp = True

    def useCcdImg(self):
        """Use the CCD images (amplifier or eimage) or not.

        CCD: Charge-coupled device.

        Returns
        -------
        bool
            Return True if the CCD image is used.
        """

        if self.useAmp or self.useEimg:
            return True
        else:
            return False

    def runOpd(
        self,
        inst,
        filterTypeName,
        rotCamInDeg,
        m1m3ForceError,
        numPro,
        iterNum,
        baseOutputDir,
        doErsDirCont,
    ):
        """Run the simulation of OPD.

        OPD: Optical path difference.

        Parameters
        ----------
        inst : str
            Instrument to use: comcam or lsstfam.
        filterTypeName : str
            Filter type name: ref, u, g, r, i, z, or y.
        rotCamInDeg : float
            The camera rotation angle in degree (-90 to 90).
        m1m3ForceError : float
            Ratio of M1M3 actuator force error between 0 and 1.
        numPro : int
            Number of processor to run PhoSim.
        iterNum : int
            Number of closed-loop iteration.
        baseOutputDir : str
            Base output directory.
        doErsDirCont : bool
            Do the erase of the content of base output directory or not.
        """

        # Check the input arguments
        camType, instName = self.getCamTypeAndInstName(inst)
        filterType = self.getFilterType(filterTypeName)
        baseOutputDir = self.checkAndCreateBaseOutputDir(baseOutputDir)

        if doErsDirCont:
            self.eraseDirectoryContent(baseOutputDir)

        # Configure the components
        self.configOfcCalc(instName)
        self.configPhosimCmpt(filterType, rotCamInDeg, m1m3ForceError, numPro)

        butlerRootPath = os.path.join(baseOutputDir, "phosimData")
        # Run the simulation
        self._runSim(
            camType=camType,
            instName=instName,
            filterType=filterTypeName,
            rotCamInDeg=rotCamInDeg,
            iterNum=iterNum,
            baseOutputDir=baseOutputDir,
            butlerRootPath=butlerRootPath,
        )

    def getCamTypeAndInstName(self, inst):
        """Get the camera type and instrument name.

        Parameters
        ----------
        inst : str
            Instrument to use: comcam or lsstfam.

        Returns
        -------
        camType : enum 'CamType' in lsst.ts.wep.Utility
            Camera type.
        instName : enum 'InstName' in lsst.ts.ofc.Utility
            Instrument name.

        Raises
        ------
        ValueError
            This instrument is not supported.
        """

        if inst == "comcam":
            return CamType.ComCam, "comcam"
        elif inst == "lsstfam":
            return CamType.LsstFamCam, "lsstfam"
        else:
            raise ValueError(f"This instrument ({inst}) is not supported.")

    def getFilterType(self, filterTypeName):
        """Get the filter type.

        Parameters
        ----------
        filterTypeName : str
            Filter type name: ref, u, g, r, i, z, or y.

        Returns
        -------
        filterType : enum 'FilterType' in lsst.ts.wep.Utility
            Filter type.

        Raises
        ------
        ValueError
            This filter type is not supported.
        """

        if filterTypeName in {"", "ref"}:
            return FilterType.REF
        elif filterTypeName == "u":
            return FilterType.U
        elif filterTypeName == "g":
            return FilterType.G
        elif filterTypeName == "r":
            return FilterType.R
        elif filterTypeName == "i":
            return FilterType.I
        elif filterTypeName == "z":
            return FilterType.Z
        elif filterTypeName == "y":
            return FilterType.Y
        else:
            raise ValueError(f"This filter type ({filterTypeName}) is not supported.")

    def checkAndCreateBaseOutputDir(self, baseOutputDir):
        """Check and create the base output directory.

        This function will create the directory if it does not exist.

        Parameters
        ----------
        baseOutputDir : str
            Base output directory.

        Returns
        -------
        str
            Base output directory.
        """

        if baseOutputDir == "":
            outputDir = getAoclcOutputPath()
        else:
            outputDir = baseOutputDir
        os.makedirs(outputDir, exist_ok=True)

        return outputDir

    def eraseDirectoryContent(self, targetDir):
        """Erase the directory content.

        Parameters
        ----------
        targetDir : str
            Target directory.
        """

        for theFile in os.listdir(targetDir):
            filePath = os.path.join(targetDir, theFile)
            if os.path.isfile(filePath):
                os.unlink(filePath)
            elif os.path.isdir(filePath):
                shutil.rmtree(filePath)

    def _runSim(
        self,
        camType,
        instName,
        filterType,
        rotCamInDeg,
        iterNum,
        baseOutputDir,
        butlerRootPath,
    ):
        """Run the simulation.

        Parameters
        ----------
        camType : enum 'CamType' in lsst.ts.wep.Utility
            Camera type.
        instName : enum 'InstName' in lsst.ts.ofc.Utility
            Instrument name.
        filterType : str
            Filter type.
        rotCamInDeg : float
            The camera rotation angle in degree (-90 to 90).
        iterNum : int
            Number of closed-loop iteration.
        baseOutputDir : str
            Base output directory.
        butlerRootPath : str
            Path to the butler gen 3 repository.
        """

        # Set the telescope state to be the same as the OFC
        state0 = self.ofcCalc.ofc_controller.aggregated_state
        self.phosimCmpt.setDofInUm(state0)

        # Get the list of referenced sensor name (field positions)
        refSensorNameList = self.getSensorNameListOfFields(instName)
        refSensorIdList = self.getSensorIdListOfFields(instName)

        # Common file and directory names
        opdZkFileName = "opd.zer"
        opdPssnFileName = "PSSN.txt"
        outputDirName = "pert"
        outputImgDirName = "img"
        iterDefaultDirName = "iter"
        dofInUmFileName = "dofPertInNextIter.mat"
        fwhmItersFileName = "fwhmIters.png"

        # Specific file names to the amplifier/eimage
        wfsZkFileName = "wfs.zer"

        # Do the iteration
        obsId = 9006000
        for iterCount in range(iterNum):

            # Set the observation Id
            self.phosimCmpt.setSurveyParam(obsId=obsId)

            # The iteration directory
            iterDirName = "%s%d" % (iterDefaultDirName, iterCount)

            # Set the output directory
            outputDir = os.path.join(baseOutputDir, iterDirName, outputDirName)
            self.phosimCmpt.setOutputDir(outputDir)

            # Set the output image directory
            outputImgDir = os.path.join(baseOutputDir, iterDirName, outputImgDirName)
            self.phosimCmpt.setOutputImgDir(outputImgDir)

            # Generate the OPD image
            argString = self.phosimCmpt.getOpdArgsAndFilesForPhoSim(instName)
            self.log.info(f"PHOSIM OPD ARGSTRING: {argString}")

            self.phosimCmpt.runPhoSim(argString)

            # Analyze the OPD data
            # Rotate OPD in the reversed direction of camera
            self.phosimCmpt.analyzeOpdData(
                instName,
                zkFileName=opdZkFileName,
                rotOpdInDeg=-rotCamInDeg,
                pssnFileName=opdPssnFileName,
            )

            # Get the PSSN from file
            pssn = self.phosimCmpt.getOpdPssnFromFile(opdPssnFileName)
            self.log.info("Calculated PSSN is %s." % pssn)

            # Get the GQ effective FWHM from file
            gqEffFwhm = self.phosimCmpt.getOpdGqEffFwhmFromFile(opdPssnFileName)
            self.log.info("GQ effective FWHM is %.4f." % gqEffFwhm)

            # Set the FWHM data
            fwhm, sensor_id = self.phosimCmpt.getListOfFwhmSensorData(
                opdPssnFileName, refSensorIdList
            )

            self.ofcCalc.set_fwhm_data(fwhm, sensor_id)

            # Generate the sky images and calculate the wavefront error
            if self.useCcdImg():
                listOfWfErr = self._calcWfErrFromImg(
                    obsId, butlerRootPath=butlerRootPath, instName=instName, snap=0
                )
            else:
                # Simulate to get the wavefront sensor data from WEP
                listOfWfErr = self.phosimCmpt.mapOpdDataToListOfWfErr(
                    opdZkFileName, refSensorIdList
                )

            # Record the wavefront error with the same order as OPD for the
            # comparison
            if self.useCcdImg():
                self.phosimCmpt.reorderAndSaveWfErrFile(
                    listOfWfErr,
                    refSensorNameList,
                    getCamera(instName),
                    zkFileName=wfsZkFileName,
                )

            # Calculate the DOF
            wfe = np.array(
                [sensor_wfe.getAnnularZernikePoly() for sensor_wfe in listOfWfErr]
            )
            sensor_ids = np.array(
                [sensor_wfe.getSensorId() for sensor_wfe in listOfWfErr]
            )

            self.ofcCalc.calculate_corrections(
                wfe=wfe,
                field_idx=sensor_ids,
                filter_name=str(filterType),
                gain=-1,
                rot=rotCamInDeg,
            )

            # Set the new aggregated DOF to phosimCmpt
            dofInUm = self.ofcCalc.ofc_controller.aggregated_state
            self.phosimCmpt.setDofInUm(dofInUm)

            # Save the DOF file
            self.phosimCmpt.saveDofInUmFileForNextIter(
                dofInUm, dofInUmFileName=dofInUmFileName
            )

            # Add the observation ID by 10 for the next iteration
            obsId += 10

        # Summarize the FWHM
        pssnFiles = [
            os.path.join(
                baseOutputDir,
                "%s%d" % (iterDefaultDirName, num),
                outputImgDirName,
                opdPssnFileName,
            )
            for num in range(iterNum)
        ]
        saveToFilePath = os.path.join(baseOutputDir, fwhmItersFileName)
        plotFwhmOfIters(pssnFiles, saveToFilePath=saveToFilePath)

    def getSensorNameListOfFields(self, instName):
        """Get the list of sensor name of fields.

        The list will be sorted based on the field index.

        Parameters
        ----------
        instName : enum 'InstName' in lsst.ts.ofc.Utility
            Instrument name.

        Returns
        -------
        list[str]
            List of sensor name.

        Raises
        ------
        ValueError
            This instrument name is not supported.
        """

        camera = getCamera(instName)
        detectorType = (
            DetectorType.WAVEFRONT if instName == "lsst" else DetectorType.SCIENCE
        )
        return [
            detector.getName()
            for detector in camera
            if detector.getType() == detectorType
        ]

    def getSensorIdListOfFields(self, instName):
        """Get the list of sensor ids of fields.

        The list will be sorted based on the field index.

        Parameters
        ----------
        instName : enum 'InstName' in lsst.ts.ofc.Utility
            Instrument name.

        Returns
        -------
        list[int]
            List of sensor ids.

        Raises
        ------
        ValueError
            This instrument name is not supported.
        """

        camera = getCamera(instName)

        detectorType = (
            DetectorType.WAVEFRONT if instName == "lsst" else DetectorType.SCIENCE
        )
        return [
            detector.getId()
            for detector in camera
            if detector.getType() == detectorType
        ]

    def _calcWfErrFromImg(self, obsId, butlerRootPath, instName, snap=0, simSeed=1000):
        """Calculate the wavefront error from the images generated by PhoSim.

        Parameters
        ----------
        obsId : int
            Observation ID used in PhoSim.
        butlerRootPath : str
            Path to the butler repository.
        snap : int, optional
            Snap. (the default is 0.)
        simSeed : int, optional
            Simulation seed numeber. (the default is 1000.)

        Returns
        -------
        list[lsst.ts.wep.ctrlIntf.SensorWavefrontError]
            List of SensorWavefrontError object.
        """

        # Assign the entra- and intra-focal observation Id
        extraObsId = obsId + 1
        intraObsId = obsId + 2

        # Generate the defocal images
        argStringList = self.phosimCmpt.getPistonCamStarArgsAndFilesForPhoSim(
            extraObsId,
            intraObsId,
            self.skySim,
            simSeed=simSeed,
            cmdSettingFileName="starDefault.cmd",
            instSettingFileName="starSingleExp.inst",
        )
        for argString in argStringList:
            self.log.info(f"PHOSIM CCD ARGSTRING: {argString}")
            self.phosimCmpt.runPhoSim(argString)

        # Repackage the images based on the image type
        self.phosimCmpt.repackagePistonCamImgs(
            instName=instName if instName == "comcam" else "lsst", isEimg=self.useEimg
        )

        # Ingest images into butler gen3
        self.ingestData(butlerRootPath=butlerRootPath, instName=instName)

        listOfWfErr = self.runWep(extraObsId, intraObsId, butlerRootPath, instName)

        return listOfWfErr

    def runWep(self, extraObsId, intraObsId, butlerRootPath, instName):
        """Run wavefront estimation pipeline task.

        Parameters
        ----------
        extraObsId : `int`
            Extra observation id.
        intraObsId : `int`
            Intra observation id.
        butlerRootPath : `str`
            Path to the butler gen3 repos.
        instName : `str`
            Instrument name.

        Returns
        -------
        listOfWfErr : `list` of `SensorWavefrontError`
            List of SensorWavefrontError with the results of the wavefront
            estimation pipeline for each sensor.
        """

        butlerInstName = "ComCam" if instName == "comcam" else "Cam"
        pipelineYaml = f"{instName}Pipeline.yaml"
        pipelineYamlPath = os.path.join(butlerRootPath, pipelineYaml)

        butler = dafButler.Butler(butlerRootPath)

        if f"LSST{butlerInstName}/calib" not in butler.registry.queryCollections():

            self.log.info("Ingesting curated calibrations.")

            runProgram(
                f"butler write-curated-calibrations {butlerRootPath} lsst.obs.lsst.Lsst{butlerInstName}"
            )

        self.writeWepConfiguration(instName, pipelineYamlPath)

        runProgram(
            f"pipetask run -b {butlerRootPath} "
            f"-i refcats,LSST{butlerInstName}/raw/all,LSST{butlerInstName}/calib "
            f"--instrument lsst.obs.lsst.Lsst{butlerInstName} "
            f"--register-dataset-types --output-run ts_phosim_{extraObsId} -p {pipelineYamlPath} -d "
            f'"exposure IN ({self.visitIdOffset+extraObsId}, {self.visitIdOffset+intraObsId})" -j 2'
        )

        # Need to redefine butler because the database changed.
        butler = dafButler.Butler(butlerRootPath)

        datasetRefs = butler.registry.queryDatasets(
            datasetType="zernikeEstimateAvg", collections=[f"ts_phosim_{extraObsId}"]
        )

        listOfWfErr = []

        for dataset in datasetRefs:
            dataId = {
                "instrument": dataset.dataId["instrument"],
                "detector": dataset.dataId["detector"],
                "exposure": dataset.dataId["exposure"],
            }

            zerCoeff = butler.get(
                "zernikeEstimateAvg",
                dataId=dataId,
                collections=[f"ts_phosim_{extraObsId}"],
            )

            sensorWavefrontData = SensorWavefrontError()
            sensorWavefrontData.setSensorId(dataset.dataId["detector"])
            sensorWavefrontData.setAnnularZernikePoly(zerCoeff)

            listOfWfErr.append(sensorWavefrontData)

        return listOfWfErr

    def writeWepConfiguration(self, instName, pipelineYamlPath):
        """Write wavefront estimation pipeline task configuration.

        Parameters
        ----------
        instName: `str`
            Name of the instrument this configuration is intended for.
        pipelineYamlPath: `str`
            Path where the pipeline task configuration yaml file
            should be saved.
        """

        butlerInstName = "ComCam" if instName == "comcam" else "Cam"

        with open(pipelineYamlPath, "w") as fp:
            fp.write(
                f"""# This yaml file is used to define the tasks and configuration of
# a Gen 3 pipeline used for testing in ts_wep.
description: wep basic processing test pipeline
# Here we specify the corresponding instrument for the data we
# will be using.
instrument: lsst.obs.lsst.Lsst{butlerInstName}
# Then we can specify each task in our pipeline by a name
# and then specify the class name corresponding to that task
tasks:
  isr:
    class: lsst.ip.isr.isrTask.IsrTask
    # Below we specify the configuration settings we want to use
    # when running the task in this pipeline. Since our data doesn't
    # include bias or flats we only want to use doApplyGains and
    # doOverscan in our isr task.
    config:
      connections.outputExposure: 'postISRCCD'
      doBias: False
      doVariance: False
      doLinearize: False
      doCrosstalk: False
      doDefect: False
      doNanMasking: False
      doInterpolate: False
      doBrighterFatter: False
      doDark: False
      doFlat: False
      doApplyGains: True
      doFringe: False
      doOverscan: True
  generateDonutCatalogOnlineTask:
    class: lsst.ts.wep.task.GenerateDonutCatalogOnlineTask.GenerateDonutCatalogOnlineTask
    # Here we specify the configurations for pointing that we added into the class
    # GenerateDonutCatalogOnlineTaskConfig.
    config:
      boresightRa: {self.boresightRa}
      boresightDec: {self.boresightDec}
      boresightRotAng: {self.boresightRotAng}
  estimateZernikesFamTask:
    class: lsst.ts.wep.task.EstimateZernikesFamTask.EstimateZernikesFamTask
    config:
      # And here we specify the configuration settings originally defined in
      # EstimateZernikesFamTaskConfig.
      donutTemplateSize: 160
      donutStampSize: 160
      initialCutoutPadding: 40
"""
            )

    def ingestData(self, butlerRootPath, instName):
        """Ingest data into a gen3 data Butler.

        Parameters
        ----------
        butlerRootPath : str
            Path to the butler repository.
        instName : str
            Instrument name.
        """
        outputImgDir = self.phosimCmpt.getOutputImgDir()

        intraRawExpDir = os.path.join(
            outputImgDir, self.phosimCmpt.getIntraFocalDirName()
        )

        extraRawExpDir = os.path.join(
            outputImgDir, self.phosimCmpt.getExtraFocalDirName()
        )

        runProgram(f"butler ingest-raws {butlerRootPath} {intraRawExpDir}")
        runProgram(f"butler ingest-raws {butlerRootPath} {extraRawExpDir}")
        if instName == "comcam":
            runProgram(
                f"butler define-visits {butlerRootPath} lsst.obs.lsst.LsstComCam"
            )
        else:
            runProgram(f"butler define-visits {butlerRootPath} lsst.obs.lsst.LsstCam")

    def runImg(
        self,
        inst,
        filterTypeName,
        rotCamInDeg,
        m1m3ForceError,
        numPro,
        iterNum,
        baseOutputDir,
        doErsDirCont,
        boresight,
        pathSkyFile,
        useEimg,
    ):
        """Run the simulation of image.

        Parameters
        ----------
        inst : str
            Instrument to use: comcam or lsstfam.
        filterTypeName : str
            Filter type name: ref, u, g, r, i, z, or y.
        rotCamInDeg : float
            The camera rotation angle in degree (-90 to 90).
        m1m3ForceError : float
            Ratio of M1M3 actuator force error between 0 and 1.
        numPro : int
            Number of processor to run PhoSim.
        iterNum : int
            Number of closed-loop iteration.
        baseOutputDir : str
            Base output directory.
        doErsDirCont : bool
            Do the erase of the content of base output directory or not.
        boresight : list[float]
            Boresight [ra, dec] in degree.
        pathSkyFile : str
            Path to the sky file.
        useEimg : bool
            Use the eimage or not.
        """

        # Check the input arguments
        camType, instName = self.getCamTypeAndInstName(inst)
        filterType = self.getFilterType(filterTypeName)
        baseOutputDir = self.checkAndCreateBaseOutputDir(baseOutputDir)

        if doErsDirCont:
            self.eraseDirectoryContent(baseOutputDir)

        self.checkBoresight(boresight, pathSkyFile)

        self.assignImgType(useEimg)

        # Configure the components
        self.configSkySim(instName, pathSkyFile=pathSkyFile, starMag=15)

        # If pathSkyFile using default OPD positions write this to disk
        # so that the Butler can load it later
        if pathSkyFile == "":
            pathSkyFile = os.path.join(baseOutputDir, "sky_info.txt")
            self.skySim.exportSkyToFile(pathSkyFile)
            self.log.info(f"Wrote new sky file to {pathSkyFile}.")

        pathIsrDir = self.createIsrDir(baseOutputDir)

        self.configOfcCalc(instName)
        self.configPhosimCmpt(
            filterType, rotCamInDeg, m1m3ForceError, numPro, boresight=boresight
        )

        # generate bluter gen3 repo if needed
        butlerRootPath = os.path.join(baseOutputDir, "phosimData")
        if self.useCcdImg():
            self.generateButler(butlerRootPath, instName)
            self.generateRefCatalog(
                instName=instName,
                butlerRootPath=butlerRootPath,
                pathSkyFile=pathSkyFile,
            )

        self.phosimCmpt.tele.setInstName(camType)

        # Run the simulation
        self._runSim(
            camType=camType,
            instName=instName,
            filterType=filterTypeName,
            rotCamInDeg=rotCamInDeg,
            iterNum=iterNum,
            baseOutputDir=baseOutputDir,
            butlerRootPath=butlerRootPath,
        )

    def checkBoresight(self, boresight, pathSkyFile):
        """Check the boresight.

        Parameters
        ----------
        boresight : list[float]
            Boresight [ra, dec] in degree.
        pathSkyFile : str
            Path to the sky file.

        Raises
        ------
        ValueError
            The boresight should be [0, 0] if no sky file assigned.
        ValueError
            The right ascension (RA) should be in [0, 360].
        ValueError
            The declination (Dec) should be in [-90, 90].
        """

        if boresight != [0, 0] and pathSkyFile == "":
            raise ValueError("The boresight should be [0, 0] if no sky file assigned.")

        ra, dec = boresight
        if ra < 0 or ra > 360:
            raise ValueError("The right ascension (RA) should be in [0, 360].")

        if dec < -90 or dec > 90:
            raise ValueError("The declination (Dec) should be in [-90, 90].")

    def createIsrDir(self, baseOutputDir, isrDirName="input"):
        """Create the ISR directory.

        ISR: Instrument signature removal.

        Parameters
        ----------
        baseOutputDir : str
            Base output directory.
        isrDirName : str, optional
            Name of the ISR directory. (the default is "input".)

        Returns
        -------
        str
            Path of the ISR directory.
        """

        isrDir = os.path.join(baseOutputDir, isrDirName)
        os.makedirs(isrDir, exist_ok=True)

        return isrDir

    def makeCalibs(self, instName, baseOutputDir, fakeFlatDirName="fake_flats"):
        """Make the calibration products.

        Only suppot the fake flat images at this moment.

        Parameters
        ----------
        instName : enum 'InstName' in lsst.ts.ofc.Utility
            Instrument name.
        baseOutputDir : str
            Base output directory.
        fakeFlatDirName : str, optional
            Directory name of the fake flat images. (the default is
            "fake_flats".)

        Returns
        -------
        str
            Path to the directory of fake flat images.
        """

        fakeFlatDir = os.path.join(baseOutputDir, fakeFlatDirName)
        os.makedirs(fakeFlatDir, exist_ok=True)

        justWfs = False
        detectors = ""
        if instName == "lsst":
            justWfs = True
        else:
            sensorNameList = self.getSensorNameListOfFields(instName)
            detectors = " ".join(sensorNameList)

        self._genFakeFlat(fakeFlatDir, justWfs=justWfs, detectors=detectors)

        return fakeFlatDir

    def _genFakeFlat(self, fakeFlatDir, justWfs=False, detectors=""):
        """Generate the fake flat images.

        Parameters
        ----------
        fakeFlatDir : str
            Directory to make the fake flat images.
        justWfs : bool, optional
            Just the wavefront sensor (WFS). (the default is False.)
        detectors : str, optional
            Detectors (e.g. R22_S11 R22_S10). (the default is "".)
        """

        # Get the current path
        currWorkDir = os.getcwd()

        # Enter the directory to make the fake flat images
        os.chdir(fakeFlatDir)

        # Make the fake flat images
        command = "makeGainImages.py"
        if justWfs:
            argstring = "--just_wfs"
        else:
            argstring = f"--detector_list {detectors}"
        runProgram(command, argstring=argstring)

        # Back to the current path
        os.chdir(currWorkDir)

    def generateButler(self, butlerRootPath, instName):
        """Generate butler gen3.

        Parameters
        ----------
        butlerRootPath: `str`
            Path to where the butler repository should be created.
        instName: `str`
            Name of the instrument.
        """

        self.log.info(f"Generating bulter gen3 in {butlerRootPath} for {instName}")

        runProgram(f"butler create {butlerRootPath}")

        if instName == "comcam":
            self.log.debug("Registering LsstComCam")
            runProgram(
                f"butler register-instrument {butlerRootPath} lsst.obs.lsst.LsstComCam"
            )
        else:
            self.log.debug("Registering LsstCam")
            runProgram(
                f"butler register-instrument {butlerRootPath} lsst.obs.lsst.LsstCam"
            )

    def generateRefCatalog(self, instName, butlerRootPath, pathSkyFile):
        """Generate reference star catalog.

        Parameters
        ----------
        instName: `str`
            Name of the instrument.
        butlerRootPath: `str`
            Path to the butler gen3 repository.
        pathSkyFile: `str`
            Path to the catalog star file.
        """
        self.log.debug("Creating reference catalog.")

        catDir = os.path.join(butlerRootPath, "skydata")
        skyFilename = os.path.join(catDir, "sky_data.csv")
        catConfigFilename = os.path.join(catDir, "cat.cfg")
        catRefConfigFilename = os.path.join(catDir, "convertRefCat.cfg")

        os.mkdir(catDir)

        # Read sky file and convert it to csv
        skyData = astropy.io.ascii.read(pathSkyFile)
        # Constructing the catalog of stars to use in the wavefront estimation
        # pipeline. Here it assign the g filter. Since this is only for target
        # selection it really doesn't matter which filter we select, as long
        # as it is a valid one.
        skyData.rename_column("Mag", "g")

        skyData.write(skyFilename, format="csv", overwrite=True)

        with open(os.path.join(catDir, "_mapper"), "w") as fp:
            fp.write("lsst.obs.lsst.LsstCamMapper\n")

        with open(catConfigFilename, "w") as fp:
            fp.write(
                """config.ra_name='Ra'
config.dec_name='Decl'
config.id_name='Id'
config.mag_column_list=['g']
"""
            )
        with open(catRefConfigFilename, "w") as fp:
            fp.write('config.datasetIncludePatterns = ["ref_cat", ]\n')
            fp.write('config.refCats = ["cal_ref_cat"]\n')

        runProgram(
            f"ingestReferenceCatalog.py {catDir} {skyFilename} --configfile {catConfigFilename}"
        )

        runProgram(
            f"butler convert --gen2root {catDir} --config-file {catRefConfigFilename} {butlerRootPath}"
        )

    @staticmethod
    def setDefaultParser(parser):
        """Set the default parser.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Input parser.

        Returns
        -------
        argparse.ArgumentParser
            Updated parser.
        """

        parser.add_argument(
            "--inst",
            type=str,
            default="comcam",
            help="Instrument to use: comcam or lsstfam. (default: comcam)",
        )

        parser.add_argument(
            "--filterType",
            type=str,
            default="",
            help="Filter type to use: U, G, R, I, Z, Y or empty string for "
            "reference wavelength. (default: '')",
        )

        parser.add_argument(
            "--rotCam",
            type=float,
            default=0.0,
            help="Rotate camera (degree) in counter-clockwise direction. (default: 0.0)",
        )

        parser.add_argument(
            "--m1m3FErr",
            type=float,
            default=0.05,
            help="Ratio of M1M3 actuator force error between 0 and 1. (default: 0.05)",
        )

        parser.add_argument(
            "--numOfProc",
            type=int,
            default=1,
            help="Number of processor to run PhoSim. (default: 1)",
        )

        parser.add_argument(
            "--iterNum",
            type=int,
            default=5,
            help="Number of closed-loop iteration. (default: 5)",
        )

        parser.add_argument("--output", type=str, default="", help="Output directory.")

        parser.add_argument(
            "--log-level", type=int, default=logging.INFO, help="Log level."
        )

        parser.add_argument(
            "--clobber",
            default=False,
            action="store_true",
            help="Delete existing output directory.",
        )

        # Return the parser intentionally to emphasise this object has been
        # updated.
        return parser

    @staticmethod
    def setImgParser(parser):
        """Set the image-specific parser.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Input parser.

        Returns
        -------
        argparse.ArgumentParser
            Updated parser.
        """

        parser.add_argument(
            "--boresightDeg",
            type=float,
            nargs=2,
            default=[0, 0],
            help="Boresight [ra, dec] in degree. The default is [0, 0].",
        )

        parser.add_argument(
            "--skyFile",
            type=str,
            default="",
            help="""
                 Text file contains the star Id, ra, dec, and magnitude.
                 The default is to use the OPD field positions with boresight
                 [ra, dec] = [0, 0].
                 """,
        )

        parser.add_argument(
            "--eimage", default=False, action="store_true", help="Use the eimage files."
        )

        # Return the parser intentionally to emphasise this object has been
        # updated.
        return parser
