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
import numpy as np
import lsst.daf.butler as dafButler
from lsst.ts.phosim.utils.Utility import getConfigDir
from lsst.ts.phosim.utils.MapSensorNameAndId import MapSensorNameAndId
from lsst.ts.wep.Utility import runProgram, DefocalType, CentroidFindType
from lsst.ts.wep.cwfs.CentroidFindFactory import CentroidFindFactory


class CreatePhosimDonutTemplates(object):
    def __init__(self, tempWorkPath):
        """
        Class to create donut templates from Phosim. The templates
        will be saved to `policy/cwfs/donutTemplateData/phosimTemplates`.
        Requires path to Phosim to be set as environment variable
        `PHOSIMPATH`.

        Parameters
        ----------
        tempWorkPath : str
            Filepath for temporary work files and directories to be created.
        """

        # Location of the data needed for Phosim to generate the templates
        self.templateDataPath = os.path.join(getConfigDir(), "donutTemplateData")
        # Set up the temporary work directory for Phosim output
        self.tempWorkPath = tempWorkPath
        # Specify the location of the butler repo for ingestion of
        # Phosim output and to store the postISR images
        self.repoDir = os.path.join(self.tempWorkPath, "input")

        # Set up butler repository
        self._setUpButlerRepo(self.repoDir)
        self.outputRunName = "donutTemplate"
        self.butler = dafButler.Butler(self.repoDir)

        # The location where we will store the final templates
        self.templateDestDir = os.path.join(self.templateDataPath, "phosimTemplates")

    def _setUpButlerRepo(self, repoDir):
        """
        Create the Gen3 Butler repository needed for ingest and ISR.

        Parameters
        ----------
        repoDir : str
            The filepath to the desired repository location.
        """

        # Set up repository
        runProgram(f"butler create {repoDir}")
        runProgram(f"butler register-instrument {repoDir} lsst.obs.lsst.LsstCam")
        runProgram(f"butler write-curated-calibrations {repoDir} lsst.obs.lsst.LsstCam")

    def _updateButler(self):
        """
        After running pipetask command reload the butler to
        get updated registry.
        """

        self.butler = dafButler.Butler(self.repoDir)

    def createWorkDirectories(self):
        """
        Create the final template directory as well as
        temporary work directories.
        """

        print("Making temporary work directories")
        # First clean up previous work if it exists
        if os.path.exists(self.tempWorkPath):
            self.cleanUpWorkDirs()

        os.makedirs(os.path.join(self.tempWorkPath, "phosimOutput", "extra"))
        os.mkdir(os.path.join(self.tempWorkPath, "phosimOutput", "intra"))
        os.mkdir(os.path.join(self.tempWorkPath, "phosimWorkDir"))
        os.mkdir(os.path.join(self.tempWorkPath, "input"))
        os.mkdir(os.path.join(self.tempWorkPath, "raw"))
        os.mkdir(os.path.join(self.tempWorkPath, "calibs"))
        self._setUpButlerRepo(self.repoDir)

    def createDetectorLists(self, detectorStr=""):
        """
        Create the list of detectors for Phosim and generating flats.

        Parameters
        ----------
        detectorStr : str, optional
            String specifying a set of detectors to generate phosim templates.
            A space is required between each detector name
            (Example: "R22_S11 R22_S10"). If the str is "" then it will
            generate a template for every detector in the focal plane.
            (The default is "".)

        Returns
        -------
        str
            String with the detector names specified in format required
            for Phosim input on the command line.
        """

        if detectorStr == "":
            # Load default sensor file for ts_wep
            sensorNameFile = MapSensorNameAndId()._sensorNameToIdFile
            # Get all sensors in file
            sensorNameList = list(sensorNameFile.getContent().keys())
        else:
            sensorNameList = detectorStr.split(" ")

        detectorStrPhosim = "|".join(sensorNameList)

        return detectorStrPhosim

    def generateDefocalImages(self, detectorStrPhosim, numOfProc):
        """
        Run Phosim to generate the defocal images using
        the provided instance catalogs which generate
        a single donut at the center of each detector.

        Parameters
        ----------
        detectorStrPhosim : str
            String with the detector names specified in format required
            for Phosim input on the command line. Example: "R22_S11|R22_S10"
        numOfProc : int
            Number of processors to use with phosim
        """

        print(f"Running phosim with {numOfProc} processors")

        phosimPath = os.getenv("PHOSIMPATH")
        runPhosimCmd = f"python {phosimPath}phosim.py"
        runPhosimArgs = f"-w {self.tempWorkPath}/phosimWorkDir "
        runPhosimArgs += f'-s "{detectorStrPhosim}" '
        runPhosimArgs += f"-p {numOfProc} "
        runPhosimArgs += "-i lsst "
        runPhosimArgs += "-e 1 "
        runPhosimArgs += f"-c {self.templateDataPath}/star.cmd "

        runPhosimArgsExtra = f"{self.templateDataPath}/starExtra.inst "
        runPhosimArgsExtra += runPhosimArgs
        runPhosimArgsExtra += f"-o {self.tempWorkPath}/phosimOutput/extra"

        runPhosimArgsIntra = f"{self.templateDataPath}/starIntra.inst "
        runPhosimArgsIntra += runPhosimArgs
        runPhosimArgsIntra += f"-o {self.tempWorkPath}/phosimOutput/intra"

        # Generate Defocal Images with Phosim
        runProgram(runPhosimCmd, argstring=runPhosimArgsExtra)
        runProgram(runPhosimCmd, argstring=runPhosimArgsIntra)

    def repackagePhosimImages(self, defocalDist=1500):
        """
        Run the phosim repackager.

        Parameters
        ----------
        defocalDist : int
            The defocal distance in micrometers, translated
            to the FOCUSZ in the header of repackaged images.
            (default: 1500)
        """

        print("Repackaging phosim output")

        argString = f"--out_dir {self.tempWorkPath}/raw "
        argStringExtra = (
            argString
            + f"{self.tempWorkPath}/phosimOutput/extra"
            + f" --focusz -{defocalDist}"
        )
        argStringIntra = (
            argString
            + f"{self.tempWorkPath}/phosimOutput/intra"
            + f" --focusz +{defocalDist}"
        )

        # Run repackager
        runProgram("phosim_repackager.py", argstring=argStringExtra)
        runProgram("phosim_repackager.py", argstring=argStringIntra)

    def ingestImages(self):
        """
        Ingest the raw extrafocal images.
        """

        print("Ingest images")

        runProgram(f"butler ingest-raws {self.repoDir} {self.tempWorkPath}/raw/*.fits")

    def runISR(self, isrConfigFile=""):
        """
        Run ISR on extrafocal images.

        Parameters
        ----------
        isrConfigFile: str, optional
            ISR pipeline configuration file path.
            (Default is in policy/cwfs/donutTemplateData.)
        """

        print("Running ISR")

        if isrConfigFile == "":
            isrConfigFile = os.path.join(
                self.templateDataPath, "createPhosimDonutTemplateConfig.yaml"
            )
        pipetaskCmdStr = f"pipetask run -b {self.repoDir}"
        pipetaskCmdStr += f" -p {isrConfigFile}"
        pipetaskCmdStr += " -i LSSTCam/raw/all,LSSTCam/calib"
        pipetaskCmdStr += " --instrument lsst.obs.lsst.LsstCam"
        pipetaskCmdStr += " --register-dataset-types"
        pipetaskCmdStr += f" --output-run {self.outputRunName}"
        runProgram(pipetaskCmdStr)

        # Update butler
        self._updateButler()

    def generateDataIdLists(self, intraVisitId, extraVisitId):
        """
        Create lists of dataIds to get exposures out of the
        butler repository for the corresponding visits.

        Parameters
        ----------
        intraVisitId : int
            Visit Id of the intrafocal images from Phosim.
        extraVisitId : int
            Visit Id of the extrafocal images from Phosim.

        Returns
        -------
        list of dicts
            List of intrafocal postISRCCD exposure dataId dictionaries
        list of dicts
            List of extrafocal postISRCCD exposure dataId dictionaries
        """

        intraSuffix = str(intraVisitId)[-5:]
        extraSuffix = str(extraVisitId)[-5:]

        registry = self.butler.registry
        postIsrDataList = list(
            registry.queryDatasets("postISRCCD", collections=[self.outputRunName])
        )
        intraIds = []
        extraIds = []
        for dataRef in postIsrDataList:
            dataRefSimple = dataRef.to_simple()
            if str(dataRefSimple.dataId.dataId["exposure"])[-5:] == intraSuffix:
                intraIds.append(dataRefSimple.dataId.dataId)
            elif str(dataRefSimple.dataId.dataId["exposure"])[-5:] == extraSuffix:
                extraIds.append(dataRefSimple.dataId.dataId)

        return intraIds, extraIds

    def cutOutIntraExtraTemplates(self, templateWidth, intraVisitId, extraVisitId):
        """
        Cut out the donut templates from the larger extrafocal and intrafocal
        Phosim images for every detector simulated.

        Parameters
        ----------
        templateWidth : int
            Width of square template image in pixels.
        intraVisitId : int
            Visit Id of the intrafocal images from Phosim.
        extraVisitId : int
            Visit Id of the extrafocal images from Phosim.
        """

        intraDataIds, extraDataIds = self.generateDataIdLists(
            intraVisitId, extraVisitId
        )

        if len(intraDataIds) > 0:
            print("Generating intra templates")
            self.cutOutTemplatesAndSave(templateWidth, DefocalType.Intra, intraDataIds)

        if len(extraDataIds) > 0:
            print("Generating extra templates")
            self.cutOutTemplatesAndSave(templateWidth, DefocalType.Extra, extraDataIds)

    def cutOutTemplatesAndSave(self, templateWidth, defocalType, dataIdList):
        """
        Loop through all detectors in folder and cut out square region
        around the donut from a CCD image to use
        as the donut template for that detector. Saves the cutout to file.

        Parameters
        ----------
        phosimImageDir : str
            Directory where the visit ISR output is located. Inside should
            be folders for each raft.
        templateWidth : int
            Width of square template image in pixels.
        defocalType : enum 'DefocalType'
            Defocal type.
        dataIdList : List of dicts
            List of dataIds of the postISR images from Phosim.
        """
        if defocalType == DefocalType.Intra:
            defocalLabel = "intra"
        else:
            defocalLabel = "extra"

        # Set output path
        phosimTemplateDir = self.templateDestDir
        # Set path to centroid file
        phosimCentroidDir = os.path.join(
            self.tempWorkPath, "phosimOutput", defocalLabel
        )

        stampHalfWidth = int(templateWidth / 2)

        centroidFind = CentroidFindFactory.createCentroidFind(
            CentroidFindType.RandomWalk
        )

        for dataId in dataIdList:
            # Open postISR File
            postIsrImg = self.butler.get(
                "postISRCCD", dataId=dataId, collections=[self.outputRunName]
            )
            detectorName = postIsrImg.getDetector().getName()
            templateName = f"{defocalLabel}_template-{detectorName}.txt"

            # Pick an area around the phosim centroid as the template
            visitId = str(dataId["exposure"])[-5:]
            centroidFilename = f"centroid_lsst_e_90{visitId}_f1_{detectorName}_E000.txt"
            centroidData = np.genfromtxt(
                os.path.join(phosimCentroidDir, centroidFilename),
                unpack=True,
                skip_header=1,
            )
            centroidX = int(centroidData[2])
            centroidY = int(centroidData[3])
            imgData = postIsrImg.getImage().getArray()
            templateStamp = imgData[
                centroidX - stampHalfWidth : centroidX + stampHalfWidth,
                centroidY - stampHalfWidth : centroidY + stampHalfWidth,
            ]
            # Reduce background noise
            templateStampBinary = centroidFind.getImgBinary(templateStamp)

            # Save to file
            np.savetxt(
                os.path.join(phosimTemplateDir, "%s" % templateName),
                templateStampBinary,
                fmt="%i",
            )

    def cleanUpWorkDirs(self):
        """
        Clean up all temporary work directories.
        """

        shutil.rmtree(self.tempWorkPath)
