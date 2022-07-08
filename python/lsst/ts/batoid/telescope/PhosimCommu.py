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
import subprocess

from lsst.ts.wep.Utility import FilterType

from lsst.ts.phosim.utils.Utility import SurfaceType


class PhosimCommu(object):

    DOF_START_IDX = 5
    DOF_NUM = 50

    def __init__(self, phosimDir=""):
        """Initialization of PhoSim communication class.

        Parameters
        ----------
        phosimDir : str, optional
            PhoSim directory. (the default is "".)
        """

        self.phosimDir = phosimDir

    def setPhoSimDir(self, phosimDir):
        """Set the directory of PhoSim.

        Parameters
        ----------
        phosimDir : str
            Directory of PhoSim.
        """

        self.phosimDir = phosimDir

    def getPhoSimDir(self):
        """Get the directory of PhoSim.

        Returns
        -------
        str
            Directory of PhoSim.
        """

        return self.phosimDir

    def getFilterId(self, filterType):
        """Get the active filter ID in PhoSim.

        Parameters
        ----------
        filterType : enum 'FilterType'
            Active filter.

        Returns
        -------
        int
            Active filter ID in PhoSim.

        Raises
        ------
        ValueError
            The filter type is not defined in PhoSim.
        """

        filterId = -1
        if filterType == FilterType.U:
            filterId = 0
        elif filterType == FilterType.G:
            filterId = 1
        elif filterType == FilterType.R:
            filterId = 2
        elif filterType == FilterType.I:
            filterId = 3
        elif filterType == FilterType.Z:
            filterId = 4
        elif filterType == FilterType.Y:
            filterId = 5
        else:
            raise ValueError(
                "The filter type (%s) is not defined in PhoSim." % filterType
            )

        return filterId

    def getSurfaceId(self, surfaceType):
        """Get the surface ID in PhoSim.

        Parameters
        ----------
        surfaceType : enum 'SurfaceType'
            Surface type.

        Returns
        -------
        int
            Surface ID in PhoSim.
        """

        # LSST surface ID in PhoSim
        # "L1F" means lense 1 front. "L1B" means lense 1 back.
        # "chip" means individual chip.
        surfaceID = -1
        if surfaceType == SurfaceType.M1:
            surfaceID = 0
        elif surfaceType == SurfaceType.M2:
            surfaceID = 1
        elif surfaceType == SurfaceType.M3:
            surfaceID = 2
        elif surfaceType == SurfaceType.L1F:
            surfaceID = 3
        elif surfaceType == SurfaceType.L1B:
            surfaceID = 4
        elif surfaceType == SurfaceType.L2F:
            surfaceID = 5
        elif surfaceType == SurfaceType.L2B:
            surfaceID = 6
        elif surfaceType == SurfaceType.FilterF:
            surfaceID = 7
        elif surfaceType == SurfaceType.FilterB:
            surfaceID = 8
        elif surfaceType == SurfaceType.L3F:
            surfaceID = 9
        elif surfaceType == SurfaceType.L3B:
            surfaceID = 10
        elif surfaceType == SurfaceType.FP:
            surfaceID = 11
        elif surfaceType == SurfaceType.Chip:
            surfaceID = 12

        return surfaceID

    def doDofPert(self, dofInUm):
        """Do the perturbation of DOF.

        DOF: Degree of freedom.

        Parameters
        ----------
        dofInUm : list or numpy.ndarray
            DOF in um.

        Returns
        -------
        str
            Perturbation command used in PhoSim.

        Raises
        ------
        ValueError
            The number of DOF is incorrect.
        """

        # Check the length of degree of freedom.
        if len(dofInUm) != int(self.DOF_NUM):
            raise ValueError("The length of DOF should be %s." % int(self.DOF_NUM))

        # List the phosim dof here
        # idx 5-9: M2 dz, dx, dy, rx, ry
        # idx 10-14: Cam dz, dx, dy, rx, ry
        # idx 15-34: M1M3 20 bending modes
        # idx 35-54: M2 20 bending modes

        # Write the perturbation of degree of freedom
        content = ""
        idxRange = range(self.DOF_START_IDX, self.DOF_START_IDX + self.DOF_NUM)
        for idx, dof in zip(idxRange, dofInUm):
            content += "move %d %7.4f \n" % (idx, dof)

        return content

    def doSurfPert(self, surfId, zkInMm):
        """Do the perturbation of surface in the basis of Zernike polynomial.

        Parameters
        ----------
        surfId : int
            PhoSim surface ID.
        zkInMm : list or numpy.ndarray
            Zernike terms in mm starts from z0.

        Returns
        -------
        str
            Perturbation command used in PhoSim.
        """

        content = ""
        for ii in range(len(zkInMm)):
            content += "izernike %d %d %s \n" % (surfId, ii, zkInMm[ii])

        return content

    def doSurfMapPert(self, surfId, surfFilePath, relScale):
        """Do the perturbation of surface map of individual optics.

        Parameters
        ----------
        surfId : int
            PhoSim surface Id.
        surfFilePath : str
            File path of surface.
        relScale : int
            Relative scaling.

        Returns
        -------
        str
            Perturbation command used in PhoSim.
        """

        # Use the absolute path of surface file
        surfFilePath = os.path.abspath(surfFilePath)

        content = "surfacemap %d %s %d \n" % (surfId, surfFilePath, relScale)

        return content

    def doCameraConfig(self, sciSensorOn=False, wfSensorOn=False, guidSensorOn=False):
        """Do the camera configuratioin.

        Parameters
        ----------
        sciSensorOn : bool, optional
            Science sensors are on. (the default is False.)
        wfSensorOn : bool, optional
            Wavefront sensors are on. (the default is False.)
        guidSensorOn : bool, optional
            Guider sensors are on. (the default is False.)

        Returns
        -------
        str
            Instance command used in PhoSim.
        """

        # Bit mask that defines which sensor groups are used
        # (For LSST: bitmask where first bit is science sensors on;
        # second bit is wavefront sensors on; third bit is guiders on)

        camConfigId = int(sciSensorOn) + 2 * int(wfSensorOn) + 4 * int(guidSensorOn)

        content = "camconfig %d \n" % camConfigId

        return content

    def doSurfLink(self, surfId1, surfId2):
        """Do the surface linkage.

        Parameters
        ----------
        surfId1 : int
            Surface Id 1.
        surfId2 : int
            Surface Id 2.

        Returns
        -------
        str
            Perturbation command used in PhoSim.
        """

        content = "surfacelink %d %d \n" % (surfId1, surfId2)

        return content

    def generateOpd(self, opdId, fieldXInDeg, fieldYInDeg, wavelengthInNm):
        """Generate the source of OPD.

        OPD: Optical path difference.

        Parameters
        ----------
        opdId : int
            OPD Id.
        fieldXInDeg : float
            Field X in degree.
        fieldYInDeg : float
            Field Y in degree.
        wavelengthInNm : float
            Wavelength of the OPD source in nm.

        Returns
        -------
        str
            Perturbation command used in PhoSim.
        """

        content = "opd %2d\t%9.6f\t%9.6f %5.1f \n" % (
            opdId,
            fieldXInDeg,
            fieldYInDeg,
            wavelengthInNm,
        )

        return content

    def generateStar(
        self,
        starId,
        ra,
        dec,
        magNorm,
        sedName,
        redshift=0,
        gamma1=0,
        gamma2=0,
        kappa=0,
        deltaRa=0,
        deltaDec=0,
        sourceType="star",
        spatialPars=0,
    ):
        """Generate the star source.

        Parameters
        ----------
        starId : int
            Star Id.
        ra : float
            The right ascension of the center of the object or image in
            decimal degrees.
        dec : float
            The declination of the center of the object in decimal degrees.
        magNorm : float
            The normalization of the flux of the object in AB magnitudes
            at (500 nm)/(1+z) (which is roughly equivalent to V (AB) or
            g (AB)).
        sedName : str
            The name of the SED file with a file path that is relative to the
            data directory in PhoSim.
        redshift : float, optional
            The redshift (or blueshift) of the object. Note that the SED does
            not need to be redshifted if using this. (the default is 0.)
        gamma1 : float, optional
            The value of the shear parameter gamma1 used in weak lensing.
            (the default is 0.)
        gamma2 : float, optional
            The value of the shear parameter gamma2 used in weak lensing.
            (the default is 0.)
        kappa : float, optional
            The value of the magnification parameter in weak lensing. (the
            default is 0.)
        deltaRa : float, optional
            The value of the declination offset in radians. This can be used
            either for weak lensing or objects that moved from another
            exposure if you do not want to change the source position in the
            first two columns. (the default is 0.)
        deltaDec : float, optional
            The value of the declination offset in radians. This can be used
            either for weak lensing or objects that moved from another
            exposure if you do not want to change the source position in the
            first two columns. (the default is 0.)
        sourceType : str, optional
            The name of the spatial model to be used as defined below. (the
            default is "star".)
        spatialPars : float, optional
            The associated parameters for each spatial model.There could be
            none or many. While the parser is reading the model it looks for
            more parameters based on the name of the model. (the default is 0.)

        Returns
        -------
        str
            Perturbation command used in PhoSim.
        """

        content = "object %2d\t%9.6f\t%9.6f %9.6f ../sky/%s " % (
            starId,
            ra,
            dec,
            magNorm,
            sedName,
        )
        content += "%.1f %.1f %.1f %.1f %.1f %.1f %s %.1f none none \n" % (
            redshift,
            gamma1,
            gamma2,
            kappa,
            deltaRa,
            deltaDec,
            sourceType,
            starId,
        )

        return content

    def getStarInstance(
        self,
        obsId,
        aFilterId,
        ra=0,
        dec=0,
        rot=0,
        mjd=49552.3,
        simSeed=1000,
        filePath=None,
    ):
        """Get the star instance catalog.

        Parameters
        ----------
        obsId : int
            Observation Id.
        aFilterId : int
            PhoSim lsst filter ID.
        ra : float, optional
            Unrefracted Right Ascension in decimal degrees. (the default is 0.)
        dec : float, optional
            Unrefracted Declination in decimal degrees. (the default is 0.)
        rot : float, optional
            Angle of sky relative to camera coordinates (from North over East)
            in decimal degrees. (the default is 0.)
        mjd : float, optional
            MJD of observation. (the default is 49552.3.)
        simSeed : int, optional
            Random number seed. (the default is 1000.)
        filePath : str, optional
            File path to save the instance. (the default is None.)

        Returns
        -------
        str
            Instance command used in PhoSim.
        """

        # Observation Parameters
        content = ""
        content += "Opsim_obshistid %d \n" % obsId
        content += "Opsim_filter %d \n" % aFilterId
        content += "mjd %.10f \n" % mjd
        content += "SIM_SEED %d \n" % simSeed

        # Add the sky information
        content += "rightascension %.6f \n" % ra
        content += "declination %.6f \n" % dec
        content += "rotskypos %.6f \n" % rot

        if filePath is not None:
            self.writeToFile(filePath, content=content, mode="w")

        return content

    def getOpdInstance(self, obsId, aFilterId, ra=0, dec=0, rot=0, filePath=None):
        """Get the default OPD instance catalog.

        OPD: Optical path difference.

        Parameters
        ----------
        obsId : int
            Observation Id.
        aFilterId : int
            PhoSim LSST filter Id.
        ra : float, optional
            Unrefracted Right Ascension in decimal degrees. (the default is 0.)
        dec : float, optional
            Unrefracted Declination in decimal degrees. (the default is 0.)
        rot : float, optional
            Angle of sky relative to camera coordinates (from North over East)
            in decimal degrees. (the default is 0.)
        filePath : str, optional
            File path to save the instance. (the default is None.)

        Returns
        -------
        str
            Instance command used in PhoSim.
        """

        content = ""
        content += "Opsim_obshistid %d \n" % obsId
        content += "Opsim_filter %d \n" % aFilterId

        # Add the sky information
        content += "rightascension %.6f \n" % ra
        content += "declination %.6f \n" % dec
        content += "rotskypos %.6f \n" % rot

        if filePath is not None:
            self.writeToFile(filePath, content=content, mode="w")

        return content

    def writeToFile(self, filePath, content=None, sourceFile=None, mode="a"):
        """Write the file based on the content to put or the source file to
        copy with.

        Parameters
        ----------
        filePath : str
            File path to write.
        content : str, optional
            Content to write into the file. (the default is None.)
        sourceFile : str, optional
            Source file to write its content into the file. (the default is
            None.)
        mode : str, optional
            Overwrite ("w") or append ("a") the file. (the default is "a".)

        Raises
        ------
        ValueError
            Mode is not supported.
        """

        if mode not in ("w", "a"):
            raise ValueError("Mode: %s is not supported." % mode)

        if (content is not None) or (sourceFile is not None):

            # Open the file. If the file path does not exist, the new file
            # will be generated.
            # Use the append instead of it.
            fid = open(filePath, mode)

            # Write the content into the file
            if content is not None:
                fid.write(content)

            # Write the content of source file into the file
            if sourceFile is not None:
                fSrc = open(sourceFile, "r")

                # Enforce to add "\n" if necessary
                fileContent = fSrc.read()
                if not fileContent.endswith("\n"):
                    fileContent = fileContent + "\n"

                fid.write(fileContent)
                fSrc.close()

            # Close the file
            fid.close()

    def writeSedFile(self, wavelengthInNm):
        """Write the SED file used for PhoSim.

        This is for the monochromatic light wavelength.
        SED: Spectral energy distribution.

        Parameters
        ----------
        wavelengthInNm : float
            Wavelength in nm.

        Returns
        -------
        str
            SED file path.
        """

        # Generate the SED file path
        sedFileName = "sed_%d.txt" % wavelengthInNm
        sedFilePath = os.path.join(self.phosimDir, "data", "sky", sedFileName)

        # Generate the file if necessary
        if not os.path.exists(sedFilePath):
            content = "%d   1.0 \n" % wavelengthInNm
            self.writeToFile(sedFilePath, content=content, mode="w")

        return sedFilePath

    def getPhoSimArgs(
        self,
        instanceFile,
        extraCommandFile=None,
        numProc=1,
        numThread=1,
        outputDir=None,
        instrument="lsst",
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
        instrument : str, optional
            Instrument site directory. (the default is "lsst".)
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

        # Use the absolute path
        instanceFile = self._getAbsPathIfNotNone(instanceFile)
        extraCommandFile = self._getAbsPathIfNotNone(extraCommandFile)
        outputDir = self._getAbsPathIfNotNone(outputDir)
        logFilePath = self._getAbsPathIfNotNone(logFilePath)

        # Prepare the argument list
        argString = "%s -i %s -e %d" % (instanceFile, instrument, e2ADC)

        if extraCommandFile is not None:
            argString += " -c %s" % extraCommandFile

        if numProc > 1:
            argString += " -p %d" % numProc

        if numThread > 1:
            argString += " -t %d" % numThread

        if sensorName is not None:
            argString += " -s %s" % sensorName

        if outputDir is not None:
            argString += " -o %s" % outputDir
            argString += " -w %s" % outputDir

        if logFilePath is not None:
            argString += " > %s 2>&1" % logFilePath

        return argString

    def _getAbsPathIfNotNone(self, filePath=None):
        """Get the absolute path if the input is not None.

        Parameters
        ----------
        filePath : str, optional
            File path.

        Returns
        -------
        str
            Absolute file path.
        """

        if filePath is not None:
            filePath = os.path.abspath(filePath)

        return filePath

    def runPhoSim(self, argstring="-v"):
        """Run the PhoSim program.

        Parameters
        ----------
        argstring : str, optional
            Arguments for PhoSim. (the default is "-v".)
        """

        # Path of phosim.py script
        phosimRunPath = os.path.join(self.phosimDir, "phosim.py")

        # Command to execute the python
        command = " ".join(["python", phosimRunPath])

        # Run the PhoSim with the related arguments
        self._runProgram(command, argstring=argstring)

    def _runProgram(self, command, binDir=None, argstring=None):
        """Run the program w/o arguments.

        Parameters
        ----------
        command : str
            Command of application.
        binDir : str, optional
            Directory of binary application. (the default is None.)
        argstring : {[type]}, optional
            Arguments of program. (the default is None.)

        Raises
        ------
        RuntimeError
            There is the error in running the program.
        """

        # Directory of binary application
        if binDir is not None:
            command = os.path.join(binDir, command)

        # Arguments for the program
        if argstring is not None:
            command += " " + argstring

        # Call the program w/o arguments
        if subprocess.call(command, shell=True) != 0:
            raise RuntimeError("Error running: %s" % command)


if __name__ == "__main__":
    pass
