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

import argparse

from lsst.ts.batoid.CloseLoopTask import CloseLoopTask


if __name__ == "__main__":

    # Set the parser
    parser = argparse.ArgumentParser(
        description="Run AOS closed-loop simulation in optical path difference (OPD) level."
    )
    parser = CloseLoopTask.setDefaultParser(parser)

    # Get the arguments
    args = parser.parse_args()

    # Run the simulation
    closeLoopTask = CloseLoopTask()
    closeLoopTask.runOpd(
        args.inst,
        args.filterType,
        args.rotCam,
        args.m1m3FErr,
        args.numOfProc,
        args.iterNum,
        args.output,
        args.clobber,
        args.pipelineFile,
    )
