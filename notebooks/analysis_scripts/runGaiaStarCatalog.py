import os
import numpy as np
from baseComcamLoop import baseComcamLoop as comcamLoop
from baseComcamLoop import _eraseFolderContent
from createPhosimCatalogNew import createPhosimCatalog
from lsst.ts.phosim.Utility import getPhoSimPath, getAoclcOutputPath, getConfigDir
from lsst.ts.wep.Utility import runProgram

# initial setting whether  to calculate opd, etc. 
opd = True
flats = True
defocalImg = True


# Load directory paths
phosimDir = getPhoSimPath()
testLabel = 'gaia'

# settings for simulation
numPro = 20 # of 90 : the number of processors to use  in phosimCmptSetting.yaml 
iterNum  = 1 # number of iterations 
numFields = 9 # 9 for all CCDs,  3 to get the result quicker... 


# We use as PhoSim catalog a GAIA star catalog ...  
 
# since at such small separations the donuts overlap, we need to 
# turn on the deblending ....
doDeblending = True 

# we want to save postage stamps - the 
# default directory is 
# outputDir + '/postage'
postageImg = True 

# dir from /analysis_scripts/ level
# - that's where we save the results:
topDir = 'results_gaia'
expDir = 'gMagGt11_R22' # name of the experiment dir 

# dir from /analysis_scripts/ level 
# - that's where we copy the flats, opd from:
copyDir = 'results_before_centroid_update/singleAmpSep'

# the opd and wfs are stored here 
os.environ["closeLoopTestDir"] = os.path.join(topDir, expDir) 

print('\nStarting ccLoop for GAIA catalog')

outputDir = os.path.join(topDir,expDir)
print('The outputDir is %s'%outputDir)
if (not os.path.exists(outputDir)):
    os.makedirs(outputDir)
        
# re-use the flats and calibs,
# since these have nothing to do with the actual stars ...
# -  copy from the sep_10  ...
if not opd and not flats : 
    print('Copying content of /sep_10/ to re-use the flats and OPD files...')
    
    # first copy the old results...
    argString = '-a '+os.path.join(copyDir,'sep_10')+'/. '+\
                 outputDir+'/'
    runProgram("cp", argstring=argString)

    # ensure that input/raw and input/rerun are empty 
    print('Deleting content of input/raw/ and input/rerun/')
    _eraseFolderContent(os.path.join(outputDir, 'input','raw'))
    _eraseFolderContent(os.path.join(outputDir, 'input','rerun'))

    # remove files that are remade
    argString = os.path.join(outputDir, 'input')
    runProgram("rm", argstring=argString+'/isr*')
    runProgram("rm", argstring=argString+'/registry*')
    runProgram("rm", argstring=argString+'/_mappe*')

# Clobber
if opd is True:
    print('We will make new OPD files in this run')
    _eraseFolderContent(outputDir)
else:
    if flats is True:
        print('We will make new flats in this run')
        _eraseFolderContent(os.path.join(outputDir, 'fake_flats'))
        _eraseFolderContent(os.path.join(outputDir, 'input'))     
    if defocalImg is True:
        print('We will make new defocal images in this run ')
        intraPath = os.path.join(outputDir, 'iter0', 'img', 'intra')
        extraPath = os.path.join(outputDir, 'iter0', 'img', 'extra')
        if os.path.exists(intraPath):
            _eraseFolderContent(intraPath)
        if os.path.exists(extraPath):
            _eraseFolderContent(extraPath)


# read the star catalog from file ... 
# it conforms to the format expected by PhoSim 
# mv /data/epyc/users/suberlak/starCatGAIA.txt analysis_scripts/results_gaia
skyFilePath = os.path.join(topDir,'starCatGAIA_gt11.txt')
print('Using sky catalog from %s'%skyFilePath)
       
# set the opd.cmd and star.cmd files ...
opdCmd  = 'opdQuickBackground.cmd'
comcamCmd = 'starQuickBackground.cmd'

print('For PhoSim using /policy/cmdFile/%s and %s'%(opdCmd,comcamCmd))

# initialize the baseComcamLoop.py Class 
ccLoop = comcamLoop() 
testName = 'gaia' 
print('The testName is %s'%testName)
ccLoop.main(phosimDir, numPro, iterNum, outputDir, testName, 
            isEimg=False,  genOpd=opd, genDefocalImg=defocalImg, 
            genFlats=flats, useMinDofIdx=False,
            inputSkyFilePath=skyFilePath, m1m3ForceError=0.05,
            doDeblending=doDeblending, postageImg=postageImg ,
            opdCmdSettingsFile=opdCmd,
            comcamCmdSettingsFile=comcamCmd)
print('Done running ccLoop for GAIA \n\n')
