# PhoSim Use

*This module is a high-level module to use PhoSim.*

## 1. Platform

- *CentOS 7*
- *python: 3.7.2*
- *scientific pipeline (newinstall.sh from master branch)*
- *phosim_syseng4 (branch: aos, tag: firstdonuts)*

## 2. Needed Package

- *[ts_wep](https://github.com/lsst-ts/ts_wep) - master branch (commit: af48bf0)*
- *[documenteer](https://github.com/lsst-sqre/documenteer) (optional)*
- *[plantuml](http://plantuml.com) (optional)*
- *[sphinxcontrib-plantuml](https://pypi.org/project/sphinxcontrib-plantuml/) (optional)*

## 3. Use of Module

*Setup the WEP environment first, and then, setup the PhoSim environment by eups:*

```bash
cd $ts_phosim_directory
setup -k -r .
scons
```

## 4. Example Script

- **calcOpd.py**: Test the OPD without the subsystem perturbation.
- **calcOpdAndSubSys.py**: Test the OPD with the subsystem perturbation.
- **checkStarAndSubSys.py**: Test the star donut in LSST camera with the subsystem perturbation.
- **checkWfsStarCoor.py**: Test to add the stars on WFS and get the images.
- **checkStarCoor.py**: Test to add the star by pixel position and get the image.
- **checkStarCoorWiLsstFAM.py**: Test to add the star by pixel position in LSST FAM condition and get the images.

## 5. Build the Document

*The user can use `package-docs build` to build the documentation. The packages of documenteer, plantuml, and sphinxcontrib-plantuml are needed. The path of plantuml.jar in doc/conf.py needs to be updated to the correct path. To clean the built documents, use `package-docs clean`. See [Building single-package documentation locally](https://developer.lsst.io/stack/building-single-package-docs.html) for further details.*

## 6. Reference of PhoSim with active optics (AOS)

- The original work was done by Bo Xin and Chuck Claver. The source code can be found in: [IM](https://github.com/bxin/IM).
