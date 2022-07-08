# -*- coding: utf-8 -*-
from .PhosimCmpt import PhosimCmpt
from .OpdMetrology import OpdMetrology
from .SkySim import SkySim

# The version file is gotten by the scons. However, the scons does not support
# the build without unit tests. This is a needed function for the Jenkins to
# use.
try:
    from .version import *
except ModuleNotFoundError:
    pass
