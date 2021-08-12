import os
import sys
import glob
import pathlib
import setuptools


install_requires = []
tests_require = ["pytest", "pytest-cov", "pytest-flake8"]
dev_requires = install_requires + tests_require + ["documenteer[pipelines]"]
# Getting paths to discover data_files and its final location.
tools_path = pathlib.Path(setuptools.__path__[0])
base_prefix = pathlib.Path(sys.base_prefix)
data_files_path = tools_path.relative_to(base_prefix).parents[1]
# The standard policy/ directory contains a series of subdirectories which
# contents must be passed in as data_files. Because pip does not resolve
# data_files automatically, we need to "manually" select all the files we want
# to copy and their destination.
# The following starts by detemining which are the subdirectories in the policy
# directory.
policy_dir = [
    pp for pp in glob.glob("policy/**", recursive=True) if pathlib.Path(pp).is_dir()
]
# Once we know the subdirectories, create a list of all files inside them, and
# create a list of tuples with (destination, [list, of, files]).
# This is later passed on to setup as data_files.
policy_files = [
    (
        os.path.join(data_files_path, pd),
        [pf for pf in glob.glob(f"{pd}/*") if pathlib.Path(pf).is_file()],
    )
    for pd in policy_dir
]

# Generate a custom __version__.py  file, compatible with our system.
scm_version_template = """# Generated by setuptools_scm
__all__ = ["__version__"]

__version__ = "{version}"
"""

setuptools.setup(
    name="ts_phosim",
    description="High-Level Module to Perturb the PhoSim.",
    use_scm_version={
        "write_to": "python/lsst/ts/phosim/version.py",
        "write_to_template": scm_version_template,
    },
    setup_requires=["setuptools_scm", "pytest-runner"],
    install_requires=install_requires,
    package_dir={"": "python"},
    packages=setuptools.find_namespace_packages(where="python"),
    data_files=policy_files,
    include_package_data=True,
    scripts=[],
    tests_require=tests_require,
    extras_require={"dev": dev_requires},
    license="GPL",
    project_urls={
        "Bug Tracker": "https://jira.lsstcorp.org/secure/Dashboard.jspa",
        "Source Code": "https://github.com/lsst-ts/ts_phosim",
    },
)
