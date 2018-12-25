#!/usr/bin/env groovy

pipeline {

    agent {
        // Use the docker to assign the Python version.
        // Use the label to assign the node to run the test.
        // The nodes in T&S teams is 'jenkins-el7-1'.
        // It is recommended by SQUARE team do not add the label.
        docker {
            image 'lsstts/aos:w_2018_47'
            args '-u root'
        }
    }

    triggers {
        pollSCM('H * * * *')
    }

    environment {
        // Use the double quote instead of single quote
        // Add the PYTHONPATH
        PYTHONPATH="${env.WORKSPACE}/python:${env.WORKSPACE}/ts_tcs_wep/python"
        // XML report path
        XML_REPORT="jenkinsReport/report.xml"
    }

    stages {
        stage ('Install Requirements') {
            steps {
                // When using the docker container, we need to change
                // the HOME path to WORKSPACE to have the authority
                // to install the packages.
                withEnv(["HOME=${env.WORKSPACE}"]) {
                    sh """
                        source /opt/rh/devtoolset-6/enable
                        source /opt/lsst/loadLSST.bash
                        conda install scikit-image
                        git clone --branch feature/newSimsTag https://github.com/lsst-ts/ts_tcs_wep.git
                        cd ts_tcs_wep/
                        python builder/setup.py build_ext --build-lib python/lsst/ts/wep/cwfs/lib
                    """
                }
            }
        }

        stage('Unit Tests') { 
            steps {
                // Direct the HOME to WORKSPACE for pip to get the
                // installed library.
                // 'PATH' can only be updated in a single shell block.
                // We can not update PATH in 'environment' block.
                // Pytest needs to export the junit report. 
                withEnv(["HOME=${env.WORKSPACE}"]) {
                    sh """
                        source /opt/rh/devtoolset-6/enable
                        source /opt/lsst/loadLSST.bash
                        setup sims_catUtils -t sims_w_2018_47
                        pytest --junitxml=${env.WORKSPACE}/${env.XML_REPORT} ${env.WORKSPACE}/tests/*.py
                    """
                }
            }
        }

        stage('Coverage Analysis') { 
            steps {
                // Do the coverage analysis for multiple files.
                withEnv(["HOME=${env.WORKSPACE}"]) {
                    sh """
                        source /opt/rh/devtoolset-6/enable
                        source /opt/lsst/loadLSST.bash
                        setup sims_catUtils -t sims_w_2018_47
                        ./coverageAnalysis.sh "${env.WORKSPACE}/tests/test*.py"
                    """
                }
            }
        }
    }

    post {        
        always {
            // The path of xml needed by JUnit is relative to
            // the workspace.
            junit 'jenkinsReport/*.xml'

            // Publish the HTML report
            publishHTML (target: [
                allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: true,
                reportDir: 'htmlcov',
                reportFiles: 'index.html',
                reportName: "Coverage Report"
              ])
        }

        cleanup {
            // clean up the workspace
            deleteDir()
        }  
    }
}