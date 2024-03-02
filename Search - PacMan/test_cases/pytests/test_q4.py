import os
import sys
print (os.getcwd())
import autograder
import grading
import imp
import optparse
import os
import re
import sys
import projectParams
import random
import pytest
random.seed(0)
try:
    from pacman import GameState
except:
    pass

def test_Q4():
    options = autograder.readCommand(['autograder.py', '-q', 'q4'])

    codePaths = options.studentCode.split(',')
    # moduleCodeDict = {}
    # for cp in codePaths:
    #     moduleName = re.match('.*?([^/]*)\.py', cp).group(1)
    #     moduleCodeDict[moduleName] = readFile(cp, root=options.codeRoot)
    # moduleCodeDict['projectTestClasses'] = readFile(options.testCaseCode, root=options.codeRoot)
    # moduleDict = loadModuleDict(moduleCodeDict)

    moduleDict = {}
    for cp in codePaths:
        moduleName = re.match('.*?([^/]*)\.py', cp).group(1)
        moduleDict[moduleName] = autograder.loadModuleFile(moduleName, os.path.join(options.codeRoot, cp))
    moduleName = re.match('.*?([^/]*)\.py', options.testCaseCode).group(1)
    moduleDict['projectTestClasses'] = autograder.loadModuleFile(moduleName, os.path.join(options.codeRoot, options.testCaseCode))

    res = autograder.evaluate(options.generateSolutions, options.testRoot, moduleDict,
                   gsOutput=options.gsOutput,
                   edxOutput=options.edxOutput, muteOutput=options.muteOutput, printTestCase=options.printTestCase,
                   questionToGrade=options.gradeQuestion, display=autograder.getDisplay(options.gradeQuestion != None, options))

    if(res['q4'] != 5):
        pytest.fail("Q4 did not pass try \"python autograder.py -q q4\" for more infomration")

if __name__ == '__main__':
    test_Q4()