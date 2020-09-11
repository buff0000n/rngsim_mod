#!/bin/python
# quick and dirty with no documentation is just the way I roll

import numpy as np

import argparse

parser = argparse.ArgumentParser(description="rngsim_mod")
parser.add_argument("--prob", dest="prob", action="store", default="0.05")
parser.add_argument("--num", dest="num", action="store", default="21")
parser.add_argument("--bail", dest="bail", action="store", default="67")
parser.add_argument("--percentile", dest="percentile", action="store", default="0.5")
args = parser.parse_args()

prob = float(args.prob)
successDrops = int(args.num)
bailAfterFailures = int(args.bail)
percentile = float(args.percentile)

maxPlayers = 100000
outputSteps = 1000


def doRuns():
    successCDist = np.zeros(128, np.uint32)
    failureCDist = np.zeros(128, np.uint32)
    maxTrials = 0

    runs = np.zeros(128, np.uint32)

    for players in range(1, maxPlayers + 1):
        success, trials = doSingleRun()
        runs = ensureSize(runs, trials)
        runs[1:trials+1] += 1
        maxTrials = max(trials, maxTrials)

        if success:
            successCDist = ensureSize(successCDist, trials, successCDist[len(successCDist) - 1])
            successCDist[trials:] += 1
        else:
            failureCDist = ensureSize(failureCDist, trials, failureCDist[len(failureCDist) - 1])
            failureCDist[trials:] += 1

        if players % outputSteps == 0:
            index = np.searchsorted(successCDist, percentile * players)
            totalRuns = np.sum(runs[0:index])
            inprogress = 0 if index >= len(runs) else runs[index+1]
            successes = successCDist[np.min([index, len(successCDist) - 1])]
            failures = failureCDist[np.min([index+1, len(failureCDist) - 1])]
            runsPerSuccess = totalRuns / successes

            print("players: {}, successes at t={}: {}, failures: {}, " 
                  "in progress: {}, total runs: {}, runs per success: {}".
                  format(players, maxTrials, successes, failures, inprogress, totalRuns, runsPerSuccess))


def ensureSize(buffer, index, defaultValue=0):
    while len(buffer) <= index:
        buffer2 = np.zeros([len(buffer)*2], np.uint32) if defaultValue == 0 \
            else np.full([len(buffer)*2], defaultValue)
        buffer2[0:len(buffer)] = buffer
        buffer = buffer2
    return buffer


def doSingleRun():
    trials = 0
    successCount = 0
    success = None
    failures = 0

    while success is None:
        trials += 1
        if drop():
            successCount += 1
            failures = 0
        else:
            failures += 1
        if successCount == successDrops:
            success = True
            # print("SUCCESS: Ran {}, got {}".format(trials, successCount))
        elif failures == bailAfterFailures:
            success = False
            # print("FAILURE: Ran {}, got {}".format(trials, successCount))

    return success, trials


# Permuted Congruential Generator seeded with entropy from the OS

rng = np.random.default_rng()


def drop():
    # uniform distribution, open-ended on the right
    return rng.uniform() < prob


doRuns()
