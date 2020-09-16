#!/bin/python
# quick and dirty with no documentation is just the way I roll

# python rngsim_mod.py --prob 0.05 --num 21 --bail 67 --percentile 100 --trials 1000000
# p: 1000000, t=768, suc: 503635 (50%), fail: 496365, ip: 0, total runs: 298808150, total runs per success: 593.3029872824566, drops per success: 29.67541175653003, average runs per success: 373.13974604624383, average runs per failure: 223.38785772566558

# python rngsim_mod.py --prob 0.05 --num 21 --bail 67 --percentile 100 --trials 1000000 --keepgoing
# p: 1000000, t=8227, suc: 503313 (50%), fail: 496687, ip: 0, total runs: 601052278, total runs per success: 1194.1918408624454, drops per success: 59.70799482628106, average runs per success: 973.845251364459, average runs per failure: 223.28609969658962

import numpy as np

import argparse

parser = argparse.ArgumentParser(description="rngsim_mod")
parser.add_argument("--trials", dest="trials", action="store", default="100000")
parser.add_argument("--prob", dest="prob", action="store", default="0.05")
parser.add_argument("--num", dest="num", action="store", default="21")
parser.add_argument("--bail", dest="bail", action="store", default="67")
parser.add_argument("--percentile", dest="percentile", action="store", default="0.5")
parser.add_argument("--keepgoing", dest="keepGoing", action="store_true", default=False)
args = parser.parse_args()

maxPlayers = int(args.trials)
prob = float(args.prob)
successDrops = int(args.num)
bailAfterFailures = int(args.bail)
percentile = float(args.percentile)
keepGoing = args.keepGoing

outputSteps = 1000


def doRuns():
    successDist = np.zeros(128, np.uint32)
    successCDist = np.zeros(128, np.uint32)
    failureDist = np.zeros(128, np.uint32)
    failureCDist = np.zeros(128, np.uint32)
    maxTrials = 0
    totalDropCount = 0

    runs = np.zeros(128, np.uint32)

    for players in range(1, maxPlayers + 1):
        success, trials, dropCount = doSingleRun()
        runs = ensureSize(runs, trials)
        runs[1:trials+1] += 1
        maxTrials = max(trials, maxTrials)
        totalDropCount += dropCount

        if success:
            successCDist = ensureSize(successCDist, trials, successCDist[len(successCDist) - 1])
            successCDist[trials:] += 1
            successDist = ensureSize(successDist, trials, 0)
            successDist[trials] += 1
        else:
            failureCDist = ensureSize(failureCDist, trials, failureCDist[len(failureCDist) - 1])
            failureCDist[trials:] += 1
            failureDist = ensureSize(failureDist, trials, 0)
            failureDist[trials] += 1

        if players % outputSteps == 0:
            index = np.searchsorted(successCDist, percentile * players)
            totalRuns = np.sum(runs[0:index])
            inprogress = 0 if index >= len(runs) else runs[index+1]
            successes = successCDist[np.min([index, len(successCDist) - 1])]
            failures = failureCDist[np.min([index+1, len(failureCDist) - 1])]
            averageRunsPerSuccess = sum([successDist[n]*n for n in range(len(successDist))]) / successes
            averageRunsPerFailure = sum([failureDist[n]*n for n in range(len(failureDist))]) / failures
            totalRunsPerSuccess = totalRuns / successes
            dropsPerSuccess = totalDropCount / successes

            print("p: {}, t={}, suc: {} ({:.00f}%), fail: {}, ip: {}, "
                  "total runs: {}, total runs per success: {}, drops per success: {}, "
                  "average runs per success: {}, average runs per failure: {}".
                  format(players, maxTrials, successes, (100 * (successes / players)), failures, inprogress,
                         totalRuns, totalRunsPerSuccess, dropsPerSuccess,
                         averageRunsPerSuccess, averageRunsPerFailure))


def ensureSize(buffer, index, defaultValue=0):
    while len(buffer) <= index:
        buffer2 = np.zeros([len(buffer)*2], np.uint32) if defaultValue == 0 \
            else np.full([len(buffer)*2], defaultValue)
        buffer2[0:len(buffer)] = buffer
        buffer = buffer2
    return buffer


def doSingleRun():
    trials = 0
    dropCount = 0
    failures = 0

    done = False
    while not done:
        trials += 1
        if drop():
            dropCount += 1
            failures = 0
        else:
            failures += 1
        if dropCount == successDrops and not keepGoing:
            done = True
            # print("SUCCESS: Ran {}, got {}".format(trials, dropCount))
        elif failures == bailAfterFailures:
            done = True
            # print("FAILURE: Ran {}, got {}".format(trials, dropCount))

    return dropCount >= successDrops, trials, dropCount


# Permuted Congruential Generator seeded with entropy from the OS

rng = np.random.default_rng()


def drop():
    # uniform distribution, open-ended on the right
    return rng.uniform() < prob


doRuns()
