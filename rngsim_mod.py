#!/bin/python
# quick and dirty with no documentation is just the way I roll

# python rngsim_mod.py --prob 0.05 --num 21 --bail 67 --percentile 100 --trials 1000000
# p: 1000000, t=768, suc: 503635 (50%), fail: 496365, ip: 0, total runs: 298808150, total runs per success: 593.3029872824566, drops per success: 29.67541175653003, average runs per success: 373.13974604624383, average runs per failure: 223.38785772566558

# python rngsim_mod.py --prob 0.05 --num 21 --bail 67 --percentile 100 --trials 1000000 --keepgoing
# p: 1000000, t=8227, suc: 503313 (50%), fail: 496687, ip: 0, total runs: 601052278, total runs per success: 1194.1918408624454, drops per success: 59.70799482628106, average runs per success: 973.845251364459, average runs per failure: 223.28609969658962

# generate a distribution graph:
# python rngsim_mod.py --prob 0.05 --num 21 --bail 67 --percentile 100 --trials 1000000 --distx 1407 --disty 200 --distyscale 0.2

import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description="rngsim_mod")
parser.add_argument("--trials", dest="trials", action="store", default="100000")
parser.add_argument("--prob", dest="prob", action="store", default="0.05")
parser.add_argument("--num", dest="num", action="store", default="21")
parser.add_argument("--bail", dest="bail", action="store", default="67")
parser.add_argument("--percentile", dest="percentile", action="store", default="0.5")
parser.add_argument("--keepgoing", dest="keepGoing", action="store_true", default=False)
parser.add_argument("--distx", dest="distX", action="store", default="0")
parser.add_argument("--disty", dest="distY", action="store", default="0")
parser.add_argument("--distyscale", dest="distYScale", action="store", default="1.0")
args = parser.parse_args()

maxPlayers = int(args.trials)
prob = float(args.prob)
successDrops = int(args.num)
bailAfterFailures = int(args.bail)
percentile = float(args.percentile)
keepGoing = args.keepGoing
distX = int(args.distX)
distY = int(args.distY)
distYScale = float(args.distYScale)

outputSteps = 1000


def doRuns():
    successDist = np.zeros(128, np.uint32)
    successCDist = np.zeros(128, np.uint32)
    failureDist = np.zeros(128, np.uint32)
    failureCDist = np.zeros(128, np.uint32)
    totalDist = np.zeros(128, np.uint32)
    totalCDist = np.zeros(128, np.uint32)
    maxTrials = 0
    totalDropCount = 0

    runs = np.zeros(128, np.uint32)

    for players in range(1, maxPlayers + 1):
        success, trials, dropCount = doSingleRun()
        runs = ensureSize(runs, trials)
        runs[1:trials+1] += 1
        maxTrials = max(trials, maxTrials)
        totalDropCount += dropCount

        totalDist, totalCDist = addResult(trials, totalDist, totalCDist)

        if success:
            successDist, successCDist = addResult(trials, successDist, successCDist)
        else:
            failureDist, failureCDist = addResult(trials, failureDist, failureCDist)

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

            if distX > 0 and distY > 0:
                showDist("dist", distX, distY, distYScale, [totalDist, failureDist], [(255, 0, 0), (0, 0, 255)],
                         totalCDist, (0, 255, 0), 1, False)
                cv2.waitKey(1)

    if distX > 0 and distY > 0:
        showDist("dist", distX, distY, distYScale, [totalDist, failureDist], [(255, 0, 0), (0, 0, 255)],
                 totalCDist, (0, 255, 0), 1, True)
        cv2.waitKey(1)


def drawDist(sizeX, sizeY, max_dist, dist, color):
    img = np.zeros((sizeY, sizeX, 3), np.uint8)
    for x in range(0, len(dist)):
        y = sizeY * (1 - (dist[x] / max_dist))
        cv2.line(img, (x, sizeY), (x, int(round(y))), color, 1)
    return img


def showDist(title, sizeX, sizeY, scaleY, dists, colors, cdist, color_cdist, width_cdist, save):
    max_dist = max([max(d) for d in dists]) * scaleY

    imgs = [drawDist(sizeX, sizeY, max_dist, dists[n], colors[n]) for n in range(0, len(dists))]

    img = imgs[0]
    for n in range(1, len(imgs)):
        img += imgs[n]

    max_cdist = max(cdist)

    for x in range(0, len(cdist) - 1):
        y1 = sizeY * (1 - (cdist[x] / max_cdist))
        y2 = sizeY * (1 - (cdist[x] / max_cdist))
        cv2.line(img, (x, int(round(y1))), (x+1, int(round(y2))), color_cdist, width_cdist)

    cv2.imshow(title, img)
    if save:
        cv2.imwrite('{}.png'.format(title), img)


def addResult(result, dist, cdist):
    cdist = ensureSize(cdist, result, cdist[len(cdist) - 1])
    cdist[result:] += 1
    dist = ensureSize(dist, result, 0)
    dist[result] += 1
    return dist, cdist


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
