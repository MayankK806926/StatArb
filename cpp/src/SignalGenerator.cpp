#include "SignalGenerator.h"
#include <numeric>
#include <cmath>

double SignalGenerator::computeZScore(double value,
                                       const std::vector<double>& window) const {
    if ((int)window.size() < cfg_.zScoreWindow) return 0.0;
    // Use last zScoreWindow observations
    int start = (int)window.size() - cfg_.zScoreWindow;
    std::vector<double> w(window.begin() + start, window.end());
    double m = Stats::mean(w);
    double s = Stats::stddev(w);
    if (s < 1e-10) return 0.0;
    return (value - m) / s;
}

std::vector<BarSignal>
SignalGenerator::generateSignals(const std::vector<double>& spread) const {
    int n = spread.size();
    std::vector<BarSignal> signals(n);

    auto rmean = Stats::rollingMean(spread, cfg_.zScoreWindow);
    auto rstd  = Stats::rollingStd (spread, cfg_.zScoreWindow);

    int  currentPos  = 0;  // 0=flat, +1=long spread, -1=short spread
    int  holdBars    = 0;

    for (int i = 0; i < n; ++i) {
        BarSignal& bs = signals[i];
        bs.spread      = spread[i];
        bs.rollingMean = rmean[i];
        bs.rollingStd  = rstd[i];
        bs.entrySignal = false;
        bs.exitSignal  = false;
        bs.stopSignal  = false;

        // Compute z-score.
        // Important: even if rolling std is ~0 (e.g. constant spread windows),
        // we still want exit/stop logic to run with zScore==0.
        if (i < cfg_.zScoreWindow) {
            bs.zScore = 0.0;
            bs.signal = currentPos;
            continue;
        }

        double denom = rstd[i];
        bs.zScore = (denom < 1e-10) ? 0.0 : (spread[i] - rmean[i]) / denom;

        // ── Position management ──
        if (currentPos != 0) {
            ++holdBars;

            // Check exit conditions
            bool meanReverted = (std::abs(bs.zScore) < cfg_.exitZScore);
            bool stopped      = (std::abs(bs.zScore) > cfg_.stopZScore);
            bool forcedExit   = (holdBars >= cfg_.maxHoldBars);

            if ((meanReverted || stopped || forcedExit)
                && holdBars >= cfg_.minHoldBars) {
                bs.exitSignal  = meanReverted;
                bs.stopSignal  = stopped;
                currentPos     = 0;
                holdBars       = 0;
            }
        }

        // ── Entry logic (only when flat) ──
        if (currentPos == 0) {
            if (bs.zScore < -cfg_.entryZScore) {
                currentPos     = +1;  // spread too low → long A, short B
                holdBars       = 0;
                bs.entrySignal = true;
            } else if (bs.zScore > cfg_.entryZScore) {
                currentPos     = -1;  // spread too high → short A, long B
                holdBars       = 0;
                bs.entrySignal = true;
            }
        }

        bs.signal = currentPos;
    }

    return signals;
}

BarSignal SignalGenerator::updateBar(double spread,
                                      const std::vector<double>& spreadHistory,
                                      int currentPositionSignal) const {
    BarSignal bs;
    bs.spread = spread;
    bs.signal = currentPositionSignal;
    bs.entrySignal = false;
    bs.exitSignal  = false;
    bs.stopSignal  = false;

    if ((int)spreadHistory.size() < cfg_.zScoreWindow) {
        bs.zScore = 0.0;
        bs.rollingMean = 0.0;
        bs.rollingStd  = 0.0;
        return bs;
    }

    int start = (int)spreadHistory.size() - cfg_.zScoreWindow;
    std::vector<double> w(spreadHistory.begin()+start, spreadHistory.end());
    bs.rollingMean = Stats::mean(w);
    bs.rollingStd  = Stats::stddev(w);
    if (bs.rollingStd < 1e-10) bs.zScore = 0.0;
    else bs.zScore = (spread - bs.rollingMean) / bs.rollingStd;

    if (currentPositionSignal != 0) {
        if (std::abs(bs.zScore) < cfg_.exitZScore) bs.exitSignal = true;
        if (std::abs(bs.zScore) > cfg_.stopZScore) bs.stopSignal = true;
    } else {
        if (bs.zScore < -cfg_.entryZScore) { bs.entrySignal = true; bs.signal = +1; }
        else if (bs.zScore > cfg_.entryZScore) { bs.entrySignal = true; bs.signal = -1; }
    }
    return bs;
}