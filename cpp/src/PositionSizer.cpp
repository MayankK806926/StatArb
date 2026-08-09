#include "PositionSizer.h"
#include <cmath>
#include <algorithm>
#include <numeric>

PositionSize PositionSizer::compute(double zScore,
                                     double entryZThreshold,
                                     double priceA, double priceB,
                                     double beta,
                                     double winRate,
                                     double avgWin,
                                     double avgLoss) const {
    PositionSize ps{};

    // ── Kelly fraction ────────────────────────────────────────────────────────
    // Simplified Kelly: f* = p/|L| - q/W
    // where p = win rate, q = loss rate, W = avg win, L = avg loss
    double lossRate = 1.0 - winRate;
    double kellyFull = 0.0;
    if (avgLoss > 1e-6 && avgWin > 1e-6)
        kellyFull = winRate / avgLoss - lossRate / avgWin;

    kellyFull = std::clamp(kellyFull, 0.0, 1.0);
    double kellyFrac = cfg_.kellyFraction * kellyFull;
    // If Kelly estimate is unavailable, fall back to fixed fraction
    if (kellyFrac < 0.01) kellyFrac = cfg_.kellyFraction * 0.25;

    // ── Z-score scaling ───────────────────────────────────────────────────────
    // Larger divergence from mean = more confident trade = larger position
    double zScale = 1.0;
    if (cfg_.zScoreScale && entryZThreshold > 1e-6) {
        zScale = std::min(std::abs(zScore) / entryZThreshold,
                          cfg_.maxZScaleMultiple);
        zScale = std::max(zScale, 0.5);  // minimum 50% of base size
    }

    // ── Notional computation ──────────────────────────────────────────────────
    double baseNotional = cfg_.portfolioNAV
                        * std::min(kellyFrac, cfg_.maxPairAllocation)
                        * zScale;

    baseNotional = std::max(baseNotional, cfg_.minPositionSize);

    // Cap at maxPairAllocation
    double maxNotional = cfg_.portfolioNAV * cfg_.maxPairAllocation;
    baseNotional = std::min(baseNotional, maxNotional);

    ps.notionalA  = baseNotional;
    ps.notionalB  = baseNotional * beta;   // hedged by β
    ps.kellyFull  = kellyFull;
    ps.kellyFrac  = kellyFrac;
    ps.zScale     = zScale;

    if (priceA > 1e-6) ps.sharesA = ps.notionalA / priceA;
    if (priceB > 1e-6) ps.sharesB = ps.notionalB / priceB;

    return ps;
}

void PositionSizer::updateFromTrades(const std::vector<Trade>& trades) {
    if (trades.size() < 5) return;

    // Express win/loss size as a FRACTION of the trade's own position size
    // (not raw dollar P&L) so units match compute()'s default arguments
    // (winRate, avgWin, avgLoss are all fractional returns).
    int wins = 0; int losses = 0;
    double sumWin = 0.0, sumLoss = 0.0;
    for (const auto& t : trades) {
        if (t.positionSize <= 1e-6) continue;
        double ret = t.pnl / t.positionSize;
        if (ret > 0) { ++wins;   sumWin  += ret; }
        else         { ++losses; sumLoss += std::abs(ret); }
    }
    if (wins == 0 || losses == 0) return;

    winRateEst_ = (double)wins / (wins + losses);
    avgWinEst_  = sumWin  / wins;
    avgLossEst_ = sumLoss / losses;

    double lossRate = 1.0 - winRateEst_;
    if (avgLossEst_ > 1e-6 && avgWinEst_ > 1e-6) {
        estimatedKelly_ = winRateEst_ / avgLossEst_ - lossRate / avgWinEst_;
        estimatedKelly_ = std::clamp(estimatedKelly_, 0.0, 1.0);
    }
}