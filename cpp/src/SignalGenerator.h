#pragma once
#include "Utils.h"

// ─── SignalConfig ─────────────────────────────────────────────────────────────
struct SignalConfig {
    // Z-score thresholds
    double entryZScore  = 2.0;   // enter trade when |z| > this
    double exitZScore   = 0.5;   // exit trade when |z| < this (mean reversion)
    double stopZScore   = 3.5;   // emergency stop if spread keeps diverging
    // Rolling window for z-score normalisation (bars)
    int    zScoreWindow = 30;
    // Minimum holding period before allowing exit (bars)
    int    minHoldBars  = 1;
    // Maximum holding period (force exit)
    int    maxHoldBars  = 60;
};

// ─── BarSignal ────────────────────────────────────────────────────────────────
struct BarSignal {
    double spread;
    double zScore;
    double rollingMean;
    double rollingStd;
    int    signal;      // +1 = long A/short B, -1 = short A/long B, 0 = flat
    bool   entrySignal;
    bool   exitSignal;
    bool   stopSignal;
};

// ─── SignalGenerator ──────────────────────────────────────────────────────────
// Converts a spread time series into actionable trading signals.
// Uses a rolling z-score normalisation over a configurable lookback window.
//
// Entry rules:
//   z < -entryZ  → long spread (buy A, sell B × β)
//   z > +entryZ  → short spread (sell A, buy B × β)
// Exit rules:
//   |z| < exitZ  → close position (spread mean-reverted)
//   NOTE: exitZ must be > 0. |z| < 0 can never be true, so exitZ == 0.0
//   would silently disable the mean-reversion exit entirely.
//   |z| > stopZ  → stop out (spread diverging, protect capital)
//   bars > maxHoldBars → force close
// ─────────────────────────────────────────────────────────────────────────────
class SignalGenerator {
public:
    explicit SignalGenerator(const SignalConfig& cfg = SignalConfig{}) : cfg_(cfg) {}

    // Process an entire spread series → vector of bar signals
    std::vector<BarSignal> generateSignals(const std::vector<double>& spread) const;

    // Single-bar update (for live / walk-forward use)
    BarSignal updateBar(double spread,
                        const std::vector<double>& spreadHistory,
                        int currentPositionSignal) const;

    // Compute z-score for a value given a history window
    double computeZScore(double value,
                          const std::vector<double>& window) const;

    const SignalConfig& config() const { return cfg_; }
    void setConfig(const SignalConfig& c) { cfg_ = c; }

private:
    SignalConfig cfg_;
};