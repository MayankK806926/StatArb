#pragma once
#include "Utils.h"

struct SizingConfig {
    double portfolioNAV       = 1'000'000.0; // total portfolio in INR/USD
    double maxPairAllocation  = 0.10;        // max 10% per pair
    double kellyFraction      = 0.25;        // fractional Kelly (safer)
    bool   zScoreScale        = true;        // scale by |z|/entryZ
    double maxZScaleMultiple  = 2.0;         // cap z-scaling at 2×
    double minPositionSize    = 1000.0;      // minimum notional
};

struct PositionSize {
    double notionalA;     // dollar/INR notional in ticker A
    double notionalB;     // dollar/INR notional in ticker B (hedged)
    double sharesA;       // number of shares (if priceA provided)
    double sharesB;
    double kellyFull;     // full Kelly fraction
    double kellyFrac;     // fractional Kelly used
    double zScale;        // z-score scaling factor applied
};

// ─── PositionSizer ────────────────────────────────────────────────────────────
// Implements fractional Kelly criterion with z-score scaling.
//
// Kelly formula:  f* = (μ - r) / σ²
//   where μ = expected return per trade, σ = std dev of trade returns
//
// In practice we use a simplified version:
//   f* = win_rate / avg_loss - loss_rate / avg_win
//   (from historical trade stats, or just use a fixed fraction)
//
// Z-score scaling: position_size *= min(|z| / entryZ, maxScale)
//   Larger spread divergence → larger bet (with cap)
// ─────────────────────────────────────────────────────────────────────────────
class PositionSizer {
public:
    explicit PositionSizer(const SizingConfig& cfg = SizingConfig{}) : cfg_(cfg) {}

    // Compute position size for a new trade
    PositionSize compute(double zScore,
                          double entryZThreshold,
                          double priceA, double priceB,
                          double beta,
                          double winRate = 0.55,
                          double avgWin  = 0.02,
                          double avgLoss = 0.01) const;

    // Update Kelly estimate from recent trade history.
    // Win rate / avg win / avg loss are computed as FRACTIONS of each
    // trade's position size (not raw dollar P&L) so they stay in the same
    // units that compute()'s default arguments use.
    void updateFromTrades(const std::vector<Trade>& trades);

    // Estimates derived by updateFromTrades() (unchanged from defaults
    // until at least 5 trades with both a win and a loss are recorded).
    double estimatedKelly() const  { return estimatedKelly_; }
    double winRateEstimate() const { return winRateEst_; }
    double avgWinEstimate() const  { return avgWinEst_; }
    double avgLossEstimate() const { return avgLossEst_; }

    const SizingConfig& config() const { return cfg_; }

private:
    SizingConfig cfg_;
    double       estimatedKelly_ = 0.25;
    double       winRateEst_ = 0.55;
    double       avgWinEst_  = 0.02;
    double       avgLossEst_ = 0.01;
};