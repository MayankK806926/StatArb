#pragma once
#include "Utils.h"

struct RiskConfig {
    double maxPortfolioDrawdown = 0.15;   // halt if portfolio DD > 15%
    double maxPairDrawdown      = 0.08;   // stop pair if pair DD > 8%
    double varLimit95           = 0.02;   // 95% 1-day VaR limit (2% of NAV)
    double maxOpenPairs         = 5;      // concurrent pair positions
    double costBpsPerTrade      = 5.0;    // basis points per leg (one-way)
    double minBreakEvenSpread   = 10.0;   // minimum spread in bps to trade
};

enum class RiskDecision { ALLOW, BLOCK_DRAWDOWN, BLOCK_VAR,
                          BLOCK_CONCENTRATION, BLOCK_SPREAD_TOO_SMALL };

struct RiskCheck {
    RiskDecision decision;
    std::string  reason;
    double       estimatedCostBps;
    double       breakEvenSpreadBps;
    bool         allowed() const { return decision == RiskDecision::ALLOW; }
};

// ─── TransactionCostModel ─────────────────────────────────────────────────────
struct TransactionCost {
    double totalCostBps;        // total round-trip cost in bps
    double breakEvenPnlBps;     // minimum spread move needed to profit
    double oneWayCostBps;

    static TransactionCost compute(double notional,
                                    double adv,           // average daily volume
                                    double bidAskBps = 2.0,
                                    double brokerageBps  = 1.0,
                                    double impactCoeff   = 0.1) {
        TransactionCost tc;
        // Market impact: sqrt model (Almgren et al.)
        double impactBps = impactCoeff * std::sqrt(notional / std::max(adv, 1.0)) * 10000.0;
        tc.oneWayCostBps    = bidAskBps/2.0 + brokerageBps + impactBps;
        tc.totalCostBps     = 2.0 * tc.oneWayCostBps;  // round trip
        tc.breakEvenPnlBps  = tc.totalCostBps;
        return tc;
    }
};

// ─── RiskManager ─────────────────────────────────────────────────────────────
class RiskManager {
public:
    explicit RiskManager(const RiskConfig& cfg = RiskConfig{}) : cfg_(cfg) {}

    // Pre-trade check: should we enter this trade?
    // recentPnl: trailing daily/trade P&L history used for the VaR gate
    // (skipped if fewer than 10 observations are supplied).
    RiskCheck checkEntry(double portfolioDD,
                          double pairDD,
                          double spreadBps,
                          int    openPositions,
                          double notional,
                          double adv = 1e6,
                          const std::vector<double>& recentPnl = {}) const;

    // Compute transaction costs for a trade
    TransactionCost computeCost(double notional, double adv = 1e6) const;

    // Compute historical VaR (95%) from recent daily P&L
    double historicalVaR(const std::vector<double>& dailyPnl,
                          double confidence = 0.95) const;

    // Apply transaction costs to gross P&L
    double netPnl(double grossPnl, double notional, double adv = 1e6) const;

    const RiskConfig& config() const { return cfg_; }

private:
    RiskConfig cfg_;
};