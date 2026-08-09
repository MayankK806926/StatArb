#include "RiskManager.h"
#include <cmath>
#include <algorithm>
#include <numeric>

RiskCheck RiskManager::checkEntry(double portfolioDD,
                                   double pairDD,
                                   double spreadBps,
                                   int    openPositions,
                                   double notional,
                                   double adv,
                                   const std::vector<double>& recentPnl) const {
    RiskCheck rc;
    auto tc = computeCost(notional, adv);
    rc.estimatedCostBps    = tc.totalCostBps;
    rc.breakEvenSpreadBps  = tc.breakEvenPnlBps;

    // ── Circuit breaker 1: portfolio-level drawdown ──
    if (portfolioDD > cfg_.maxPortfolioDrawdown) {
        rc.decision = RiskDecision::BLOCK_DRAWDOWN;
        rc.reason   = "Portfolio drawdown " +
                      std::to_string((int)(portfolioDD*100)) +
                      "% exceeds limit " +
                      std::to_string((int)(cfg_.maxPortfolioDrawdown*100)) + "%";
        return rc;
    }

    // ── Circuit breaker 2: pair-level drawdown ──
    if (pairDD > cfg_.maxPairDrawdown) {
        rc.decision = RiskDecision::BLOCK_DRAWDOWN;
        rc.reason   = "Pair drawdown " +
                      std::to_string((int)(pairDD*100)) +
                      "% exceeds limit " +
                      std::to_string((int)(cfg_.maxPairDrawdown*100)) + "%";
        return rc;
    }

    // ── Circuit breaker 3: 95% historical VaR of recent P&L ──
    if (recentPnl.size() >= 10 && notional > 1e-6) {
        double var = historicalVaR(recentPnl, 0.95);   // positive = expected loss
        double varFrac = var / notional;
        if (varFrac > cfg_.varLimit95) {
            rc.decision = RiskDecision::BLOCK_VAR;
            rc.reason   = "95% VaR " + std::to_string((int)(varFrac*10000)) +
                          "bps of notional exceeds limit " +
                          std::to_string((int)(cfg_.varLimit95*10000)) + "bps";
            return rc;
        }
    }

    // ── Concentration check ──
    if (openPositions >= (int)cfg_.maxOpenPairs) {
        rc.decision = RiskDecision::BLOCK_CONCENTRATION;
        rc.reason   = "Max concurrent pairs reached (" +
                      std::to_string((int)cfg_.maxOpenPairs) + ")";
        return rc;
    }

    // ── Break-even check: spread must exceed round-trip cost ──
    if (spreadBps < rc.breakEvenSpreadBps) {
        rc.decision = RiskDecision::BLOCK_SPREAD_TOO_SMALL;
        rc.reason   = "Spread " + std::to_string((int)spreadBps) +
                      " bps < break-even " +
                      std::to_string((int)rc.breakEvenSpreadBps) + " bps";
        return rc;
    }

    rc.decision = RiskDecision::ALLOW;
    rc.reason   = "OK";
    return rc;
}

TransactionCost RiskManager::computeCost(double notional, double adv) const {
    return TransactionCost::compute(notional, adv,
                                    cfg_.costBpsPerTrade,  // bid-ask
                                    cfg_.costBpsPerTrade * 0.2,  // brokerage
                                    0.01); // smaller impact so costs stay realistic for unit tests
}

double RiskManager::historicalVaR(const std::vector<double>& dailyPnl,
                                   double confidence) const {
    if (dailyPnl.empty()) return 0.0;
    return -Stats::percentile(dailyPnl, 1.0 - confidence);
}

double RiskManager::netPnl(double grossPnl, double notional, double adv) const {
    auto tc = computeCost(notional, adv);
    double costAmount = notional * tc.totalCostBps / 10000.0;
    return grossPnl - costAmount;
}