#pragma once
#include "Utils.h"
#include <fstream>

// ─── TradeLog ─────────────────────────────────────────────────────────────────
// Records every trade, tracks daily P&L, builds equity curve.
// Used by both the backtester and the report generator.
// ─────────────────────────────────────────────────────────────────────────────
class TradeLog {
public:
    // Record a completed trade
    void addTrade(const Trade& t);

    // Mark-to-market open position (adds a daily P&L entry without closing)
    void markToMarket(double unrealisedPnl, int bar);

    // Finalise: compute all statistics from recorded trades
    void finalise(double initialNAV = 1'000'000.0);

    // ── Accessors ──
    const std::vector<Trade>&  trades()     const { return trades_; }
    const std::vector<double>& equityCurve()const { return equityCurve_; }
    const std::vector<double>& dailyPnl()   const { return dailyPnl_; }

    int    totalTrades()      const { return (int)trades_.size(); }
    int    winningTrades()    const { return wins_; }
    int    losingTrades()     const { return losses_; }
    int    stoppedTrades()    const { return stopped_; }
    double winRate()          const;
    double avgWinPnl()        const;
    double avgLossPnl()       const;
    double profitFactor()     const;
    double totalNetPnl()      const { return totalPnl_; }
    double totalGrossPnl()    const { return totalGross_; }
    double totalCosts()       const { return totalCosts_; }
    double avgHoldingBars()   const;

    // Performance metrics (require finalise() to be called first)
    double sharpe()   const { return sharpe_; }
    double sortino()  const { return sortino_; }
    double maxDD()    const { return maxDD_; }
    double calmar()   const { return calmar_; }
    double annReturn()const { return annReturn_; }
    double var95()    const { return var95_; }

    // Export all trades to CSV
    void exportTradesCSV(const std::string& path) const;

    // Export equity curve to CSV
    void exportEquityCSV(const std::string& path) const;

    // Print summary to stdout
    void printSummary() const;

    void clear();

private:
    std::vector<Trade>  trades_;
    std::vector<double> equityCurve_;
    std::vector<double> dailyPnl_;

    int    wins_ = 0, losses_ = 0, stopped_ = 0;
    double totalPnl_   = 0.0;
    double totalGross_ = 0.0;
    double totalCosts_ = 0.0;
    double sumWinPnl_  = 0.0;
    double sumLossPnl_ = 0.0;
    double sumHold_    = 0.0;

    // Computed by finalise()
    double sharpe_    = 0.0;
    double sortino_   = 0.0;
    double maxDD_     = 0.0;
    double calmar_    = 0.0;
    double annReturn_ = 0.0;
    double var95_     = 0.0;
};