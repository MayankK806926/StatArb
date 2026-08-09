#pragma once
#include "Utils.h"
#include "TradeLog.h"
#include "WalkForwardEngine.h"
#include <string>

// ─── ReportGenerator ─────────────────────────────────────────────────────────
// Produces:
//   1. Console tearsheet (8 performance metrics)
//   2. Per-window walk-forward summary
//   3. CSV exports: trades, equity curve, window summary
//   4. Comparison: in-sample vs out-of-sample Sharpe (overfitting check)
// ─────────────────────────────────────────────────────────────────────────────
class ReportGenerator {
public:
    // Full tearsheet to stdout
    void printTearsheet(const TradeLog& log,
                        const std::string& title = "BACKTEST TEARSHEET") const;

    // Walk-forward window comparison table
    void printWalkForwardSummary(
        const std::vector<WindowResult>& windows) const;

    // In-sample vs OOS overfitting check
    void printOverfitCheck(
        const std::vector<WindowResult>& windows) const;

    // Export all results to output directory
    void exportAll(const TradeLog&                    combinedLog,
                   const std::vector<WindowResult>&   windows,
                   const std::string&                 outDir) const;

private:
    std::string progressBar(double val, double maxVal, int width = 20) const;
    std::string formatPnl(double v) const;
};