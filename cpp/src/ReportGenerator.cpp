#include "ReportGenerator.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <ctime>

namespace fs = std::filesystem;

static std::string timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[32]; std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

std::string ReportGenerator::progressBar(double val, double maxVal, int w) const {
    if (maxVal <= 0) return repeatStr("░", w);
    int f = (int)(std::clamp(val/maxVal, 0.0, 1.0) * w);
    return repeatStr("█", f) + repeatStr("░", w-f);
}

std::string ReportGenerator::formatPnl(double v) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    if (v >= 0) oss << "+" << v;
    else         oss << v;
    return oss.str();
}

// ─── printTearsheet ───────────────────────────────────────────────────────────
void ReportGenerator::printTearsheet(const TradeLog& log,
                                      const std::string& title) const {
    const std::string line = repeatStr("═", 62);
    std::cout << "\n╔" << line << "╗\n";
    std::cout << "║  " << std::left << std::setw(60) << title << "║\n";
    std::cout << "║  Generated: " << std::setw(49) << timestamp() << "║\n";
    std::cout << "╠" << line << "╣\n";

    auto row = [&](const std::string& lbl, const std::string& val,
                   const std::string& bar = "") {
        std::cout << "║  " << std::left << std::setw(30) << lbl
                  << std::setw(14) << val
                  << std::setw(18) << bar << "║\n";
    };

    // ── Trade statistics ──
    std::cout << "║  " << repeatStr("─", 58) << "  ║\n";
    std::cout << "║  TRADE STATISTICS"
              << std::string(44, ' ') << "║\n";
    std::cout << "║  " << repeatStr("─", 58) << "  ║\n";

    row("Total trades",
        std::to_string(log.totalTrades()));
    row("Win / Loss / Stopped",
        std::to_string(log.winningTrades()) + " / " +
        std::to_string(log.losingTrades()) + " / " +
        std::to_string(log.stoppedTrades()));

    double wr = log.winRate() * 100.0;
    row("Win rate",
        std::to_string((int)wr) + "%",
        progressBar(wr, 100.0));

    {
        std::ostringstream os; os << std::fixed << std::setprecision(2)
                                   << log.avgHoldingBars() << " bars";
        row("Avg hold duration", os.str());
    }
    {
        std::ostringstream os; os << std::fixed << std::setprecision(4) << log.profitFactor();
        row("Profit factor", os.str(),
            progressBar(std::min(log.profitFactor(), 3.0), 3.0));
    }
    {
        std::ostringstream os; os << std::fixed << std::setprecision(2) << log.avgWinPnl();
        row("Avg win P&L", os.str());
    }
    {
        std::ostringstream os; os << std::fixed << std::setprecision(2) << log.avgLossPnl();
        row("Avg loss P&L", os.str());
    }

    // ── P&L ──
    std::cout << "║  " << repeatStr("─", 58) << "  ║\n";
    std::cout << "║  P&L BREAKDOWN"
              << std::string(47, ' ') << "║\n";
    std::cout << "║  " << repeatStr("─", 58) << "  ║\n";
    row("Gross P&L",           formatPnl(log.totalGrossPnl()));
    row("Transaction costs",   formatPnl(-log.totalCosts()));
    row("Net P&L",             formatPnl(log.totalNetPnl()));

    // ── Risk metrics ──
    std::cout << "║  " << repeatStr("─", 58) << "  ║\n";
    std::cout << "║  RISK-ADJUSTED PERFORMANCE (8 METRICS)"
              << std::string(22, ' ') << "║\n";
    std::cout << "║  " << repeatStr("─", 58) << "  ║\n";

    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(3) << log.annReturn()*100 << "%";
        row("Annualised return", os.str(),
            progressBar(std::max(log.annReturn()*100, 0.0), 30.0));
    }
    {
        std::ostringstream os; os << std::fixed << std::setprecision(3) << log.sharpe();
        row("① Sharpe ratio",   os.str(),
            progressBar(std::max(log.sharpe(), 0.0), 3.0));
    }
    {
        std::ostringstream os; os << std::fixed << std::setprecision(3) << log.sortino();
        row("② Sortino ratio",  os.str(),
            progressBar(std::max(log.sortino(), 0.0), 3.0));
    }
    {
        std::ostringstream os; os << std::fixed << std::setprecision(1) << log.maxDD()*100 << "%";
        row("③ Max drawdown",   os.str(),
            progressBar(log.maxDD()*100, 50.0));
    }
    {
        std::ostringstream os; os << std::fixed << std::setprecision(3) << log.calmar();
        row("④ Calmar ratio",   os.str(),
            progressBar(std::max(log.calmar(), 0.0), 3.0));
    }
    {
        std::ostringstream os; os << std::fixed << std::setprecision(1) << wr << "%";
        row("⑤ Win rate",       os.str());
    }
    {
        std::ostringstream os; os << std::fixed << std::setprecision(4) << log.profitFactor();
        row("⑥ Profit factor",  os.str());
    }
    {
        std::ostringstream os; os << std::fixed << std::setprecision(2) << log.var95();
        row("⑦ 95% VaR (1-day)", os.str());
    }
    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(1) << log.avgHoldingBars() << " bars";
        row("⑧ Avg trade duration", os.str());
    }

    std::cout << "╚" << line << "╝\n\n";
}

// ─── printWalkForwardSummary ──────────────────────────────────────────────────
void ReportGenerator::printWalkForwardSummary(
    const std::vector<WindowResult>& windows) const
{
    if (windows.empty()) return;
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           WALK-FORWARD WINDOW SUMMARY                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    WalkForwardEngine::printWindowTable(windows);
}

// ─── printOverfitCheck ────────────────────────────────────────────────────────
void ReportGenerator::printOverfitCheck(
    const std::vector<WindowResult>& windows) const
{
    if (windows.empty()) return;
    std::cout << "\n╔══════════════════════════════════════╗\n";
    std::cout << "║    OVERFITTING CHECK                 ║\n";
    std::cout << "║  IS Sharpe vs OOS Sharpe per window  ║\n";
    std::cout << "╚══════════════════════════════════════╝\n\n";

    double totalIS = 0, totalOOS = 0;
    int degraded = 0;

    for (const auto& w : windows) {
        std::string pair = w.selectedPair.tickerA + "/" + w.selectedPair.tickerB;
        double ratio = (w.isSharpe != 0) ? w.oosSharpe / w.isSharpe : 0.0;
        std::string status = (ratio >= 0.5) ? "OK" : "DEGRADED";
        if (ratio < 0.5) ++degraded;
        totalIS  += w.isSharpe;
        totalOOS += w.oosSharpe;

        std::cout << "  W" << std::setw(2) << w.windowIdx
                  << " " << std::setw(12) << pair.substr(0,11)
                  << "  IS=" << std::fixed << std::setprecision(2) << std::setw(6) << w.isSharpe
                  << "  OOS=" << std::setw(6) << w.oosSharpe
                  << "  ratio=" << std::setw(5) << ratio
                  << "  [" << status << "]\n";
    }
    double n = windows.size();
    std::cout << "\n  Avg IS Sharpe : " << std::fixed << std::setprecision(3) << totalIS/n << "\n";
    std::cout << "  Avg OOS Sharpe: " << totalOOS/n << "\n";
    std::cout << "  IS→OOS Decay  : "
              << std::setprecision(1)
              << (1.0 - totalOOS/std::max(totalIS, 1e-6)) * 100.0 << "%\n";
    std::cout << "  Degraded windows: " << degraded << "/" << (int)n << "\n\n";
}

// ─── exportAll ────────────────────────────────────────────────────────────────
void ReportGenerator::exportAll(
    const TradeLog&                    combinedLog,
    const std::vector<WindowResult>&   windows,
    const std::string&                 outDir) const
{
    fs::create_directories(outDir);

    combinedLog.exportTradesCSV(outDir + "/trades.csv");
    combinedLog.exportEquityCSV(outDir + "/equity.csv");

    // Window summary CSV
    {
        std::ofstream f(outDir + "/windows.csv");
        f << "window,pair,train_start,train_end,test_start,test_end,"
          << "num_trades,oos_net_pnl,oos_sharpe,oos_max_dd,oos_win_rate,"
          << "is_sharpe,kalman_delta\n";
        for (const auto& w : windows) {
            f << w.windowIdx << ","
              << w.selectedPair.tickerA << "/" << w.selectedPair.tickerB << ","
              << w.trainStart << "," << w.trainEnd << ","
              << w.testStart  << "," << w.testEnd  << ","
              << w.numTrades  << ","
              << std::fixed << std::setprecision(4)
              << w.oosNetPnl  << "," << w.oosSharpe << ","
              << w.oosMaxDD   << "," << w.oosWinRate << ","
              << w.isSharpe   << "," << w.kalmanDelta << "\n";
        }
        std::cout << "[INFO] Wrote " << outDir << "/windows.csv\n";
    }

    // Text report
    {
        std::ofstream f(outDir + "/report.txt");
        // Redirect cout temporarily not feasible; write key stats manually
        f << "Statistical Arbitrage Report\n";
        f << "Generated: " << timestamp() << "\n\n";
        f << "Total trades   : " << combinedLog.totalTrades()   << "\n";
        f << "Total net P&L  : " << combinedLog.totalNetPnl()   << "\n";
        f << "Sharpe ratio   : " << combinedLog.sharpe()         << "\n";
        f << "Sortino ratio  : " << combinedLog.sortino()        << "\n";
        f << "Max drawdown   : " << combinedLog.maxDD()*100      << "%\n";
        f << "Calmar ratio   : " << combinedLog.calmar()         << "\n";
        f << "Win rate       : " << combinedLog.winRate()*100    << "%\n";
        f << "Profit factor  : " << combinedLog.profitFactor()   << "\n";
        f << "WF windows     : " << windows.size()               << "\n";
        std::cout << "[INFO] Wrote " << outDir << "/report.txt\n";
    }
}