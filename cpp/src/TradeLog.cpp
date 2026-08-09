#include "TradeLog.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

void TradeLog::addTrade(const Trade& t) {
    trades_.push_back(t);
    totalPnl_   += t.pnl;
    totalGross_ += t.grossPnl;
    totalCosts_ += t.transactionCost;
    sumHold_    += (t.exitBar - t.entryBar);

    if (t.pnl > 0)      { ++wins_;   sumWinPnl_  += t.pnl; }
    else if (t.pnl < 0) { ++losses_; sumLossPnl_ += std::abs(t.pnl); }
    // pnl == 0: break-even trade, counted in totalTrades() but not as a win or a loss
    if (t.stoppedOut) ++stopped_;

    dailyPnl_.push_back(t.pnl);
}

void TradeLog::markToMarket(double unrealisedPnl, int /*bar*/) {
    dailyPnl_.push_back(unrealisedPnl);
}

void TradeLog::finalise(double initialNAV) {
    if (dailyPnl_.empty()) return;

    // Build equity curve
    equityCurve_.resize(dailyPnl_.size() + 1);
    equityCurve_[0] = initialNAV;
    for (size_t i = 0; i < dailyPnl_.size(); ++i)
        equityCurve_[i+1] = equityCurve_[i] + dailyPnl_[i];

    // Daily return series
    std::vector<double> dailyRets(dailyPnl_.size());
    for (size_t i = 0; i < dailyPnl_.size(); ++i)
        dailyRets[i] = (equityCurve_[i] > 1e-6) ? dailyPnl_[i] / equityCurve_[i] : 0.0;

    sharpe_    = Stats::sharpeRatio(dailyRets);
    sortino_   = Stats::sortinoRatio(dailyRets);
    maxDD_     = Stats::maxDrawdown(equityCurve_);
    calmar_    = Stats::calmarRatio(equityCurve_, dailyRets);
    annReturn_ = Stats::mean(dailyRets) * 252.0;
    var95_     = Stats::percentile(dailyPnl_, 0.05);   // 5th percentile
}

double TradeLog::winRate() const {
    int n = wins_ + losses_;
    return n == 0 ? 0.0 : (double)wins_ / n;
}

double TradeLog::avgWinPnl() const {
    return wins_ == 0 ? 0.0 : sumWinPnl_ / wins_;
}

double TradeLog::avgLossPnl() const {
    return losses_ == 0 ? 0.0 : -sumLossPnl_ / losses_;
}

double TradeLog::profitFactor() const {
    if (sumLossPnl_ < 1e-6)
        return sumWinPnl_ > 1e-6
             ? std::numeric_limits<double>::infinity()  // wins, no losses at all
             : 0.0;                                     // no trades / no wins either
    return sumWinPnl_ / sumLossPnl_;
}

double TradeLog::avgHoldingBars() const {
    return trades_.empty() ? 0.0 : sumHold_ / trades_.size();
}

void TradeLog::exportTradesCSV(const std::string& path) const {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ERR] Cannot write " << path << "\n"; return;
    }
    f << "ticker_a,ticker_b,direction,entry_bar,exit_bar,hold_bars,"
      << "entry_spread,exit_spread,entry_z,exit_z,"
      << "hedge_ratio,position_size,gross_pnl,cost,net_pnl,stopped\n";
    for (const auto& t : trades_) {
        std::string dir = (t.direction == TradeDirection::LONG_A_SHORT_B)
                         ? "LONG_A" : "SHORT_A";
        f << t.tickerA << "," << t.tickerB << "," << dir << ","
          << t.entryBar << "," << t.exitBar << ","
          << (t.exitBar - t.entryBar) << ","
          << std::fixed << std::setprecision(6)
          << t.entrySpread << "," << t.exitSpread << ","
          << t.entryZ << "," << t.exitZ << ","
          << t.hedgeRatio << "," << t.positionSize << ","
          << t.grossPnl << "," << t.transactionCost << ","
          << t.pnl << "," << (t.stoppedOut ? "1" : "0") << "\n";
    }
    std::cout << "[INFO] Wrote " << path << " (" << trades_.size() << " trades)\n";
}

void TradeLog::exportEquityCSV(const std::string& path) const {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ERR] Cannot write " << path << "\n"; return;
    }
    f << "bar,equity,daily_pnl\n";
    for (size_t i = 0; i < equityCurve_.size(); ++i) {
        double pnl = (i == 0) ? 0.0 : dailyPnl_[i-1];
        f << i << "," << std::fixed << std::setprecision(2)
          << equityCurve_[i] << "," << pnl << "\n";
    }
    std::cout << "[INFO] Wrote " << path << "\n";
}

void TradeLog::printSummary() const {
    auto sep = repeatStr("─", 56);
    std::cout << "\n╔" << repeatStr("═", 54) << "╗\n";
    std::cout << "║         TRADE LOG SUMMARY                            ║\n";
    std::cout << "╚" << repeatStr("═", 54) << "╝\n";
    std::cout << sep << "\n";
    std::cout << std::left;
    auto row = [](const std::string& lbl, auto val, const std::string& unit="") {
        std::cout << "  " << std::setw(28) << lbl << ": "
                  << std::fixed << std::setprecision(4) << val
                  << " " << unit << "\n";
    };
    row("Total trades",        totalTrades());
    row("Winning trades",      winningTrades());
    row("Losing trades",       losingTrades());
    row("Stopped-out trades",  stoppedTrades());
    row("Win rate",            winRate()*100, "%");
    row("Avg win P&L",         avgWinPnl());
    row("Avg loss P&L",        avgLossPnl());
    row("Profit factor",       profitFactor());
    row("Avg hold (bars)",     avgHoldingBars());
    std::cout << sep << "\n";
    row("Total gross P&L",     totalGrossPnl());
    row("Total costs",         totalCosts());
    row("Total net P&L",       totalNetPnl());
    std::cout << sep << "\n";
    row("Annualised return",   annReturn()*100, "%");
    row("Sharpe ratio",        sharpe());
    row("Sortino ratio",       sortino());
    row("Max drawdown",        maxDD()*100, "%");
    row("Calmar ratio",        calmar());
    row("95% VaR (daily)",     var95());
    std::cout << sep << "\n\n";
}

void TradeLog::clear() {
    trades_.clear(); equityCurve_.clear(); dailyPnl_.clear();
    wins_ = losses_ = stopped_ = 0;
    totalPnl_ = totalGross_ = totalCosts_ = 0;
    sumWinPnl_ = sumLossPnl_ = sumHold_ = 0;
    sharpe_ = sortino_ = maxDD_ = calmar_ = annReturn_ = var95_ = 0;
}