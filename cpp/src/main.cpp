// ─── StatArb - Statistical Arbitrage / Pairs Trading System ──────────────────
// Entry point: parses CLI args, loads data, runs cointegration screening,
// walk-forward backtest, and writes full tearsheet + CSV exports.
//
// Build:   cmake -B build && cmake --build build
// Usage:   ./statarb --data data/prices --out data/results --mode wf
//          ./statarb --mode pair --tickerA HDFCBANK --tickerB ICICIBANK
//          ./statarb --mode screen --tickers TCS,INFY,WIPRO,HCLTECH
// ─────────────────────────────────────────────────────────────────────────────
#include "DataFetcher.h"
#include "CointegrationEngine.h"
#include "KalmanFilter.h"
#include "SignalGenerator.h"
#include "PositionSizer.h"
#include "RiskManager.h"
#include "WalkForwardEngine.h"
#include "ReportGenerator.h"
#include "TradeLog.h"
#include <iostream>
#include <string>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

// ─── CLI config ───────────────────────────────────────────────────────────────
struct CliConfig {
    std::string mode        = "wf";       // wf | pair | screen | generate
    std::string dataDir     = "data/prices";
    std::string outDir      = "data/results";
    std::string tickerA;
    std::string tickerB;
    std::vector<std::string> tickers;
    int    trainDays        = 252;
    int    testDays         = 63;
    double pValueThreshold  = 0.05;
    double initialNAV       = 1'000'000.0;
    double entryZ           = 2.0;
    double exitZ            = 0.5;
    double stopZ            = 3.5;
    int    zWindow          = 30;
    bool   verbose          = true;
    bool   expandingWindow  = false;
    bool   generateData     = false;
    int    syntheticBars    = 600;
};

void printUsage(const char* prog) {
    std::cout <<
        "\nUsage: " << prog << " [OPTIONS]\n\n"
        "Modes:\n"
        "  --mode wf        Walk-forward backtest on full universe (default)\n"
        "  --mode pair      Single-pair deep analysis\n"
        "  --mode screen    Cointegration-only screen, no backtest\n"
        "  --mode generate  Generate synthetic CSV data for testing\n\n"
        "Options:\n"
        "  --data    <dir>     Price CSV directory (default: data/prices)\n"
        "  --out     <dir>     Output directory    (default: data/results)\n"
        "  --tickers A,B,C     Comma-separated ticker list\n"
        "  --tickerA <t>       First ticker for --mode pair\n"
        "  --tickerB <t>       Second ticker for --mode pair\n"
        "  --train   <N>       Training window in bars (default: 252)\n"
        "  --test    <N>       Test window in bars    (default: 63)\n"
        "  --nav     <N>       Initial portfolio NAV  (default: 1000000)\n"
        "  --entryZ  <f>       Z-score entry threshold (default: 2.0)\n"
        "  --exitZ   <f>       Z-score exit threshold  (default: 0.5)\n"
        "  --stopZ   <f>       Z-score stop threshold  (default: 3.5)\n"
        "  --zwin    <N>       Z-score rolling window  (default: 30)\n"
        "  --pval    <f>       ADF p-value threshold   (default: 0.05)\n"
        "  --expand            Use expanding (vs rolling) train window\n"
        "  --generate  <N>     Synthesise N bars of test data and exit\n"
        "  --quiet             Suppress per-bar output\n"
        "  --help              Show this message\n\n"
        "Examples:\n"
        "  ./statarb --mode generate --tickers HDFC,ICICI,TCS,INFY\n"
        "  ./statarb --mode wf --tickers HDFC,ICICI,TCS,INFY\n"
        "  ./statarb --mode pair --tickerA HDFC --tickerB ICICI\n";
}

CliConfig parseArgs(int argc, char** argv) {
    CliConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i+1 >= argc) {
                std::cerr << "[ERR] Missing value for " << a << "\n";
                printUsage(argv[0]);
                std::exit(1);
            }
            return std::string(argv[++i]);
        };
        try {
            if      (a == "--help")    { printUsage(argv[0]); std::exit(0); }
            else if (a == "--mode")    cfg.mode    = next();
            else if (a == "--data")    cfg.dataDir = next();
            else if (a == "--out")     cfg.outDir  = next();
            else if (a == "--tickerA") cfg.tickerA = next();
            else if (a == "--tickerB") cfg.tickerB = next();
            else if (a == "--train")   cfg.trainDays = std::stoi(next());
            else if (a == "--test")    cfg.testDays  = std::stoi(next());
            else if (a == "--nav")     cfg.initialNAV= std::stod(next());
            else if (a == "--entryZ")  cfg.entryZ    = std::stod(next());
            else if (a == "--exitZ")   cfg.exitZ     = std::stod(next());
            else if (a == "--stopZ")   cfg.stopZ     = std::stod(next());
            else if (a == "--zwin")    cfg.zWindow   = std::stoi(next());
            else if (a == "--pval")    cfg.pValueThreshold = std::stod(next());
            else if (a == "--expand")  cfg.expandingWindow = true;
            else if (a == "--quiet")   cfg.verbose   = false;
            else if (a == "--generate") { cfg.generateData = true;
                                          cfg.syntheticBars = std::stoi(next()); }
            else if (a == "--tickers") {
                std::string list = next();
                std::istringstream ss(list);
                std::string tok;
                while (std::getline(ss, tok, ','))
                    if (!tok.empty()) cfg.tickers.push_back(tok);
            } else {
                std::cerr << "[ERR] Unknown option: " << a << "\n";
                printUsage(argv[0]);
                std::exit(1);
            }
        } catch (const std::exception& e) {
            std::cerr << "[ERR] Invalid value for " << a << ": " << e.what() << "\n";
            std::exit(1);
        }
    }
    return cfg;
}

// ─── Build BacktestConfig from CliConfig ──────────────────────────────────────
BacktestConfig buildBacktestConfig(const CliConfig& cli) {
    BacktestConfig bc;
    bc.trainWindowBars  = cli.trainDays;
    bc.testWindowBars   = cli.testDays;
    bc.stepBars         = cli.testDays;
    bc.expandingWindow  = cli.expandingWindow;
    bc.initialNAV       = cli.initialNAV;
    bc.verbose          = cli.verbose;
    bc.optimiseDelta    = true;

    bc.signalCfg.entryZScore  = cli.entryZ;
    bc.signalCfg.exitZScore   = cli.exitZ;
    bc.signalCfg.stopZScore   = cli.stopZ;
    bc.signalCfg.zScoreWindow = cli.zWindow;
    bc.signalCfg.maxHoldBars  = cli.testDays;

    bc.sizingCfg.portfolioNAV = cli.initialNAV;

    return bc;
}

// ─── Mode: generate synthetic data ───────────────────────────────────────────
void modeGenerate(const CliConfig& cli) {
    std::cout << "[INFO] Generating synthetic CSV data in " << cli.dataDir << "\n";
    fs::create_directories(cli.dataDir);

    std::vector<std::string> tickers = cli.tickers.empty()
        ? std::vector<std::string>{"HDFC", "ICICI", "TCS", "INFY",
                                    "WIPRO", "HCLTECH"}
        : cli.tickers;

    // Generate a base series + correlated partner for each pair
    unsigned seed = 42;
    double baseStart = 1000.0;
    for (size_t i = 0; i < tickers.size(); ++i) {
        double vol   = 0.012 + (i % 3) * 0.003;
        double drift = 0.00005 * (1 + (int)(i % 2));
        DataFetcher::writeSyntheticCSV(
            cli.dataDir + "/" + tickers[i] + ".csv",
            tickers[i], cli.syntheticBars,
            baseStart * (0.8 + 0.1 * i), drift, vol, seed + (unsigned)i);
    }
    std::cout << "[INFO] Generated " << tickers.size()
              << " synthetic price series.\n";
    std::cout << "       Now run: ./statarb --mode wf --tickers "
              << tickers[0];
    for (size_t i = 1; i < tickers.size(); ++i) std::cout << "," << tickers[i];
    std::cout << "\n";
}

// ─── Mode: cointegration screen only ─────────────────────────────────────────
void modeScreen(const CliConfig& cli, DataFetcher& fetcher) {
    std::cout << "\n[INFO] Running cointegration screen...\n\n";

    CointegrationEngine coint;
    coint.pValueThreshold = cli.pValueThreshold;

    std::unordered_map<std::string, std::vector<double>> prices;
    for (const auto& t : cli.tickers) {
        if (fetcher.hasData(t))
            prices[t] = fetcher.getSeries(t).close;
    }

    auto candidates = coint.screenAllPairs(cli.tickers, prices, true);

    std::cout << "\n══════ TOP COINTEGRATED PAIRS ══════\n";
    std::cout << std::left
              << std::setw(16) << "Pair"
              << std::setw(8)  << "β"
              << std::setw(10) << "ADF Stat"
              << std::setw(8)  << "p-val"
              << std::setw(8)  << "Hurst"
              << std::setw(8)  << "HL(d)"
              << "\n" << repeatStr("─", 58) << "\n";
    for (size_t i = 0; i < std::min(candidates.size(), (size_t)10); ++i) {
        const auto& c = candidates[i];
        std::cout << std::left
                  << std::setw(16) << (c.tickerA + "/" + c.tickerB)
                  << std::fixed << std::setprecision(4)
                  << std::setw(8)  << c.beta
                  << std::setw(10) << c.adfStat
                  << std::setw(8)  << c.pValue
                  << std::setw(8)  << c.hurstExponent
                  << std::setw(8)  << (int)c.halfLifeDays
                  << "\n";
    }
    std::cout << "\n";
}

// ─── Mode: single pair deep analysis ─────────────────────────────────────────
void modePair(const CliConfig& cli, DataFetcher& fetcher) {
    if (cli.tickerA.empty() || cli.tickerB.empty()) {
        std::cerr << "[ERR] --tickerA and --tickerB required for --mode pair\n";
        return;
    }
    if (!fetcher.hasData(cli.tickerA) || !fetcher.hasData(cli.tickerB)) {
        std::cerr << "[ERR] One or both tickers not loaded.\n";
        return;
    }

    auto [serA, serB] = fetcher.alignSeries(cli.tickerA, cli.tickerB);
    std::cout << "\n[INFO] Aligned series: " << serA.size() << " bars\n\n";

    // Cointegration test
    CointegrationEngine coint;
    coint.pValueThreshold = cli.pValueThreshold;
    auto pc = coint.engleGranger(cli.tickerA, cli.tickerB,
                                   serA.close, serB.close);

    std::cout << "══ Engle-Granger Cointegration Test ══\n";
    std::cout << "  Pair         : " << cli.tickerA << "/" << cli.tickerB << "\n";
    std::cout << "  β (OLS)      : " << pc.beta       << "\n";
    std::cout << "  ADF stat     : " << pc.adfStat    << "\n";
    std::cout << "  p-value      : " << pc.pValue
              << " (" << (pc.cointegrated ? "PASS" : "FAIL") << ")\n";
    std::cout << "  Hurst exp    : " << pc.hurstExponent << "\n";
    std::cout << "  Half-life    : " << (int)pc.halfLifeDays << " bars\n";
    std::cout << "  Cointegrated : " << (pc.cointegrated ? "YES ✓" : "NO ✗") << "\n\n";

    // Walk-forward backtest on this pair
    BacktestConfig bc = buildBacktestConfig(cli);
    WalkForwardEngine engine(bc);
    auto log = engine.runSinglePair(cli.tickerA, cli.tickerB,
                                     serA.close, serB.close, true);

    ReportGenerator rg;
    rg.printTearsheet(log, "SINGLE PAIR: " + cli.tickerA + "/" + cli.tickerB);

    fs::create_directories(cli.outDir);
    std::vector<WindowResult> empty;
    rg.exportAll(log, empty, cli.outDir);
}

// ─── Mode: walk-forward on full universe ─────────────────────────────────────
void modeWalkForward(const CliConfig& cli, DataFetcher& fetcher) {
    std::cout << "\n[INFO] Walk-forward backtest on " << cli.tickers.size()
              << " tickers\n";

    std::unordered_map<std::string, std::vector<double>> prices;
    std::vector<std::string> loadedTickers;
    for (const auto& t : cli.tickers) {
        if (fetcher.hasData(t)) {
            prices[t] = fetcher.getSeries(t).close;
            loadedTickers.push_back(t);
        }
    }

    if (loadedTickers.size() < 2) {
        std::cerr << "[ERR] Need at least 2 loaded tickers for walk-forward.\n";
        return;
    }

    BacktestConfig bc = buildBacktestConfig(cli);
    WalkForwardEngine engine(bc);
    TradeLog combinedLog;
    auto windows = engine.run(loadedTickers, prices, combinedLog);

    ReportGenerator rg;
    rg.printTearsheet(combinedLog, "WALK-FORWARD COMBINED TEARSHEET");
    rg.printWalkForwardSummary(windows);
    rg.printOverfitCheck(windows);

    fs::create_directories(cli.outDir);
    rg.exportAll(combinedLog, windows, cli.outDir);
    std::cout << "[INFO] All results written to " << cli.outDir << "/\n";
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    CliConfig cli = parseArgs(argc, argv);

    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Statistical Arbitrage / Pairs Trading System  v1.0     ║\n";
    std::cout << "║   Gale-Shapley × Kalman × Engle-Granger × Walk-Forward   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";

    // ── Generate mode: no data loading needed ──
    if (cli.mode == "generate" || cli.generateData) {
        modeGenerate(cli);
        return 0;
    }

    // ── Load price data ──
    if (cli.tickers.empty() && !cli.tickerA.empty() && !cli.tickerB.empty()) {
        cli.tickers = {cli.tickerA, cli.tickerB};
    }
    if (cli.tickers.empty()) {
        std::cerr << "[ERR] No tickers specified. Use --tickers A,B,C or "
                     "--tickerA A --tickerB B\n";
        printUsage(argv[0]);
        return 1;
    }

    DataFetcher fetcher(cli.dataDir);
    fetcher.loadTickers(cli.tickers);

    bool anyLoaded = false;
    for (const auto& t : cli.tickers)
        if (fetcher.hasData(t)) { anyLoaded = true; break; }

    if (!anyLoaded) {
        std::cerr << "[ERR] No tickers loaded. Run --mode generate first to "
                     "create synthetic data.\n";
        return 1;
    }

    // ── Dispatch mode ──
    if      (cli.mode == "screen") modeScreen   (cli, fetcher);
    else if (cli.mode == "pair")   modePair      (cli, fetcher);
    else                           modeWalkForward(cli, fetcher);

    return 0;
}