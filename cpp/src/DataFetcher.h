#pragma once
#include "Utils.h"
#include <fstream>
#include <map>

// ─── DataFetcher ──────────────────────────────────────────────────────────────
// Reads OHLCV CSVs from data/prices/<TICKER>.csv
// Expected format: Date,Open,High,Low,Close,Volume,AdjClose
//   or simply:     Date,Close
// Stores all loaded tickers in a map for fast access.
// ─────────────────────────────────────────────────────────────────────────────
class DataFetcher {
public:
    explicit DataFetcher(const std::string& dataDir = "data/prices")
        : dataDir_(dataDir) {}

    // Load one ticker from CSV. Returns false on failure.
    bool loadTicker(const std::string& ticker);

    // Load multiple tickers at once
    void loadTickers(const std::vector<std::string>& tickers);

    // Get loaded series (throws if not found)
    const TimeSeries& getSeries(const std::string& ticker) const;

    // Align two series to the same date range (inner join on dates)
    std::pair<TimeSeries, TimeSeries>
    alignSeries(const std::string& a, const std::string& b) const;

    // All loaded tickers
    std::vector<std::string> loadedTickers() const;

    bool hasData(const std::string& ticker) const {
        return store_.count(ticker) > 0;
    }

    // Write a synthetic CSV for testing
    static void writeSyntheticCSV(const std::string& path,
                                   const std::string& ticker,
                                   int n = 500,
                                   double startPrice = 100.0,
                                   double drift = 0.0001,
                                   double vol = 0.01,
                                   unsigned seed = 42);

private:
    std::string dataDir_;
    std::map<std::string, TimeSeries> store_;

    static std::vector<std::string> splitCSV(const std::string& line);
    static std::string trim(const std::string& s);
    // Convert "YYYY-MM-DD" to serial day number
    static double dateToSerial(const std::string& date);
};