#include "DataFetcher.h"
#include <sstream>
#include <stdexcept>
#include <random>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// ─── Helpers ─────────────────────────────────────────────────────────────────
std::string DataFetcher::trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b-a+1);
}

std::vector<std::string> DataFetcher::splitCSV(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(trim(tok));
    return out;
}

// Gregorian to serial (days since 1970-01-01)
double DataFetcher::dateToSerial(const std::string& date) {
    if (date.size() < 10) return 0.0;
    int y = std::stoi(date.substr(0,4));
    int m = std::stoi(date.substr(5,2));
    int d = std::stoi(date.substr(8,2));
    // Zeller-like serial
    int a = (14-m)/12, yr = y+4800-a, mo = m+12*a-3;
    int jdn = d + (153*mo+2)/5 + 365*yr + yr/4 - yr/100 + yr/400 - 32045;
    return static_cast<double>(jdn - 2440588); // days since 1970-01-01
}

// ─── loadTicker ───────────────────────────────────────────────────────────────
bool DataFetcher::loadTicker(const std::string& ticker) {
    std::string path = dataDir_ + "/" + ticker + ".csv";
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[WARN] DataFetcher: cannot open " << path << "\n";
        return false;
    }

    TimeSeries ts;
    ts.ticker = ticker;

    std::string line;
    bool header = true;
    int closeCol = -1, adjCloseCol = -1, dateCol = 0;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto fields = splitCSV(line);

        if (header) {
            // Detect column positions from header
            for (int i = 0; i < (int)fields.size(); ++i) {
                std::string h = fields[i];
                std::transform(h.begin(), h.end(), h.begin(), ::tolower);
                if (h == "adjclose" || h == "adj close" || h == "adj_close")
                    adjCloseCol = i;
                else if (h == "close")
                    closeCol = i;
                else if (h == "date")
                    dateCol = i;
            }
            if (adjCloseCol == -1 && closeCol == -1) {
                std::cerr << "[ERR] DataFetcher: no Close/AdjClose column in " << path << "\n";
                return false;
            }
            header = false;
            continue;
        }

        if ((int)fields.size() <= std::max(closeCol, adjCloseCol)) continue;

        try {
            double serial = dateToSerial(fields[dateCol]);
            int priceCol = (adjCloseCol != -1) ? adjCloseCol : closeCol;
            double price = std::stod(fields[priceCol]);
            if (price <= 0.0) continue;
            ts.dates.push_back(serial);
            ts.close.push_back(price);
        } catch (...) { continue; }
    }

    if (ts.close.size() < 50) {
        std::cerr << "[WARN] DataFetcher: too few rows for " << ticker
                  << " (" << ts.close.size() << ")\n";
        return false;
    }

    ts.computeLogReturns();
    store_[ticker] = std::move(ts);
    std::cout << "[INFO] Loaded " << ticker
              << " (" << store_[ticker].size() << " bars)\n";
    return true;
}

void DataFetcher::loadTickers(const std::vector<std::string>& tickers) {
    for (const auto& t : tickers) loadTicker(t);
}

const TimeSeries& DataFetcher::getSeries(const std::string& ticker) const {
    auto it = store_.find(ticker);
    if (it == store_.end())
        throw std::runtime_error("DataFetcher: ticker not loaded: " + ticker);
    return it->second;
}

std::vector<std::string> DataFetcher::loadedTickers() const {
    std::vector<std::string> out;
    for (const auto& [k,v] : store_) out.push_back(k);
    return out;
}

// ─── alignSeries ─────────────────────────────────────────────────────────────
std::pair<TimeSeries, TimeSeries>
DataFetcher::alignSeries(const std::string& a, const std::string& b) const {
    const auto& sa = getSeries(a);
    const auto& sb = getSeries(b);

    // Build date → index maps
    std::unordered_map<int, int> idxA, idxB;
    for (int i = 0; i < sa.size(); ++i)
        idxA[(int)sa.dates[i]] = i;
    for (int i = 0; i < sb.size(); ++i)
        idxB[(int)sb.dates[i]] = i;

    TimeSeries outA, outB;
    outA.ticker = a; outB.ticker = b;

    for (const auto& [d, ia] : idxA) {
        auto it = idxB.find(d);
        if (it == idxB.end()) continue;
        int ib = it->second;
        outA.dates.push_back(d);
        outA.close.push_back(sa.close[ia]);
        outB.dates.push_back(d);
        outB.close.push_back(sb.close[ib]);
    }

    // Sort by date
    std::vector<int> idx(outA.dates.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int x, int y){
        return outA.dates[x] < outA.dates[y];
    });

    TimeSeries sA, sB;
    sA.ticker = a; sB.ticker = b;
    for (int i : idx) {
        sA.dates.push_back(outA.dates[i]);
        sA.close.push_back(outA.close[i]);
        sB.dates.push_back(outB.dates[i]);
        sB.close.push_back(outB.close[i]);
    }
    sA.computeLogReturns();
    sB.computeLogReturns();

    return {sA, sB};
}

// ─── writeSyntheticCSV ────────────────────────────────────────────────────────
void DataFetcher::writeSyntheticCSV(const std::string& path,
                                     const std::string& ticker,
                                     int n, double startPrice,
                                     double drift, double vol,
                                     unsigned seed) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path);
    f << "Date,Close\n";
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(drift, vol);
    double price = startPrice;
    // Start from 2020-01-02
    int serial0 = (int)DataFetcher::dateToSerial("2020-01-02");
    int day = 0;
    for (int i = 0; i < n; ++i) {
        while (true) {
            int s = serial0 + day;
            ++day;
            // Skip weekends (simplified)
            int dow = (s + 4) % 7; // 0=Sun
            if (dow == 0 || dow == 6) continue;
            // Convert serial back to date
            int jd = s + 2440588;
            int a2 = jd + 32044;
            int b2 = (4*a2+3)/146097;
            int c2 = a2 - (146097*b2)/4;
            int d2 = (4*c2+3)/1461;
            int e2 = c2 - (1461*d2)/4;
            int m2 = (5*e2+2)/153;
            int dy = e2-(153*m2+2)/5+1;
            int mo = m2+3-12*(m2/10);
            int yr = 100*b2+d2-4800+m2/10;
            char buf[16];
            snprintf(buf, sizeof(buf), "%04d-%02d-%02d", yr, mo, dy);
            price *= std::exp(nd(rng));
            f << buf << "," << std::fixed << std::setprecision(4) << price << "\n";
            break;
        }
    }
    std::cout << "[INFO] Wrote synthetic CSV for " << ticker << ": "
               << path << " (" << n << " bars)\n";
}