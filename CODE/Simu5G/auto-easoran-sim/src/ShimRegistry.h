
#ifndef __SHIM_REGISTRY_H_
#define __SHIM_REGISTRY_H_

#include <map>
#include <string>
#include <mutex>

// A tiny, process-local singleton to share current per-gNB settings between the REP shim and the metrics publisher.
class ShimRegistry {
public:
    struct BsConfig {
        double txPowerDbm = 42.0;
        std::string esMode = "active";
        double cioBiasDb = 0.0;
    };

    static ShimRegistry& instance() { static ShimRegistry r; return r; }

    void setConfig(const std::string& bsId, const BsConfig& cfg) {
        std::lock_guard<std::mutex> g(mu); table[bsId] = cfg;
    }
    bool getConfig(const std::string& bsId, BsConfig& out) {
        std::lock_guard<std::mutex> g(mu);
        auto it = table.find(bsId);
        if (it==table.end()) return false; out = it->second; return true;
    }
    std::map<std::string,BsConfig> snapshot() {
        std::lock_guard<std::mutex> g(mu); return table;
    }
private:
    std::mutex mu;
    std::map<std::string,BsConfig> table;
};
#endif
