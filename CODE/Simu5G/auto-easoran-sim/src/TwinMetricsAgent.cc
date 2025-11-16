#include "TwinMetricsAgent.h"
#include "ShimRegistry.h"

#include <zmq.h>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <ctime>
#include <algorithm>

using namespace omnetpp;

Define_Module(TwinMetricsAgent);

// ISO8601 UTC timestamp for convenience
static std::string nowUtcIso() {
    auto t  = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(t);
    std::tm gmt{};
#ifdef _WIN32
    gmtime_s(&gmt, &tt);
#else
    gmtime_r(&tt, &gmt);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &gmt);
    return buf;
}

void TwinMetricsAgent::initialize() {
    pubBind         = par("pubBind").stdstringValue();
    publishInterval = par("publishInterval").doubleValue(); // seconds
    numGnbs         = par("numGnbs").intValue();
    twinId          = par("twinId").stdstringValue();
    echoToLog       = par("echoToLog").boolValue();

    tick = new cMessage("tick");
    scheduleAt(simTime() + publishInterval, tick);
}

void TwinMetricsAgent::ensureSockets() {
    if (!zmqCtx) zmqCtx = zmq_ctx_new();
    if (!zmqPub) {
        zmqPub = zmq_socket(zmqCtx, ZMQ_PUB);
        if (zmq_bind(zmqPub, pubBind.c_str()) != 0)
            throw cRuntimeError("TwinMetricsAgent: PUB bind failed on %s", pubBind.c_str());
        EV_INFO << "TwinMetricsAgent: Metrics PUB bound to " << pubBind << "\n";
    }
}

std::string TwinMetricsAgent::snapshotJson() {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);

    ss << "{";
    ss << "\"topic\":\"all/tick\",";
    ss << "\"schema_version\":1,";
    ss << "\"twin_id\":\"" << twinId << "\",";
    ss << "\"sim_time_s\":" << simTime().dbl() << ",";
    ss << "\"wall_time_utc\":\"" << nowUtcIso() << "\",";
    ss << "\"transient\":false,";

    ss << "\"cells\":[";
    auto cfgs = ShimRegistry::instance().snapshot();

    bool first = true;
    for (int i = 0; i < numGnbs; ++i) {
        std::string bsid = std::to_string(1000 + i + 1);

        // Base profiles (synthetic but smooth & distinct per cell)
        double phase = 0.30 * i;
        double t     = SIMTIME_DBL(simTime());

        // PRB swings around 55% with ~±15% sinusoid
        double prb = 55.0 + 15.0 * std::sin(0.01 * t + phase);

        // Apply orchestrated config effects
        ShimRegistry::BsConfig cfg;
        auto it = cfgs.find(bsid);
        if (it != cfgs.end()) cfg = it->second;

        // Heuristic: increasing tx power slightly raises PRB (more users hang on);
        // keep within sane bounds.
        prb = std::clamp(prb + (cfg.txPowerDbm - 42.0) * 0.8, 5.0, 95.0);

        // Throughput inversely related to PRB saturation (very rough demo model)
        double dl = std::max(5.0, 120.0 * (1.0 - prb / 100.0));   // Mbps
        double ul = std::max(1.0,  20.0 * (1.0 - prb / 120.0));   // Mbps

        // Radio quality signals
        double sinr = 20.0 + 5.0 * std::sin(0.005 * t + 0.7 * i); // dB
        double rsrp = -95.0 + (cfg.txPowerDbm - 42.0) * 0.6;      // dBm

        int activeUes = 20 + (i * 3) % 15; // stable, cell-specific

        if (!first) ss << ",";
        first = false;

        ss << "{";
        ss << "\"bs_id\":\"" << bsid << "\",";
        ss << "\"prb_util_pct\":" << prb << ",";
        ss << "\"dl_mbps\":"      << dl  << ",";
        ss << "\"ul_mbps\":"      << ul  << ",";
        ss << "\"active_ues\":"   << activeUes << ",";
        ss << "\"avg_sinr_db\":"  << sinr << ",";
        ss << "\"avg_rsrp_dbm\":" << rsrp << ",";
        ss << "\"tx_power_dbm\":" << cfg.txPowerDbm << ",";
        ss << "\"es_mode\":\""    << cfg.esMode << "\"";
        ss << "}";
    }
    ss << "]"; // cells
    ss << "}";

    return ss.str();
}

void TwinMetricsAgent::publishTick() {
    std::string payload = snapshotJson();

    zmq_msg_t msg;
    zmq_msg_init_size(&msg, payload.size());
    memcpy(zmq_msg_data(&msg), payload.data(), payload.size());
    zmq_msg_send(&msg, zmqPub, 0);
    zmq_msg_close(&msg);

    if (echoToLog) EV_INFO << "TwinMetricsAgent: published metrics tick for " << numGnbs << " cells\n";
}

void TwinMetricsAgent::handleMessage(cMessage* msg) {
    if (msg == tick) {
        try {
            ensureSockets();
            publishTick();
        } catch (std::exception& e) {
            EV_WARN << "TwinMetricsAgent: publishTick error: " << e.what() << "\n";
        }
        scheduleAt(simTime() + publishInterval, tick);
    }
}

void TwinMetricsAgent::finish() {
    if (tick)   { cancelAndDelete(tick); tick = nullptr; }
    if (zmqPub) { zmq_close(zmqPub); zmqPub = nullptr; }
    if (zmqCtx) { zmq_ctx_term(zmqCtx); zmqCtx = nullptr; }
}
