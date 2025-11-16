#include "OrchestrationShim.h"
#include "ShimRegistry.h"

#include <zmq.h>
#include <sstream>
#include <algorithm>
#include <ctime>

using namespace omnetpp;

Define_Module(OrchestrationShim);

void OrchestrationShim::initialize() {
    repBind       = par("repBind").stdstringValue();
    statusPubBind = par("statusPubBind").stdstringValue();
    echoToLog     = par("echoToLog").boolValue();

    tick = new cMessage("poll");
    scheduleAt(simTime() + 0.2, tick);
}

void OrchestrationShim::ensureSockets() {
    if (!zmqCtx) zmqCtx = zmq_ctx_new();
    if (!zmqRep) {
        zmqRep = zmq_socket(zmqCtx, ZMQ_REP);
        if (zmq_bind(zmqRep, repBind.c_str()) != 0)
            throw cRuntimeError("REP bind failed on %s", repBind.c_str());
        EV_INFO << "REP bound " << repBind << "\n";
    }
    if (!zmqPub) {
        zmqPub = zmq_socket(zmqCtx, ZMQ_PUB);
        if (zmq_bind(zmqPub, statusPubBind.c_str()) != 0)
            throw cRuntimeError("Status PUB bind failed on %s", statusPubBind.c_str());
        EV_INFO << "Status PUB bound " << statusPubBind << "\n";
    }
}

void OrchestrationShim::publishStatus(const std::string& event, const std::string& payload) {
    std::stringstream ss;
    ss << "{"
       << "\"event\":\"" << event << "\","
       << "\"sim_time_s\":" << simTime().dbl() << ","
       << "\"payload\":" << payload
       << "}";

    std::string s = ss.str();
    zmq_msg_t msg;
    zmq_msg_init_size(&msg, s.size());
    memcpy(zmq_msg_data(&msg), s.data(), s.size());
    zmq_msg_send(&msg, zmqPub, 0);
    zmq_msg_close(&msg);
}

std::string OrchestrationShim::applyPlan(const std::string& req) {
    // Minimal JSON scanning: look for "bs_id", "delta_tx_db", "tx_power_dbm", "es_mode"
    size_t pos = 0;
    int applied = 0;

    while (true) {
        auto b = req.find("\"bs_id\"", pos);
        if (b == std::string::npos) break;

        auto c  = req.find(':', b);
        auto q1 = req.find('"', c + 1);
        auto q2 = req.find('"', q1 + 1);
        if (q1 == std::string::npos || q2 == std::string::npos) break;
        std::string bsid = req.substr(q1 + 1, q2 - q1 - 1);

        ShimRegistry::BsConfig cfg;
        ShimRegistry::instance().getConfig(bsid, cfg);

        // delta_tx_db (number)
        auto dpos = req.find("\"delta_tx_db\"", q2);
        if (dpos != std::string::npos) {
            auto dc = req.find(':', dpos);
            auto de = req.find_first_of(",}", dc + 1);
            std::string delta = req.substr(dc + 1, de - dc - 1);
            delta.erase(std::remove_if(delta.begin(), delta.end(), ::isspace), delta.end());
            try { cfg.txPowerDbm += std::stod(delta); } catch (...) {}
        }

        // tx_power_dbm (number)
        auto tpos = req.find("\"tx_power_dbm\"", q2);
        if (tpos != std::string::npos) {
            auto tc = req.find(':', tpos);
            auto te = req.find_first_of(",}", tc + 1);
            std::string val = req.substr(tc + 1, te - tc - 1);
            val.erase(std::remove_if(val.begin(), val.end(), ::isspace), val.end());
            try { cfg.txPowerDbm = std::stod(val); } catch (...) {}
        }

        // es_mode (string)
        auto epos = req.find("\"es_mode\"", q2);
        if (epos != std::string::npos) {
            auto ec  = req.find(':', epos);
            auto eq1 = req.find('"', ec + 1);
            auto eq2 = req.find('"', eq1 + 1);
            if (eq1 != std::string::npos && eq2 != std::string::npos)
                cfg.esMode = req.substr(eq1 + 1, eq2 - eq1 - 1);
        }

        ShimRegistry::instance().setConfig(bsid, cfg);
        applied++;
        pos = q2 + 1;
    }

    std::stringstream rep;
    rep << "{"
        << "\"accepted\":true,"
        << "\"applied_count\":" << applied << ","
        << "\"applied_at_sim_s\":" << simTime().dbl()
        << "}";

    publishStatus("action/applied", rep.str());
    return rep.str();
}

void OrchestrationShim::pollOnce() {
    ensureSockets();

    zmq_msg_t req;
    zmq_msg_init(&req);
    int rc = zmq_msg_recv(&req, zmqRep, ZMQ_DONTWAIT);
    if (rc < 0) { zmq_msg_close(&req); return; }

    std::string body((char*)zmq_msg_data(&req), rc);
    zmq_msg_close(&req);

    if (echoToLog) EV_INFO << "Received plan: " << body << "\n";

    std::string reply = applyPlan(body);

    zmq_msg_t rep;
    zmq_msg_init_size(&rep, reply.size());
    memcpy(zmq_msg_data(&rep), reply.data(), reply.size());
    zmq_msg_send(&rep, zmqRep, 0);
    zmq_msg_close(&rep);
}

void OrchestrationShim::handleMessage(cMessage* msg) {
    if (msg == tick) {
        try { pollOnce(); } catch (...) {}
        scheduleAt(simTime() + 0.2, tick);
    }
}

void OrchestrationShim::finish() {
    if (tick)   { cancelAndDelete(tick); tick = nullptr; }
    if (zmqRep) { zmq_close(zmqRep); zmqRep = nullptr; }
    if (zmqPub) { zmq_close(zmqPub); zmqPub = nullptr; }
    if (zmqCtx) { zmq_ctx_term(zmqCtx); zmqCtx = nullptr; }
}
