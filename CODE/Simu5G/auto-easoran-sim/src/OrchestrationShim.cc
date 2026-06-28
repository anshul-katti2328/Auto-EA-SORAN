#include "OrchestrationShim.h"
#include "ControllableGnbPhy.h"
#include "ShimRegistry.h"

#include <zmq.h>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <cctype>

using namespace omnetpp;

Define_Module(OrchestrationShim);
ControllableGnbPhy*
OrchestrationShim::resolveGnbPhy(
    const std::string& bsId,
    int& cellIndex
) {
    cellIndex = -1;

    std::size_t consumedCharacters = 0;
    int numericBsId = 0;

    try {
        numericBsId = std::stoi(
            bsId,
            &consumedCharacters
        );
    }
    catch (...) {
        return nullptr;
    }

    if (consumedCharacters != bsId.size()) {
        return nullptr;
    }

    // External IDs 1001..1030 map to gnb[0]..gnb[29].
    cellIndex = numericBsId - 1001;

    cModule* network =
        getParentModule();

    if (network == nullptr) {
        cellIndex = -1;
        return nullptr;
    }

    const int numberOfGnbs =
        network->getSubmoduleVectorSize(
            "gnb"
        );

    if (cellIndex < 0 ||
        cellIndex >= numberOfGnbs) {

        cellIndex = -1;
        return nullptr;
    }

    cModule* gnb =
        network->getSubmodule(
            "gnb",
            cellIndex
        );

    cModule* cellularNic =
        gnb != nullptr
            ? gnb->getSubmodule(
                "cellularNic"
            )
            : nullptr;

    cModule* phyModule =
        cellularNic != nullptr
            ? cellularNic->getSubmodule(
                "phy"
            )
            : nullptr;

    return dynamic_cast<
        ControllableGnbPhy*
    >(phyModule);
}

void OrchestrationShim::initialize() {
    repBind       = par("repBind").stdstringValue();
    statusPubBind = par("statusPubBind").stdstringValue();
    echoToLog     = par("echoToLog").boolValue();

    tick = new cMessage("poll");
    scheduleAt(simTime() + 0.05, tick);
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

std::string OrchestrationShim::applyPlan(
    const std::string& request
) {
    auto readNumber = [&request](
        const std::string& key,
        double& output
    ) -> bool {
        const std::string token = "\"" + key + "\"";
        const auto keyPos = request.find(token);

        if (keyPos == std::string::npos)
            return false;

        const auto colonPos = request.find(':', keyPos);
        if (colonPos == std::string::npos)
            return false;

        const auto endPos = request.find_first_of(
            ",}", colonPos + 1
        );

        std::string value = request.substr(
            colonPos + 1,
            endPos - colonPos - 1
        );

        value.erase(
            std::remove_if(
                value.begin(),
                value.end(),
                ::isspace
            ),
            value.end()
        );

        try {
            output = std::stod(value);
            return true;
        }
        catch (...) {
            return false;
        }
    };

    auto readString = [&request](
        const std::string& key,
        std::string& output
    ) -> bool {
        const std::string token = "\"" + key + "\"";
        const auto keyPos = request.find(token);

        if (keyPos == std::string::npos)
            return false;

        const auto colonPos = request.find(':', keyPos);
        const auto quoteStart = request.find(
            '"', colonPos + 1
        );
        const auto quoteEnd = request.find(
            '"', quoteStart + 1
        );

        if (
            colonPos == std::string::npos ||
            quoteStart == std::string::npos ||
            quoteEnd == std::string::npos
        ) {
            return false;
        }

        output = request.substr(
            quoteStart + 1,
            quoteEnd - quoteStart - 1
        );

        return true;
    };

    auto readEsModes = [&request](
        std::array<int, 6>& output
    ) -> bool {
        const auto keyPos = request.find("\"es_modes\"");

        if (keyPos == std::string::npos)
            return false;

        const auto arrayStart = request.find('[', keyPos);
        const auto arrayEnd = request.find(']', arrayStart);

        if (
            arrayStart == std::string::npos ||
            arrayEnd == std::string::npos
        ) {
            return false;
        }

        const std::string arrayText = request.substr(
            arrayStart + 1,
            arrayEnd - arrayStart - 1
        );

        std::stringstream stream(arrayText);
        std::string item;
        int index = 0;

        while (
            std::getline(stream, item, ',') &&
            index < 6
        ) {
            try {
                output[index] = (
                    std::stoi(item) != 0 ? 1 : 0
                );
            }
            catch (...) {
                output[index] = 0;
            }

            index++;
        }

        return index == 6;
    };

    std::string bsId;

    if (!readString("bs_id", bsId) || bsId.empty()) {
        std::stringstream reply;
        reply
            << "{"
            << "\"accepted\":false,"
            << "\"applied_count\":0,"
            << "\"error\":\"missing_or_invalid_bs_id\","
            << "\"applied_at_sim_s\":"
            << simTime().dbl()
            << "}";

        publishStatus(
            "action/rejected",
            reply.str()
        );

        return reply.str();
    }

        int cellIndex = -1;

    ControllableGnbPhy* gnbPhy =
        resolveGnbPhy(
            bsId,
            cellIndex
        );

    if (gnbPhy == nullptr) {
        std::stringstream reply;

        reply
            << "{"
            << "\"accepted\":false,"
            << "\"applied_count\":0,"
            << "\"bs_id\":\"" << bsId << "\","
            << "\"error\":\"gnb_phy_not_found\","
            << "\"applied_at_sim_s\":"
            << simTime().dbl()
            << "}";

        publishStatus(
            "action/rejected",
            reply.str()
        );

        return reply.str();
    }

    ShimRegistry::BsConfig config;

    const bool existingConfiguration =
        ShimRegistry::instance().getConfig(
            bsId,
            config
        );

    if (!existingConfiguration) {
        config.txPowerDbm =
            gnbPhy->getConfiguredTxPowerDbm();

        config.bandwidthMhz = 20.0;
        config.carrierCount = 1;
        config.esMode = "active";
        config.esModes = {0, 0, 0, 0, 0, 0};
    }

    double numericValue = 0.0;

    if (readNumber("delta_tx_db", numericValue)) {
        config.txPowerDbm += numericValue;
    }

    if (readNumber("tx_power_dbm", numericValue)) {
        config.txPowerDbm = numericValue;
    }

    if (readNumber("bandwidth_mhz", numericValue)) {
        config.bandwidthMhz = numericValue;
    }

    if (readNumber("carrier_count", numericValue)) {
        config.carrierCount = static_cast<int>(
            numericValue
        );
    }

    std::string stringValue;

    if (readString("es_mode", stringValue)) {
        config.esMode = stringValue;
    }

    const bool flagsProvided = readEsModes(
        config.esModes
    );

    // Derive flags when only a named ES mode was supplied.
    if (!flagsProvided) {
        config.esModes = {0, 0, 0, 0, 0, 0};

        if (config.esMode == "sleep") {
            config.esModes[0] = 1;
        }
        else if (config.esMode == "carrier_off") {
            config.esModes[1] = 1;
        }
        else if (config.esMode == "mimo_reduce") {
            config.esModes[2] = 1;
        }
        else if (config.esMode == "micro_dtx") {
            config.esModes[3] = 1;
        }
        else if (config.esMode == "bwp_adaptation") {
            config.esModes[4] = 1;
        }
        else if (
            config.esMode == "rf_chain_sleep" ||
            config.esMode == "light_sleep"
        ) {
            config.esModes[5] = 1;
        }
    }

    // Defensive bounds.
    config.txPowerDbm = std::clamp(
        config.txPowerDbm,
        5.0,
        43.0
    );

    if (config.bandwidthMhz > 0.0) {
        config.bandwidthMhz = std::clamp(
            config.bandwidthMhz,
            5.0,
            100.0
        );
    }

    config.carrierCount = std::clamp(
        config.carrierCount,
        1,
        4
    );
        // Apply supported controls to the real Simu5G PHY.
    gnbPhy->setConfiguredTxPowerDbm(
        config.txPowerDbm
    );

    const bool deepSleepRequested =
        config.esMode == "sleep" ||
        config.esModes[0] == 1;

    gnbPhy->setRadioSleeping(
        deepSleepRequested
    );

    ShimRegistry::instance().setConfig(
        bsId,
        config
    );

        std::stringstream reply;

    const bool esModePhysicallyApplied =
        config.esMode == "active" ||
        config.esMode == "normal" ||
        config.esMode == "sleep";

    reply
        << "{"
        << "\"accepted\":true,"
        << "\"applied_count\":1,"
        << "\"bs_id\":\"" << bsId << "\","
        << "\"cell_index\":" << cellIndex << ","
        << "\"tx_power_dbm\":"
        << config.txPowerDbm << ","
        << "\"physical_tx_power_dbm\":"
        << (
            deepSleepRequested
                ? -200.0
                : config.txPowerDbm
        )
        << ","
        << "\"radio_sleeping\":"
        << (
            deepSleepRequested
                ? "true"
                : "false"
        )
        << ","
        << "\"requested_bandwidth_mhz\":"
        << config.bandwidthMhz << ","
        << "\"physical_bandwidth_mhz\":20.0,"
        << "\"requested_carrier_count\":"
        << config.carrierCount << ","
        << "\"physical_carrier_count\":1,"
        << "\"es_mode\":\""
        << config.esMode << "\","
        << "\"es_mode_physically_applied\":"
        << (
            esModePhysicallyApplied
                ? "true"
                : "false"
        )
        << ","
        << "\"energy_mode_model\":\""
        << (
            esModePhysicallyApplied
                ? "simu5g_phy_and_energy_model"
                : "analytical_energy_model_only"
        )
        << "\","
        << "\"applied_at_sim_s\":"
        << simTime().dbl()
        << "}";

    publishStatus(
        "action/applied",
        reply.str()
    );

    return reply.str();
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
        scheduleAt(simTime() + 0.05, tick);
    }
}

void OrchestrationShim::finish() {
    if (tick)   { cancelAndDelete(tick); tick = nullptr; }
    if (zmqRep) { zmq_close(zmqRep); zmqRep = nullptr; }
    if (zmqPub) { zmq_close(zmqPub); zmqPub = nullptr; }
    if (zmqCtx) { zmq_ctx_term(zmqCtx); zmqCtx = nullptr; }
}
