#ifndef __ORCH_SHIM_H_
#define __ORCH_SHIM_H_

#include <omnetpp.h>

#include <string>

class ControllableGnbPhy;

class OrchestrationShim :
    public omnetpp::cSimpleModule
{
  private:
    std::string repBind;
    std::string statusPubBind;

    bool echoToLog = true;

    omnetpp::cMessage* tick = nullptr;

    void* zmqCtx = nullptr;
    void* zmqRep = nullptr;
    void* zmqPub = nullptr;

    ControllableGnbPhy* resolveGnbPhy(
        const std::string& bsId,
        int& cellIndex
    );

  protected:
    virtual void initialize() override;

    virtual void handleMessage(
        omnetpp::cMessage* message
    ) override;

    virtual void finish() override;

    void ensureSockets();
    void pollOnce();

    std::string applyPlan(
        const std::string& request
    );

    void publishStatus(
        const std::string& event,
        const std::string& payload
    );
};

#endif