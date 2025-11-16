#ifndef __ORCH_SHIM_H_
#define __ORCH_SHIM_H_

#include <omnetpp.h>
#include <string>
#include <vector>

class OrchestrationShim : public omnetpp::cSimpleModule
{
  private:
    std::string repBind, statusPubBind;
    bool echoToLog = true;
    omnetpp::cMessage* tick = nullptr;
    void* zmqCtx = nullptr; void* zmqRep = nullptr; void* zmqPub = nullptr;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(omnetpp::cMessage* msg) override;
    virtual void finish() override;

    void ensureSockets();
    void pollOnce();
    std::string applyPlan(const std::string& req);
    void publishStatus(const std::string& event, const std::string& payload);
};
#endif
