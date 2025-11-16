
#ifndef __TWIN_METRICS_AGENT_H_
#define __TWIN_METRICS_AGENT_H_

#include <omnetpp.h>
#include <string>

class TwinMetricsAgent : public omnetpp::cSimpleModule
{
  private:
    std::string pubBind, twinId;
    double publishInterval = 10.0;
    int numGnbs = 7;
    bool echoToLog = false;
    omnetpp::cMessage* tick = nullptr;
    void* zmqCtx = nullptr; void* zmqPub = nullptr;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(omnetpp::cMessage* msg) override;
    virtual void finish() override;

    void ensureSockets();
    void publishTick();
    std::string snapshotJson();
};
#endif
