
package autoeasoran.zmqonly;

simple TwinMetricsAgent
{
    parameters:
        string pubBind = default("tcp://127.0.0.1:5556");
        double publishInterval = default(10);
        int numGnbs = default(7);
        string twinId = default("simu5g-twin");
        bool echoToLog = default(false);
        @display("i=block/process");
}
