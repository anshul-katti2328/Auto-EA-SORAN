package autoeasoran.zmqonly;

simple TwinMetricsAgent
{
    parameters:
        string pubBind = default("tcp://127.0.0.1:5556");
        double publishInterval = default(10);
        int numGnbs = default(30);
        string twinId = default("simu5g-twin-blr");
        bool echoToLog = default(false);
        @display("i=block/process");
}
