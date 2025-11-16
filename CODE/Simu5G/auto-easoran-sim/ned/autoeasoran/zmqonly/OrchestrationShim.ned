
package autoeasoran.zmqonly;

simple OrchestrationShim
{
    parameters:
        string repBind = default("tcp://127.0.0.1:6000");
        string statusPubBind = default("tcp://127.0.0.1:5557");
        bool echoToLog = default(true);
        @display("i=block/box");
}
