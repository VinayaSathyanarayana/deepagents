from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
import asyncio

# Transaction Agent
class TransactionAgent(Agent):
    class TransactionBehaviour(CyclicBehaviour):
        async def run(self):
            print("TransactionAgent: Sending transaction...")
            msg = Message(to="fraud@localhost")  # FraudDetectionAgent
            msg.body = "TXN: ₹5000 to merchant123"
            await self.send(msg)
            await asyncio.sleep(5)

    async def setup(self):
        self.add_behaviour(self.TransactionBehaviour())

# Fraud Detection Agent
class FraudDetectionAgent(Agent):
    class MonitorBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print(f"FraudDetectionAgent: Received - {msg.body}")
                if "₹5000" in msg.body:
                    print("FraudDetectionAgent: Flagged as suspicious!")
                    alert = Message(to="compliance@localhost")
                    alert.body = "Suspicious transaction flagged"
                    await self.send(alert)

    async def setup(self):
        self.add_behaviour(self.MonitorBehaviour())

# Compliance Agent
class ComplianceAgent(Agent):
    class CheckBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print(f"ComplianceAgent: Received alert - {msg.body}")
                print("ComplianceAgent: Reviewing transaction against RBI norms...")

    async def setup(self):
        self.add_behaviour(self.CheckBehaviour())

# Launch agents
txn_agent = TransactionAgent("txn@localhost", "txnpass")
fraud_agent = FraudDetectionAgent("fraud@localhost", "fraudpass")
compliance_agent = ComplianceAgent("compliance@localhost", "compliancepass")

async def main():
    await txn_agent.start()
    await fraud_agent.start()
    await compliance_agent.start()
    await asyncio.sleep(20)
    await txn_agent.stop()
    await fraud_agent.stop()
    await compliance_agent.stop()

asyncio.run(main())