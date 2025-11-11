import asyncio
import argparse
import os
import google.generativeai as genai
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message

# Gemini setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# Shared state
transaction_log = []
fraud_log = []
validation_log = []

# Transaction Agent
class TransactionAgent(Agent):
    def __init__(self, jid, password, txn_count=10):
        super().__init__(jid, password)
        self.txn_count = txn_count

    class TransactionBehaviour(OneShotBehaviour):
        async def run(self):
            for i in range(self.agent.txn_count):
                txn = f"TXN {i+1}: ₹{1000 + i*500} to merchant{i}@2AM from IP 192.168.1.{i}"
                transaction_log.append(txn)
                print(f"TransactionAgent: Sending - {txn}")
                msg = Message(to="fraud@localhost", body=txn)
                await self.send(msg)
                await asyncio.sleep(1)

    async def setup(self):
        self.add_behaviour(self.TransactionBehaviour())

# Gemini-powered Fraud Agent
class GeminiFraudAgent(Agent):
    class FraudBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=5)
            if msg:
                print(f"GeminiFraudAgent: Received - {msg.body}")
                prompt = f"Assess fraud risk for this transaction:\n{msg.body}"
                response = model.generate_content(prompt)
                verdict = response.text.strip()
                print(f"GeminiFraudAgent: Gemini verdict - {verdict}")
                fraud_log.append((msg.body, verdict))
                if "suspicious" in verdict.lower():
                    alert = Message(to="validator@localhost", body=f"{msg.body} | Gemini: {verdict}")
                    await self.send(alert)

    async def setup(self):
        self.add_behaviour(self.FraudBehaviour())

# Validator Agent
class ValidatorAgent(Agent):
    class ValidateBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=5)
            if msg:
                print(f"ValidatorAgent: Received - {msg.body}")
                txn, verdict = msg.body.split(" | Gemini: ")
                valid = "Valid" if "₹1000" in txn else "Needs Review"
                print(f"ValidatorAgent: Validation result - {valid}")
                validation_log.append((txn, verdict, valid))

    async def setup(self):
        self.add_behaviour(self.ValidateBehaviour())

# Audit Agent
class AuditAgent(Agent):
    class AuditBehaviour(OneShotBehaviour):
        async def run(self):
            print("\nAuditAgent: Starting audit...")
            print(f"Total Transactions: {len(transaction_log)}")
            print(f"Fraud Assessments: {len(fraud_log)}")
            print(f"Validations: {len(validation_log)}")
            print("\n--- Fraud Log ---")
            for txn, verdict in fraud_log:
                print(f"{txn} → {verdict}")
            print("\n--- Validation Log ---")
            for txn, verdict, result in validation_log:
                print(f"{txn} → {verdict} → {result}")
            print("\nAuditAgent: Audit complete.")

    async def setup(self):
        await asyncio.sleep(15)  # Wait for other agents to finish
        self.add_behaviour(self.AuditBehaviour())

# CLI for transaction count
parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=10, help="Number of transactions to simulate")
args = parser.parse_args()

# Launch agents
txn_agent = TransactionAgent("txn@localhost", "txnpass", txn_count=args.count)
fraud_agent = GeminiFraudAgent("fraud@localhost", "fraudpass")
validator_agent = ValidatorAgent("validator@localhost", "validatorpass")
audit_agent = AuditAgent("audit@localhost", "auditpass")

async def main():
    await txn_agent.start()
    await fraud_agent.start()
    await validator_agent.start()
    await audit_agent.start()
    await asyncio.sleep(30)
    await txn_agent.stop()
    await fraud_agent.stop()
    await validator_agent.stop()
    await audit_agent.stop()

asyncio.run(main())