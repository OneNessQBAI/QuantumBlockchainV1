import fastapi as _fastapi
from fastapi import HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List
from quantum_blockchain import QuantumBlockchain

# Initialize FastAPI and Quantum Blockchain
app = _fastapi.FastAPI()
blockchain = QuantumBlockchain()

# Define models for transaction and smart contract requests
class Transaction(BaseModel):
    sender: str
    recipient: str
    amount: float

class SmartContractDeployment(BaseModel):
    contract_id: str
    contract_code: str

class SmartContractExecution(BaseModel):
    contract_id: str
    args: List[str] = []

# Endpoint for creating a new transaction
@app.post("/transactions/new")
def new_transaction(transaction: Transaction):
    index = blockchain.add_transaction(transaction.sender, transaction.recipient, transaction.amount)
    return {"message": f"Transaction will be added to Block {index}"}

# Endpoint for mining a new block
@app.get("/mine")
def mine_block():
    block = blockchain.mine_block()
    response = {
        "message": "New Block Forged",
        "index": block['index'],
        "transactions": block['data'],
        "proof": block['proof'],
        "previous_hash": block['previous_hash'],
        "quantum_signature": block['quantum_signature']
    }
    return response

# Endpoint for retrieving the blockchain
@app.get("/chain")
def get_chain():
    response = {
        "chain": blockchain.chain,
        "length": len(blockchain.chain),
    }
    return response

# Endpoint for validating the blockchain
@app.get("/validate")
def validate_chain():
    is_valid = blockchain.is_chain_valid()
    return {"valid": is_valid}

# Endpoint for retrieving the quantum state
@app.get("/quantum_state")
def get_quantum_state():
    state_vector = blockchain.get_quantum_state()
    return {"quantum_state": state_vector}

# Endpoint for deploying a smart contract
@app.post("/smart_contract/deploy")
def deploy_smart_contract(deployment: SmartContractDeployment):
    try:
        blockchain.create_smart_contract(deployment.contract_id, deployment.contract_code)
        return {"message": f"Smart contract '{deployment.contract_id}' deployed successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint for executing a smart contract
@app.post("/smart_contract/execute")
def execute_smart_contract(execution: SmartContractExecution):
    try:
        result = blockchain.execute_smart_contract(execution.contract_id, *execution.args)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint for listing all deployed smart contracts
@app.get("/smart_contracts")
def list_smart_contracts():
    return {"smart_contracts": list(blockchain.smart_contracts.keys())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
