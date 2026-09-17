import time
import hashlib
import json
from typing import List, Dict, Optional

class TransportationTrustLedger:
    """
    Transportation Trust Layer (Optional Blockchain Audit Module):
    Maintains an immutable cryptographically chained audit log of all
    route assignments, incident injections, and optimization interventions.
    """
    def __init__(self):
        self.chain: List[Dict] = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = {
            "block_index": 0,
            "timestamp": time.time(),
            "event_type": "GENESIS",
            "data": {"message": "Q-Transit Nexus Cryptographic Audit Ledger Initialized"},
            "prev_hash": "0" * 64,
            "hash": ""
        }
        genesis["hash"] = self._compute_hash(genesis)
        self.chain.append(genesis)

    def _compute_hash(self, block: Dict) -> str:
        block_string = json.dumps({
            "block_index": block["block_index"],
            "timestamp": block["timestamp"],
            "event_type": block["event_type"],
            "data": block["data"],
            "prev_hash": block["prev_hash"]
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def record_event(self, event_type: str, data: Dict) -> Dict:
        prev_block = self.chain[-1]
        new_block = {
            "block_index": len(self.chain),
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data,
            "prev_hash": prev_block["hash"],
            "hash": ""
        }
        new_block["hash"] = self._compute_hash(new_block)
        self.chain.append(new_block)
        return new_block

    def get_recent_blocks(self, count: int = 10) -> List[Dict]:
        return self.chain[-count:]
