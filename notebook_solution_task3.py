"""
TASK 3: Production Hardening - Best Practices
==============================================
Implementing Security, Compliance, Logging, and Error Resilience.

Copy this code into your Jupyter notebook cell for Task 3.
"""

# ============================================================================
# INSTALLATION (Run this once)
# ============================================================================
# !pip install structlog cryptography pydantic tenacity

import os
import time
import json
import logging
import hashlib
import re
from typing import List, Dict, Any, Optional, Union
from functools import wraps

# Enhancement: Structured Logging
import structlog

# Enhancement: Security & Cryptography
from cryptography.fernet import Fernet

# Enhancement: Validation
from pydantic import BaseModel, Field, validator

# Enhancement: Resilience
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================================================================
# 1. STRUCTURED LOGGING
# ============================================================================

def setup_logging():
    """Configures structured (JSON) logging for production monitoring."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger()

logger = setup_logging()

# ============================================================================
# 2. RESILIENCE (Retries & Error Handling)
# ============================================================================

class RAGSystemError(Exception):
    """Base custom exception for the RAG system."""
    pass

class APIConnectionError(RAGSystemError):
    """Raised when external APIs (ArXiv, PubMed, OpenAI) fail."""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def protected_api_call(func, *args, **kwargs):
    """
    Generic wrapper to protect any network-based function call.
    Uses exponential backoff retries.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error("api_call_failed", error=str(e), attempt="retrying...")
        raise APIConnectionError(f"Persistent failure after retries: {e}")

# ============================================================================
# 3. GDPR/HIPAA COMPLIANCE (Security & PII)
# ============================================================================

class ComplianceHandler:
    """
    Handles PII detection, Redaction, and Audit Trails.
    Essential for HIPAA (Medical Data) and GDPR (Privacy).
    """
    
    # Common PII Regex Patterns
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    }

    @staticmethod
    def audit_trail(user_id: str, action: str, paper_id: Optional[str] = None):
        """Logs every access attempt for compliance audits."""
        logger.info(
            "audit_log",
            user_id=ComplianceHandler.hash_id(user_id),
            action=action,
            paper_id=paper_id,
            timestamp=time.time()
        )

    @staticmethod
    def hash_id(identifier: str) -> str:
        """One-way cryptographic hash for pseudonymization."""
        return hashlib.sha256(identifier.encode()).hexdigest()

    @staticmethod
    def redact_pii(text: str) -> str:
        """Identifies and redacts sensitive information from scientific text."""
        redacted_text = text
        for pii_type, pattern in ComplianceHandler.PII_PATTERNS.items():
            matches = list(re.finditer(pattern, redacted_text))
            if matches:
                logger.warning("pii_detected", pii_type=pii_type, count=len(matches))
                redacted_text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted_text)
        return redacted_text

# ============================================================================
# 4. INPUT VALIDATION (Pydantic)
# ============================================================================

class QueryRequest(BaseModel):
    """Schema for validating user queries before they hit the LLM."""
    query: str = Field(..., min_length=5, max_length=500)
    user_id: str
    top_k: int = Field(default=5, ge=1, le=20)

    @validator('query')
    def prevent_prompt_injection(cls, v):
        """Basic safeguard against malicious instructions."""
        forbidden_keywords = ["ignore previous instructions", "system prompt", "delete database"]
        if any(keyword in v.lower() for keyword in forbidden_keywords):
            raise ValueError("Potential prompt injection detected.")
        return v

# ============================================================================
# 5. RATE LIMITER
# ============================================================================

class RequestThrottler:
    """Ensures we stay within API limits per minute."""
    def __init__(self, rpm_limit: int = 20):
        self.interval = 60.0 / rpm_limit
        self.last_call = 0.0

    def throttle(self):
        now = time.time()
        wait_time = self.interval - (now - self.last_call)
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_call = time.time()

# ============================================================================
# DEMONSTRATION WORKFLOW
# ============================================================================

def run_task3_demo():
    print("=" * 60)
    print("🚀 RUNNING TASK 3: Production Hardening & Compliance")
    print("=" * 60)

    # A. Validation
    user_query = "   What are the latest findings on COVID-19 mRNA vaccines?   "
    print(f"\n1. Validating User Input...")
    try:
        request = QueryRequest(query=user_query, user_id="user_john_doe", top_k=5)
        print(f"   ✅ Query validated: '{request.query}'")
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
        return

    # B. Compliance & Redaction
    raw_paper_content = """
    This study by lead researcher dr_smith@university.edu (Phone: 555-0199) 
    found that vaccines are highly effective. Patient IP: 192.168.1.1.
    """
    print(f"\n2. Applying GDPR/HIPAA Redaction...")
    safe_content = ComplianceHandler.redact_pii(raw_paper_content)
    print("   Original Text: Leading researcher info present.")
    print(f"   Cleaned Text: {safe_content.strip()}")

    # C. Audit Trail
    print(f"\n3. Logging Audit Trail (Pseudonymized)...")
    ComplianceHandler.audit_trail(request.user_id, "search_query", paper_id="arxiv_2301.0001")
    
    # D. Resilience Demo (Retries)
    print(f"\n4. Resilience Simulation (Retries)...")
    def simulate_flaky_api():
        import random
        if random.random() < 0.6: # 60% failure rate
            raise ConnectionError("Network timeout!")
        return "Success from API"

    try:
        result = protected_api_call(simulate_flaky_api)
        print(f"   ✅ API Result: {result}")
    except APIConnectionError as e:
        print(f"   ❌ Final Failure: {e}")

    print("\n" + "=" * 60)
    print("✅ TASK 3 COMPLETE (Check JSON output above for logs)")
    print("=" * 60)

if __name__ == "__main__":
    run_task3_demo()
