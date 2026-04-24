"""
Task 3: Production Hardening and Security
Implements data compliance (PII redaction), audit trails, and system resilience.
"""

import time
import json
import logging
import hashlib
import re
from typing import List, Dict, Any, Optional, Callable
from functools import wraps

import structlog
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration for structured logging
def configure_logger():
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger()

logger = configure_logger()


class ValidatedQuery(BaseModel):
    """Schema for incoming user requests with validation."""
    query_text: str = Field(..., min_length=5, max_length=500)
    user_id: str = Field(...)

    @field_validator('query_text')
    @classmethod
    def sanitize_query(cls, v):
        # Basic sanitization for potential injection or corruption
        return re.sub(r'[<>{}[\]\\]', '', v).strip()


class ComplianceManager:
    """Handles data protection and regulatory compliance (GDPR/HIPAA)."""
    
    # Patterns for common PII (Personal Identifiable Information)
    PII_PATTERNS = {
        'EMAIL': r'[\w\.-]+@[\w\.-]+\.\w+',
        'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
        'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    }

    @classmethod
    def redact_content(cls, text: str) -> str:
        """Removes PII from strings to prevent data leakage."""
        redacted = text
        for label, pattern in cls.PII_PATTERNS.items():
            redacted = re.sub(pattern, f"<{label}_REDACTED>", redacted)
        return redacted

    @classmethod
    def pseudonymize_user(cls, user_id: str) -> str:
        """hashes user IDs to comply with data minimization principles."""
        return hashlib.sha256(user_id.encode()).hexdigest()[:12]


def audit_log(action_type: str):
    """Decorator to record system operations for auditing purposes."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            log_entry = {
                "action": action_type,
                "timestamp": time.time(),
                "status": "initiated"
            }
            logger.info("audit_event", **log_entry)
            
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                logger.info("audit_event", 
                            action=action_type, 
                            status="success", 
                            duration_ms=round(duration * 1000, 2))
                return result
            except Exception as e:
                logger.error("audit_event_error", 
                             action=action_type, 
                             status="failure", 
                             error=str(e))
                raise
        return wrapper
    return decorator


class ResilientProcessor:
    """Processor with integrated reliability mechanisms."""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    @audit_log("document_retrieval")
    def process_request(self, validated_data: ValidatedQuery):
        """Simulates a call to an external service with retry logic."""
        # Simulated workload
        print(f"Processing request for anonymized user: {ComplianceManager.pseudonymize_user(validated_data.user_id)}")
        
        # Example: Redacting PII from input before logging or storing
        safe_query = ComplianceManager.redact_content(validated_data.query_text)
        logger.debug("safe_query_processed", query=safe_query)
        
        return {"status": "success", "processed_query": safe_query}


def run_compliance_demo():
    print("Executing Task 3 proof-of-concept...")
    processor = ResilientProcessor()
    
    raw_payload = {
        "query_text": "Please analyze records for john.doe@example.com regarding SSN 000-11-2222",
        "user_id": "user_88321"
    }
    
    try:
        # Validation
        validated = ValidatedQuery(**raw_payload)
        
        # Resilient and Compliant Processing
        response = processor.process_request(validated)
        
        print("\nCompliance Results:")
        print(f"  Original:  {raw_payload['query_text']}")
        print(f"  Redacted:  {response['processed_query']}")
        print(f"  User Hash: {ComplianceManager.pseudonymize_user(validated.user_id)}")
        
    except Exception as e:
        print(f"Workflow failed: {e}")


if __name__ == "__main__":
    run_compliance_demo()
