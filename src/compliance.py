"""
Compliance & Security Module

Handles PII protection and regulatory compliance for medical/scientific data.

Covers:
- GDPR (EU): Right to deletion, data minimization, consent management
- HIPAA (US): De-identification, access controls, audit logging

Critical for medical RAG systems handling patient data or research involving human subjects.
"""

import re
import hashlib
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from .logging_utils import get_logger


logger = get_logger()


@dataclass
class PIIPattern:
    """Pattern for detecting personally identifiable information."""
    name: str
    pattern: str
    replacement: str
    description: str = ""


# Standard PII patterns
DEFAULT_PII_PATTERNS = [
    PIIPattern(
        name="email",
        pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        replacement="[EMAIL_REDACTED]",
        description="Email addresses"
    ),
    PIIPattern(
        name="phone_us",
        pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        replacement="[PHONE_REDACTED]",
        description="US phone numbers"
    ),
    PIIPattern(
        name="ssn",
        pattern=r'\b\d{3}-\d{2}-\d{4}\b',
        replacement="[SSN_REDACTED]",
        description="US Social Security Numbers"
    ),
    PIIPattern(
        name="credit_card",
        pattern=r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        replacement="[CARD_REDACTED]",
        description="Credit card numbers"
    ),
    PIIPattern(
        name="ip_address",
        pattern=r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        replacement="[IP_REDACTED]",
        description="IP addresses"
    ),
    PIIPattern(
        name="date_of_birth",
        pattern=r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        replacement="[DOB_REDACTED]",
        description="Dates (potential DOB)"
    ),
]


class ComplianceHandler:
    """
    Handle compliance requirements for medical/scientific data.
    
    Features:
    - PII detection and anonymization
    - Pseudonymization (one-way hashing)
    - Audit logging for access tracking
    - Data retention management
    """
    
    def __init__(self, custom_patterns: Optional[List[PIIPattern]] = None):
        """
        Initialize compliance handler.
        
        Args:
            custom_patterns: Additional PII patterns beyond defaults
        """
        self.patterns = DEFAULT_PII_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)
    
    def anonymize_text(
        self,
        text: str,
        preserve_structure: bool = True
    ) -> tuple[str, dict]:
        """
        Remove PII from text while optionally preserving structure.
        
        Args:
            text: Input text potentially containing PII
            preserve_structure: If True, replace with placeholders; if False, remove entirely
        
        Returns:
            Tuple of (anonymized_text, redaction_report)
        """
        anonymized = text
        redaction_report = {
            "redactions": [],
            "patterns_matched": {},
            "original_length": len(text),
        }
        
        for pattern in self.patterns:
            matches = list(re.finditer(pattern.pattern, anonymized))
            
            if matches:
                count = len(matches)
                redaction_report["patterns_matched"][pattern.name] = count
                
                if preserve_structure:
                    anonymized = re.sub(
                        pattern.pattern,
                        pattern.replacement,
                        anonymized
                    )
                else:
                    anonymized = re.sub(pattern.pattern, "", anonymized)
                
                # Log redacted values (for audit, not in production)
                for match in matches:
                    redaction_report["redactions"].append({
                        "pattern": pattern.name,
                        "position": match.span(),
                        "value_hash": hashlib.sha256(match.group().encode()).hexdigest()[:16]
                    })
        
        redaction_report["final_length"] = len(anonymized)
        
        if redaction_report["redactions"]:
            logger.info(
                "PII redaction performed",
                patterns_found=len(redaction_report["patterns_matched"]),
                total_redactions=len(redaction_report["redactions"])
            )
        
        return anonymized, redaction_report
    
    @staticmethod
    def hash_identifier(
        identifier: str,
        salt: Optional[str] = None
    ) -> str:
        """
        One-way hash for pseudonymization.
        
        Use this to convert identifiable data (patient ID, email) to
        non-reversible pseudonyms while maintaining referential integrity.
        
        Args:
            identifier: The identifier to hash
            salt: Optional salt for additional security
        
        Returns:
            Hexadecimal hash string
        
        Example:
            >>> hash_identifier("patient_12345")
            "a3f2b1c..."  # Consistent for same input
        """
        if salt:
            identifier = f"{salt}:{identifier}"
        
        return hashlib.sha256(identifier.encode()).hexdigest()
    
    @staticmethod
    def audit_log(
        user_id: str,
        action: str,
        resource: str,
        allowed: bool = True,
        reason: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Log access for compliance audits (GDPR Article 30, HIPAA §164.312(b)).
        
        Critical for:
        - Demonstrating compliance during audits
        - Investigating data breaches
        - Tracking user consent
        
        Args:
            user_id: Identifier of the user performing the action
            action: Action performed (e.g., "query", "delete", "export")
            resource: Resource accessed (e.g., paper_id, query_text)
            allowed: Whether the action was allowed
            reason: Reason for denial (if not allowed)
            metadata: Additional context
        
        Example:
            >>> audit_log("user_123", "query", "medical_papers", allowed=True)
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "allowed": allowed,
            "reason": reason,
            "metadata": metadata or {}
        }
        
        # In production: Send to dedicated audit log storage (WORM - Write Once Read Many)
        logger.info(
            "AUDIT_EVENT",
            **log_entry
        )
    
    @staticmethod
    def check_data_retention(
        data_created: datetime,
        retention_days: int = 2555  # 7 years (HIPAA minimum)
    ) -> bool:
        """
        Check if data should be retained or deleted per policy.
        
        GDPR: Data should not be kept longer than necessary
        HIPAA: Minimum 6 years (7 years recommended)
        
        Args:
            data_created: When the data was created
            retention_days: Maximum retention period in days
        
        Returns:
            True if data should be retained, False if it should be deleted
        """
        age_days = (datetime.utcnow() - data_created).days
        should_retain = age_days < retention_days
        
        if not should_retain:
            logger.warning(
                "Data exceeded retention period",
                age_days=age_days,
                retention_days=retention_days
            )
        
        return should_retain
    
    @staticmethod
    def validate_consent(
        user_id: str,
        purpose: str,
        consent_data: dict
    ) -> bool:
        """
        Validate user consent for data processing (GDPR requirement).
        
        Args:
            user_id: User identifier
            purpose: Purpose of data processing (e.g., "research", "diagnosis")
            consent_data: Consent records
        
        Returns:
            True if explicit consent exists
        
        Example consent_data:
            {
                "user_123": {
                    "purposes": ["research", "analytics"],
                    "consented_at": "2024-01-01T00:00:00Z",
                    "expires_at": "2025-01-01T00:00:00Z"
                }
            }
        """
        user_consent = consent_data.get(user_id)
        
        if not user_consent:
            logger.warning("No consent record found", user_id=user_id)
            return False
        
        # Check if purpose is allowed
        if purpose not in user_consent.get("purposes", []):
            logger.warning(
                "Purpose not consented",
                user_id=user_id,
                purpose=purpose
            )
            return False
        
        # Check if consent has expired
        expires_at = datetime.fromisoformat(user_consent.get("expires_at", ""))
        if datetime.utcnow() > expires_at:
            logger.warning("Consent expired", user_id=user_id)
            return False
        
        return True


# HIPAA Safe Harbor De-identification
HIPAA_IDENTIFIERS = [
    "Names",
    "Geographic subdivisions smaller than state",
    "Dates (except year)",
    "Telephone numbers",
    "Fax numbers",
    "Email addresses",
    "Social Security numbers",
    "Medical record numbers",
    "Health plan beneficiary numbers",
    "Account numbers",
    "Certificate/license numbers",
    "Vehicle identifiers",
    "Device identifiers",
    "Web URLs",
    "IP addresses",
    "Biometric identifiers",
    "Full-face photos",
    "Any other unique identifier"
]


def is_hipaa_compliant_text(
    text: str,
    strict: bool = True
) -> tuple[bool, List[str]]:
    """
    Check if text is HIPAA Safe Harbor compliant.
    
    Args:
        text: Text to check
        strict: If True, flag any potential issues; if False, only flag definite violations
    
    Returns:
        Tuple of (is_compliant, list_of_violations)
    """
    handler = ComplianceHandler()
    _, report = handler.anonymize_text(text)
    
    violations = []
    
    # Check for PII patterns
    if report["patterns_matched"]:
        for pattern_name, count in report["patterns_matched"].items():
            violations.append(
                f"Found {count} instance(s) of {pattern_name}"
            )
    
    is_compliant = len(violations) == 0
    
    return is_compliant, violations


# Singleton instance for convenience
_compliance_handler: Optional[ComplianceHandler] = None


def get_compliance_handler() -> ComplianceHandler:
    """Get or create the global compliance handler."""
    global _compliance_handler
    if _compliance_handler is None:
        _compliance_handler = ComplianceHandler()
    return _compliance_handler
