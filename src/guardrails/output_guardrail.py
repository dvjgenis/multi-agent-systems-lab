"""
Output Guardrail
Checks system outputs for safety violations.

This module implements comprehensive output validation including:
- PII (Personally Identifiable Information) detection
- Harmful content detection
- Factual consistency checking
- Bias detection
- Content sanitization
"""

import re
import logging
from typing import Dict, Any, List, Optional


class OutputGuardrail:
    """
    Guardrail for checking output safety.
    
    Implements multiple validation checks:
    1. PII detection (emails, phones, SSNs, etc.)
    2. Harmful content detection
    3. Factual consistency (against sources)
    4. Bias detection
    5. Content sanitization
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("guardrails.output")
        
        # PII detection patterns
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_us": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "phone_international": r'\b(?:\+\d{1,3}[-.\s]?)?(?!19\d{2}|20[0-2]\d)(?:\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
            "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "mac_address": r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b',
        }
        
        # Compile regex patterns
        self.pii_regexes = {
            key: re.compile(pattern) 
            for key, pattern in self.pii_patterns.items()
        }
        
        # Harmful content patterns
        self.harmful_patterns = [
            (r"\b(kill|murder|assassinate|execute)\s+(yourself|someone|people)", "violence", "high"),
            (r"\b(suicide|self-harm|self\s+harm)\b", "self_harm", "high"),
            (r"\b(bomb|explosive|weapon|gun)\s+(making|building|how\s+to)", "dangerous_instructions", "high"),
            (r"\b(hack|exploit|attack)\s+(website|system|database)", "illegal_instructions", "high"),
            (r"\b(illegal|unlawful)\s+(activity|action|method)", "illegal_content", "medium"),
        ]
        
        # Bias indicators (simplified - can be enhanced)
        self.bias_patterns = [
            (r"\b(all|every|none|no)\s+(men|women|people|users)\s+(are|do|think)", "overgeneralization", "medium"),
            (r"\b(always|never)\s+(better|worse)", "absolute_statements", "low"),
        ]

    def validate(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output response.

        Args:
            response: Generated response to validate
            sources: Optional list of sources used (for fact-checking)

        Returns:
            Dictionary with:
                - valid: Boolean indicating if response is valid
                - violations: List of violation dictionaries
                - sanitized_output: Sanitized version of response
        """
        violations = []
        
        # 1. PII detection
        pii_violations = self._check_pii(response)
        violations.extend(pii_violations)
        
        # 2. Harmful content detection
        harmful_violations = self._check_harmful_content(response)
        violations.extend(harmful_violations)
        
        # 3. Bias detection
        bias_violations = self._check_bias(response)
        violations.extend(bias_violations)
        
        # 4. Factual consistency (if sources provided)
        if sources:
            consistency_violations = self._check_factual_consistency(response, sources)
            violations.extend(consistency_violations)
        
        # Determine if valid
        is_valid = len(violations) == 0
        
        # Sanitize if needed
        sanitized_output = self._sanitize(response, violations) if violations else response
        
        return {
            "valid": is_valid,
            "violations": violations,
            "sanitized_output": sanitized_output
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for personally identifiable information.

        Args:
            text: Response text

        Returns:
            List of PII violations
        """
        violations = []
        
        for pii_type, regex in self.pii_regexes.items():
            matches = regex.findall(text)
            if matches:
                # Filter out false positives (e.g., years like 2024)
                if pii_type == "ip_address":
                    # Filter out common non-IP patterns
                    filtered_matches = [
                        m for m in matches 
                        if not (m.startswith("192.168") or m.startswith("127.0.0"))
                    ]
                    matches = filtered_matches
                
                if matches:
                    violations.append({
                        "validator": "pii",
                        "pii_type": pii_type,
                        "reason": f"Contains {pii_type}",
                        "severity": "high",
                        "matches": matches[:5],  # Limit to first 5 matches
                        "count": len(matches)
                    })
        

        # Filter out false positives: years (1900-2099) and page numbers
        filtered_violations = []
        for violation in violations:
            pii_type = violation.get("pii_type", "")
            matches = violation.get("matches", [])
            
            if pii_type == "phone_international":
                filtered_matches = []
                for match in matches:
                    match_str = str(match).strip()
                    # Skip years (1900-2099)
                    if match_str.isdigit() and len(match_str) == 4:
                        year_val = int(match_str)
                        if 1900 <= year_val <= 2099:
                            continue
                    # Skip page number patterns (e.g., "2504-2289")
                    if re.match(r'^\d{4}-\d{4}$', match_str):
                        continue
                    # Skip if it's just a 4-digit number that looks like a year
                    if re.match(r'^\d{4}$', match_str) and match_str.startswith(('19', '20')):
                        continue
                    filtered_matches.append(match)
                
                if filtered_matches:
                    violation["matches"] = filtered_matches
                    violation["count"] = len(filtered_matches)
                    filtered_violations.append(violation)
            else:
                filtered_violations.append(violation)
        
        violations = filtered_violations
        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for harmful or inappropriate content.

        Args:
            text: Response text

        Returns:
            List of harmful content violations
        """
        violations = []
        
        text_lower = text.lower()
        
        for pattern, category, severity in self.harmful_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violations.append({
                    "validator": "harmful_content",
                    "reason": f"May contain harmful content: {category}",
                    "severity": severity,
                    "category": category
                })
        
        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check if response is consistent with sources.

        This is a simplified check - a full implementation would use
        LLM-based verification or fact-checking APIs.

        Args:
            response: Generated response
            sources: List of source dictionaries

        Returns:
            List of consistency violations
        """
        violations = []
        
        # Basic check: ensure response mentions sources
        if sources:
            source_titles = [s.get("title", "").lower() for s in sources if s.get("title")]
            response_lower = response.lower()
            
            # Check if response mentions any source titles
            mentions_sources = any(
                title[:20] in response_lower or response_lower in title[:50]
                for title in source_titles
                if title
            )
            
            # This is a weak check - in production, use LLM-based verification
            # For now, we'll just log if no sources are mentioned
            if not mentions_sources and len(sources) > 3:
                violations.append({
                    "validator": "factual_consistency",
                    "reason": "Response may not be well-grounded in provided sources",
                    "severity": "low",
                    "category": "source_attribution"
                })
        
        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for biased language.

        Args:
            text: Response text

        Returns:
            List of bias violations
        """
        violations = []
        
        text_lower = text.lower()
        
        for pattern, category, severity in self.bias_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violations.append({
                    "validator": "bias",
                    "reason": f"Potentially biased language: {category}",
                    "severity": severity,
                    "category": category
                })
        
        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize text by removing/redacting violations.

        Args:
            text: Original text
            violations: List of violations found

        Returns:
            Sanitized text
        """
        sanitized = text
        
        # Redact PII
        for violation in violations:
            if violation.get("validator") == "pii":
                pii_type = violation.get("pii_type")
                matches = violation.get("matches", [])
                
                if pii_type and matches:
                    regex = self.pii_regexes.get(pii_type)
                    if regex:
                        for match in matches:
                            sanitized = sanitized.replace(match, f"[{pii_type.upper()} REDACTED]")
        
        # Redact harmful content patterns
        for violation in violations:
            if violation.get("validator") == "harmful_content":
                category = violation.get("category", "")
                if category:
                    # Find and redact the harmful pattern
                    for pattern, cat, _ in self.harmful_patterns:
                        if cat == category:
                            sanitized = re.sub(
                                pattern,
                                f"[{category.upper()} CONTENT REDACTED]",
                                sanitized,
                                flags=re.IGNORECASE
                            )
                            break
        
        return sanitized
