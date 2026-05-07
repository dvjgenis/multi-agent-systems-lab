"""
Input Guardrail
Checks user inputs for safety violations.

This module implements comprehensive input validation including:
- Toxic/harmful language detection
- Prompt injection detection
- Query length and format validation
- Off-topic query detection
- Prohibited content detection
"""

import re
import logging
from typing import Dict, Any, List, Optional


class InputGuardrail:
    """
    Guardrail for checking input safety.
    
    Implements multiple validation checks:
    1. Length validation
    2. Toxic language detection
    3. Prompt injection detection
    4. Off-topic query detection
    5. Prohibited keyword detection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("guardrails.input")
        
        # Validation thresholds
        self.min_length = config.get("min_length", 5)
        self.max_length = config.get("max_length", 2000)
        
        # Topic configuration
        self.topic = config.get("topic", "HCI Research")
        self.require_topic_relevance = config.get("require_topic_relevance", False)
        
        # Prohibited keywords (can be extended)
        self.prohibited_keywords = config.get("prohibited_keywords", [
            "hack", "attack", "exploit", "bypass", "crack", "illegal",
            "harmful", "dangerous", "violence", "weapon"
        ])
        
        # Prompt injection patterns
        self.injection_patterns = [
            r"ignore\s+(previous|all|above)\s+(instructions|rules|prompts)",
            r"disregard\s+(previous|all|above)",
            r"forget\s+(everything|all|previous)",
            r"system\s*:",
            r"sudo\s+",
            r"root\s+access",
            r"admin\s+privileges",
            r"override\s+",
            r"bypass\s+safety",
            r"jailbreak",
            r"<\|system\|>",
            r"\[SYSTEM\]",
            r"you\s+are\s+now",
            r"pretend\s+you\s+are",
            r"act\s+as\s+if",
        ]
        
        # Compile regex patterns for efficiency
        self.injection_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in self.injection_patterns]
        
        # Topic-related keywords (for relevance checking)
        self.topic_keywords = {
            "HCI Research": [
                "human", "computer", "interaction", "interface", "design", "usability",
                "user experience", "UX", "UI", "accessibility", "ergonomics",
                "interaction design", "user interface", "user-centered", "HCI"
            ],
            "general": []  # Empty means no topic restriction
        }

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query.

        Args:
            query: User input to validate

        Returns:
            Dictionary with:
                - valid: Boolean indicating if query is valid
                - violations: List of violation dictionaries
                - sanitized_input: Sanitized version of input (if needed)
        """
        violations = []
        
        # 1. Length validation
        length_violations = self._check_length(query)
        violations.extend(length_violations)
        
        # 2. Prompt injection detection
        injection_violations = self._check_prompt_injection(query)
        violations.extend(injection_violations)
        
        # 3. Toxic/harmful language detection
        toxic_violations = self._check_toxic_language(query)
        violations.extend(toxic_violations)
        
        # 4. Prohibited keywords
        keyword_violations = self._check_prohibited_keywords(query)
        violations.extend(keyword_violations)
        
        # 5. Topic relevance (if required)
        if self.require_topic_relevance:
            relevance_violations = self._check_relevance(query)
            violations.extend(relevance_violations)
        
        # Determine if valid
        is_valid = len(violations) == 0
        
        # Sanitize if needed (basic sanitization)
        sanitized_input = self._sanitize_input(query, violations) if violations else query
        
        return {
            "valid": is_valid,
            "violations": violations,
            "sanitized_input": sanitized_input
        }

    def _check_length(self, text: str) -> List[Dict[str, Any]]:
        """
        Check query length.

        Args:
            text: Query text

        Returns:
            List of length violations
        """
        violations = []
        
        if len(text) < self.min_length:
            violations.append({
                "validator": "length",
                "reason": f"Query too short (minimum {self.min_length} characters)",
                "severity": "low",
                "value": len(text),
                "threshold": self.min_length
            })
        
        if len(text) > self.max_length:
            violations.append({
                "validator": "length",
                "reason": f"Query too long (maximum {self.max_length} characters)",
                "severity": "medium",
                "value": len(text),
                "threshold": self.max_length
            })
        
        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prompt injection attempts.

        Args:
            text: Query text

        Returns:
            List of prompt injection violations
        """
        violations = []
        
        for i, regex in enumerate(self.injection_regexes):
            if regex.search(text):
                pattern = self.injection_patterns[i]
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Potential prompt injection detected: {pattern}",
                    "severity": "high",
                    "pattern": pattern,
                    "category": "security"
                })
        
        return violations

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for toxic/harmful language.

        Args:
            text: Query text

        Returns:
            List of toxic language violations
        """
        violations = []
        
        # Toxic language patterns (can be enhanced with ML models)
        toxic_patterns = [
            (r"\b(kill|murder|death\s+wish|suicide)\b", "violence", "high"),
            (r"\b(hate|despise|loathe)\s+(you|your|people|group)", "hate_speech", "high"),
            (r"\b(racist|sexist|homophobic|transphobic)\b", "discrimination", "high"),
            (r"\b(threat|harm|hurt|injure)\s+(you|someone|people)", "threats", "high"),
        ]
        
        text_lower = text.lower()
        for pattern, category, severity in toxic_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violations.append({
                    "validator": "toxic_language",
                    "reason": f"Potentially toxic language detected: {category}",
                    "severity": severity,
                    "category": category
                })
        
        return violations

    def _check_prohibited_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prohibited keywords.

        Args:
            text: Query text

        Returns:
            List of prohibited keyword violations
        """
        violations = []
        
        text_lower = text.lower()
        for keyword in self.prohibited_keywords:
            if keyword.lower() in text_lower:
                # Check if it's part of a legitimate word (e.g., "hack" in "hackathon")
                word_boundary_pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(word_boundary_pattern, text_lower):
                    violations.append({
                        "validator": "prohibited_keyword",
                        "reason": f"Query contains prohibited keyword: {keyword}",
                        "severity": "medium",
                        "keyword": keyword,
                        "category": "content_policy"
                    })
        
        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Check if query is relevant to the system's purpose.

        Args:
            query: Query text

        Returns:
            List of relevance violations
        """
        violations = []
        
        if not self.topic or self.topic == "general":
            return violations
        
        # Get topic keywords
        keywords = self.topic_keywords.get(self.topic, [])
        if not keywords:
            return violations
        
        # Check if query contains any topic-related keywords
        query_lower = query.lower()
        has_topic_keyword = any(keyword.lower() in query_lower for keyword in keywords)
        
        if not has_topic_keyword:
            violations.append({
                "validator": "relevance",
                "reason": f"Query may not be relevant to {self.topic}",
                "severity": "low",
                "category": "off_topic"
            })
        
        return violations

    def _sanitize_input(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize input by removing or redacting problematic content.

        Args:
            text: Original text
            violations: List of violations found

        Returns:
            Sanitized text
        """
        sanitized = text
        
        # For prompt injection, remove the pattern
        for violation in violations:
            if violation.get("validator") == "prompt_injection":
                pattern = violation.get("pattern", "")
                if pattern:
                    try:
                        sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)
                    except:
                        pass
        
        # For prohibited keywords, replace with [REDACTED]
        for violation in violations:
            if violation.get("validator") == "prohibited_keyword":
                keyword = violation.get("keyword", "")
                if keyword:
                    sanitized = re.sub(
                        r'\b' + re.escape(keyword) + r'\b',
                        "[REDACTED]",
                        sanitized,
                        flags=re.IGNORECASE
                    )
        
        return sanitized
