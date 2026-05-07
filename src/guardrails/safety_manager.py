"""
Safety Manager
Coordinates safety guardrails and logs safety events.

This module manages input and output safety checks, coordinates
guardrails, and logs all safety events for auditing.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
from pathlib import Path
from src.guardrails.input_guardrail import InputGuardrail
from src.guardrails.output_guardrail import OutputGuardrail


class SafetyManager:
    """
    Manages safety guardrails for the multi-agent system.
    
    Coordinates input and output guardrails, handles violations,
    and logs all safety events for auditing and analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety manager.

        Args:
            config: Safety configuration from config.yaml
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.log_events = config.get("log_events", True)
        self.logger = logging.getLogger("safety")
        
        # Safety event log
        self.safety_events: List[Dict[str, Any]] = []
        
        # Prohibited categories
        self.prohibited_categories = config.get("prohibited_categories", [
            "harmful_content",
            "personal_attacks",
            "misinformation",
            "off_topic_queries"
        ])
        
        # Violation response strategy
        self.on_violation = config.get("on_violation", {})
        self.violation_action = self.on_violation.get("action", "refuse")
        self.violation_message = self.on_violation.get(
            "message",
            "I cannot process this request due to safety policies."
        )
        
        # Initialize guardrails
        if self.enabled:
            # Merge safety config with system config for guardrails
            guardrail_config = {
                **config,
                "topic": config.get("topic", "HCI Research"),
                "require_topic_relevance": config.get("require_topic_relevance", False)
            }
            
            self.input_guardrail = InputGuardrail(guardrail_config)
            self.output_guardrail = OutputGuardrail(guardrail_config)
            self.logger.info("Safety guardrails initialized")
        else:
            self.input_guardrail = None
            self.output_guardrail = None
            self.logger.info("Safety guardrails disabled")
        
        # Safety log file path
        log_config = config.get("logging", {})
        self.safety_log_file = log_config.get("safety_log", "logs/safety_events.log")
        
        # Ensure log directory exists
        if self.safety_log_file:
            log_path = Path(self.safety_log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if input query is safe to process.

        Args:
            query: User query to check

        Returns:
            Dictionary with:
                - safe: Boolean indicating if query is safe
                - violations: List of violations found
                - sanitized_query: Sanitized version (if violations found)
                - action: Recommended action ("allow", "refuse", "sanitize")
        """
        if not self.enabled or not self.input_guardrail:
            return {
                "safe": True,
                "violations": [],
                "sanitized_query": query,
                "action": "allow"
            }
        
        # Run input validation
        validation_result = self.input_guardrail.validate(query)
        
        is_safe = validation_result["valid"]
        violations = validation_result.get("violations", [])
        sanitized_query = validation_result.get("sanitized_input", query)
        
        # Determine action based on violations
        action = self._determine_action(violations, "input")
        
        # Log safety event
        if not is_safe and self.log_events:
            self._log_safety_event("input", query, violations, is_safe, sanitized_query)
        
        return {
            "safe": is_safe,
            "violations": violations,
            "sanitized_query": sanitized_query,
            "action": action
        }

    def check_output_safety(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if output response is safe to return.

        Args:
            response: Generated response to check
            sources: Optional list of sources used (for fact-checking)

        Returns:
            Dictionary with:
                - safe: Boolean indicating if response is safe
                - violations: List of violations found
                - sanitized_response: Sanitized version (if violations found)
                - action: Recommended action ("allow", "refuse", "sanitize")
        """
        if not self.enabled or not self.output_guardrail:
            return {
                "safe": True,
                "violations": [],
                "sanitized_response": response,
                "action": "allow"
            }
        
        # Run output validation
        validation_result = self.output_guardrail.validate(response, sources)
        
        is_safe = validation_result["valid"]
        violations = validation_result.get("violations", [])
        sanitized_response = validation_result.get("sanitized_output", response)
        
        # Determine action based on violations
        action = self._determine_action(violations, "output")
        
        # Apply action
        if action == "refuse":
            final_response = self.violation_message
        elif action == "sanitize":
            final_response = sanitized_response
        else:
            final_response = response
        
        # Log safety event
        if not is_safe and self.log_events:
            self._log_safety_event("output", response, violations, is_safe, final_response)
        
        return {
            "safe": is_safe,
            "violations": violations,
            "sanitized_response": sanitized_response,
            "response": final_response,
            "action": action
        }

    def _determine_action(self, violations: List[Dict[str, Any]], check_type: str) -> str:
        """
        Determine what action to take based on violations.

        Args:
            violations: List of violations
            check_type: "input" or "output"

        Returns:
            Action to take: "allow", "refuse", or "sanitize"
        """
        if not violations:
            return "allow"
        
        # Check for high-severity violations
        high_severity = any(v.get("severity") == "high" for v in violations)
        
        if high_severity:
            # High severity violations -> refuse
            return "refuse"
        
        # Check if configured action is sanitize
        if self.violation_action == "sanitize":
            return "sanitize"
        
        # Default to refuse for safety
        return "refuse"

    def _sanitize_response(self, response: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize response by removing or redacting unsafe content.

        Args:
            response: Original response
            violations: List of violations found

        Returns:
            Sanitized response
        """
        # Output guardrail handles sanitization
        if self.output_guardrail:
            validation_result = self.output_guardrail.validate(response)
            return validation_result.get("sanitized_output", response)
        
        # Fallback: basic redaction
        sanitized = response
        for violation in violations:
            if violation.get("validator") == "pii":
                matches = violation.get("matches", [])
                for match in matches:
                    sanitized = sanitized.replace(match, "[REDACTED]")
        
        return sanitized

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool,
        sanitized_content: Optional[str] = None
    ):
        """
        Log a safety event.

        Args:
            event_type: "input" or "output"
            content: The content that was checked
            violations: List of violations found
            is_safe: Whether content passed safety checks
            sanitized_content: Sanitized version (if applicable)
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "violation_count": len(violations),
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "content_length": len(content)
        }
        
        if sanitized_content:
            event["sanitized_preview"] = sanitized_content[:200] + "..." if len(sanitized_content) > 200 else sanitized_content
        
        # Add violation details
        if violations:
            event["violation_categories"] = [v.get("category", v.get("validator", "unknown")) for v in violations]
            event["max_severity"] = max([v.get("severity", "low") for v in violations], key=lambda s: ["low", "medium", "high"].index(s) if s in ["low", "medium", "high"] else 0)
        
        self.safety_events.append(event)
        
        # Log to logger
        if is_safe:
            self.logger.info(f"Safety check passed: {event_type}")
        else:
            self.logger.warning(
                f"Safety violation detected: {event_type} - "
                f"{len(violations)} violation(s) - "
                f"Categories: {event.get('violation_categories', [])}"
            )
        
        # Write to safety log file
        if self.safety_log_file and self.log_events:
            try:
                with open(self.safety_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """
        Get all logged safety events.

        Returns:
            List of safety event dictionaries
        """
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get statistics about safety events.

        Returns:
            Dictionary with safety statistics
        """
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])
        
        # Count violations by category
        violation_categories = {}
        for event in self.safety_events:
            if not event["safe"]:
                categories = event.get("violation_categories", [])
                for cat in categories:
                    violation_categories[cat] = violation_categories.get(cat, 0) + 1
        
        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0,
            "violation_categories": violation_categories
        }

    def clear_events(self):
        """Clear safety event log."""
        self.safety_events = []
        self.logger.info("Safety event log cleared")
