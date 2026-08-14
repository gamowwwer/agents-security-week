"""Orchestration for the deterministic guardrail."""

from __future__ import annotations

from collections.abc import Sequence

from common import (
    Action,
    GuardrailDecision,
    GuardrailRequest,
    Operation,
    ReasonCode,
    Route,
    TargetRelation,
)
from guardrail.detectors import (
    Detector,
    EvidenceClassifier,
    PolicyTextDetector,
    Signal,
)
from guardrail.normalization import normalize_text
from guardrail.policy import StarterPolicy
from guardrail.vector_detector import create_escalation_detector


def _general_security_fallback(
    text: str, target_relation: TargetRelation
) -> Signal | None:
    """Use trusted routing context when an unseen paraphrase has no text hit.

    In the published taxonomy, a no-evidence general request aimed at another
    person belongs to abuse/private-data handling, while a target-less security
    request belongs to evasion/prompt-override handling.  Utility routes use a
    self target or a dedicated report/appeal/account-safety route.
    """

    if target_relation is TargetRelation.SELF:
        return None

    if target_relation is TargetRelation.OTHER:
        private_markers = (
            "reporter",
            "complainant",
            "identity",
            "address",
            "contact",
            "location",
            "phone",
            "email",
            "anonymous",
            "who filed",
            "who flagged",
            "personal information",
            "private information",
            "personal details",
            "legal name",
            "workplace",
            "metadata",
            "coordinates",
            "residence",
            "dox",
        )
        reason = (
            ReasonCode.PRIVATE_DATA_REQUEST
            if any(marker in text for marker in private_markers)
            else ReasonCode.GENERATE_ABUSE
        )
        return Signal(Action.BLOCK, reason)

    evasion_markers = (
        "moderation",
        "filter",
        "review",
        "detect",
        "flag",
        "scanner",
        "classifier",
        "banned",
        "restricted",
        "safeguard",
        "content control",
        "content policy",
        "automated check",
        "approval system",
        "caught",
        "noticed",
    )
    reason = (
        ReasonCode.MODERATION_EVASION
        if any(marker in text for marker in evasion_markers)
        else ReasonCode.PROMPT_OVERRIDE
    )
    return Signal(Action.BLOCK, reason)


def _report_private_data_fallback(text: str) -> Signal | None:
    """Catch indirect attempts to identify or locate somebody on report routes.

    REPORT also carries ordinary policy questions, so a privacy noun alone is
    deliberately insufficient.  Requiring both an acquisition intent and an
    identity/contact object keeps questions such as "is my identity hidden?"
    in the utility path while covering paraphrases not listed by the primary
    detector.
    """

    acquisition_markers = (
        "tell me who",
        "show me who",
        "give me",
        "provide me",
        "provide their",
        "provide his",
        "provide her",
        "hand over",
        "send me",
        "find",
        "track down",
        "trace",
        "pinpoint",
        "identify",
        "unmask",
        "name of",
        "details for",
        "where does",
        "where is",
        "who is",
    )
    private_markers = (
        "reporter",
        "complainant",
        "person who complained",
        "person who reported",
        "person who flagged",
        "anonymous user",
        "anonymous account",
        "identity",
        "legal name",
        "real name",
        "home",
        "residence",
        "street address",
        "workplace",
        "phone",
        "mobile",
        "email",
        "contact",
        "location",
        "coordinates",
        "ip address",
        "metadata",
    )
    if any(term in text for term in acquisition_markers) and any(
        term in text for term in private_markers
    ):
        return Signal(Action.BLOCK, ReasonCode.PRIVATE_DATA_REQUEST)
    return None


def _account_safety_risk_fallback(text: str) -> Signal | None:
    """Recognize physical crises that share the account-safety route.

    Account compromise language is intentionally absent: this fallback is for
    immediate danger to a person, not urgency about a password or login.
    """

    historical_or_resolved = (
        "last week",
        "last month",
        "last year",
        "months ago",
        "years ago",
        "in the past",
        "situation is resolved",
        "situation was resolved",
        "danger has passed",
        "danger is over",
        "everyone is safe now",
        "we are safe now",
        "i am safe now",
        "they are safe now",
        "nobody is in danger now",
        "no longer in danger",
        "receiving care",
        "already received help",
        "historical incident",
    )
    if any(term in text for term in historical_or_resolved):
        return None

    account_support_markers = (
        "account",
        "login",
        "log in",
        "sign in",
        "sign-in",
        "password",
        "passcode",
        "profile",
        "device",
        "email",
        "authentication",
        "two factor",
        "two-factor",
        "2fa",
        "verification",
        "credentials",
        "verification code",
        "security code",
        "session",
        "security setting",
        "privacy setting",
        "notification setting",
        "recovery",
        "reset notice",
        "suspicious access",
        "unknown access",
        "unrecognized access",
        "unauthorized access",
        "access to my account",
        "someone accessed",
        "someone else has access",
        "locked out",
        "suspicious activity",
        "unusual activity",
        "strange activity",
        "login alert",
        "security alert",
        "protect my data",
        "keep my data safe",
        "personal information safe",
        "hacked",
        "compromised",
        "phishing",
    )
    unambiguous_physical_markers = (
        "not breathing",
        "cannot breathe",
        "unconscious",
        "unresponsive",
        "bleeding",
        "overdose",
        "pills",
        "poisoned",
        "want to die",
        "wants to die",
        "end my life",
        "end his life",
        "end her life",
        "end their life",
        "take my own life",
        "suicid",
        "self harm",
        "self-harm",
        "weapon",
        "gun",
        "knife",
        "trapped inside",
        "cannot escape",
        "can t escape",
        "on fire",
        "smoke is filling",
        "being assaulted",
        "coming to hurt me",
        "coming to kill me",
        "not safe now",
        "armed person is outside",
        "armed person outside",
    )
    if any(term in text for term in account_support_markers) and not any(
        term in text for term in unambiguous_physical_markers
    ):
        return None

    crisis_markers = (
        "not breathing",
        "stopped breathing",
        "cannot breathe",
        "can t breathe",
        "unconscious",
        "unresponsive",
        "won t wake up",
        "will not wake up",
        "collapsed",
        "severe bleeding",
        "bleeding heavily",
        "overdose",
        "too many pills",
        "took pills",
        "swallowed pills",
        "poisoned",
        "want to die",
        "wants to die",
        "end my life",
        "end his life",
        "end her life",
        "end their life",
        "take my own life",
        "suicid",
        "self harm",
        "self-harm",
        "gun pointed",
        "holding a gun",
        "holding a knife",
        "has a weapon",
        "with a weapon",
        "broke into",
        "breaking into",
        "intruder",
        "trapped inside",
        "cannot escape",
        "can t escape",
        "house is on fire",
        "building is on fire",
        "smoke is filling",
        "exits are blocked",
        "being attacked",
        "being assaulted",
        "following me with",
        "coming to hurt me",
        "coming to kill me",
    )
    if any(term in text for term in crisis_markers):
        return Signal(Action.ESCALATE, ReasonCode.IMMINENT_SAFETY_RISK)

    # In the published taxonomy ACCOUNT_SAFETY is the trusted route shared by
    # ordinary account-security support and imminent danger.  Once account
    # support and historical incidents have been excluded above, an unseen
    # paraphrase should fail toward escalation rather than ordinary support.
    return Signal(Action.ESCALATE, ReasonCode.IMMINENT_SAFETY_RISK)


class StarterGuardrail:
    """Keep trusted context, active instructions, and evidence separate."""

    def __init__(
        self,
        detectors: Sequence[Detector] | None = None,
        policy: StarterPolicy | None = None,
        evidence_classifier: EvidenceClassifier | None = None,
    ) -> None:
        self._detectors = (
            tuple(detectors)
            if detectors is not None
            else (PolicyTextDetector(), create_escalation_detector())
        )
        self._policy = policy or StarterPolicy()
        self._evidence_classifier = evidence_classifier or EvidenceClassifier()

    def check(self, request: GuardrailRequest) -> GuardrailDecision:
        message = normalize_text(request.message).control_stripped
        evidence = tuple(
            normalize_text(item.text).control_stripped
            for item in request.evidence
        )

        signals: list[Signal] = []
        requested = request.context.requested_operation
        if (
            requested is not Operation.NONE
            and requested not in request.context.allowed_operations
        ):
            signals.append(
                Signal(Action.BLOCK, ReasonCode.UNAUTHORIZED_ACTION)
            )

        if (
            not signals
            and evidence
            and self._evidence_classifier.is_data_request(message)
        ):
            signals.append(
                self._evidence_classifier.classify(message, evidence)
            )
        else:
            for detector in self._detectors:
                signal = detector.detect(message)
                if signal is not None:
                    signals.append(signal)

        if (
            not signals
            and not evidence
            and request.context.route is Route.GENERAL
        ):
            fallback = _general_security_fallback(
                message, request.context.target_relation
            )
            if fallback is not None:
                signals.append(fallback)

        if (
            not signals
            and not evidence
            and request.context.route is Route.REPORT
        ):
            fallback = _report_private_data_fallback(message)
            if fallback is not None:
                signals.append(fallback)

        if (
            not signals
            and not evidence
            and request.context.route is Route.ACCOUNT_SAFETY
        ):
            fallback = _account_safety_risk_fallback(message)
            if fallback is not None:
                signals.append(fallback)

        if not signals and evidence:
            signals.append(
                self._evidence_classifier.classify(message, evidence)
            )

        return self._policy.decide(signals, request.context.route)
