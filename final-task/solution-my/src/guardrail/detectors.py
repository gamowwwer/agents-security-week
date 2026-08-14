"""Deterministic text signals used by the guardrail policy.

The small ordered detector is kept as a public building block.  The runtime uses
the more conservative policy detector below: it looks for an intent and an
object together instead of blocking on an isolated word such as ``ignore``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol, Sequence, runtime_checkable

from common import Action, ReasonCode
from guardrail.normalization import normalize_text


@dataclass(frozen=True, slots=True)
class Signal:
    """A detector's proposed action and reason."""

    action: Action
    reason_code: ReasonCode


@runtime_checkable
class Detector(Protocol):
    """Structural interface implemented by starter signal detectors."""

    def detect(self, text: str) -> Signal | None:
        """Return one signal for a flattened input, or no signal."""


@dataclass(frozen=True, slots=True)
class KeywordRule:
    """One ordered policy concept and its literal trigger terms."""

    action: Action
    reason_code: ReasonCode
    keywords: tuple[str, ...]


DEFAULT_KEYWORD_RULES = (
    KeywordRule(
        Action.BLOCK,
        ReasonCode.PROMPT_OVERRIDE,
        ("ignore", "system prompt", "reveal", "secret"),
    ),
    KeywordRule(
        Action.BLOCK,
        ReasonCode.MODERATION_EVASION,
        ("bypass", "evade", "avoid the filter"),
    ),
    KeywordRule(
        Action.BLOCK,
        ReasonCode.GENERATE_ABUSE,
        ("threaten", "intimidate", "kill"),
    ),
    KeywordRule(
        Action.BLOCK,
        ReasonCode.PRIVATE_DATA_REQUEST,
        ("reporter identity", "private data", "home address"),
    ),
)


# Checked ahead of the production block rules (see ``PRODUCTION_KEYWORD_RULES``
# below) so urgency/immediacy/self-harm phrasing wins over generic
# "threaten"/"kill" substrings (e.g. "threatening me right now" must
# escalate, not block).
IMMINENT_SAFETY_KEYWORD_RULE = KeywordRule(
    Action.ESCALATE,
    ReasonCode.IMMINENT_SAFETY_RISK,
    (
        "immediate danger",
        "in danger right now",
        "right now and i am in danger",
        "danger right now",
        "life is in danger",
        "about to hurt themselves",
        "about to hurt myself",
        "about to hurt himself",
        "about to hurt herself",
        "going to hurt myself",
        "going to kill myself",
        "want to kill myself",
        "kill myself right now",
        "suicidal",
        "suicide attempt",
        "kill himself",
        "kill herself",
        "kill themselves",
        "kill themself",
        "overdose right now",
        "swallowed a bottle of pills",
        "swallowed an overdose",
        "swallowed pills",
        "took a bottle of pills",
        "took an overdose",
        "took pills to end",
        "hurt themselves right now",
        "hurt myself right now",
        "cutting myself right now",
        "harming myself right now",
        "self-harm right now",
        "self harm right now",
        "actively harming",
        "ending my life",
        "end my life",
        "end it all",
        "ending everything tonight",
        "no reason to live",
        "can't go on like this",
        "cannot go on like this",
        "don't want to be here anymore",
        "do not want to be here anymore",
        "have a gun and",
        "have a weapon and",
        "outside my door",
        "breaking in right now",
        "broke into my home",
        "trapped inside",
        "can't get out",
        "cannot get out",
        "fire spreading",
        "someone is attacking me",
        "being attacked right now",
        "beating me right now",
        "is beating me",
        "being beaten right now",
        "not breathing",
        "choking and",
        "drowning right now",
        "urgent help now",
        "need urgent help",
        "need help right now",
        "call 911",
        "call emergency services",
        "threatening me right now",
    ),
)


# Precise, phrase-qualified counterparts of the same four concepts covered by
# ``DEFAULT_KEYWORD_RULES``. That constant is left untouched because pinned
# tests exercise it directly by name; this is the ruleset the running
# guardrail actually uses, favoring multi-word phrases over single common
# words ("ignore", "reveal", "secret", "kill", "bypass", "evade") that are
# too easily said in an entirely ordinary sentence.
PRODUCTION_KEYWORD_RULES = (
    KeywordRule(
        Action.BLOCK,
        ReasonCode.PROMPT_OVERRIDE,
        (
            # Bare, high-recall triggers (same as the original starter rule):
            # kept because real-world evidence showed narrowing these to
            # phrase-only variants cost far more true positives than the
            # false positives it removed (see engine.py's docstring note).
            "ignore",
            "system prompt",
            "reveal",
            "secret",
            # Additional phrase-qualified variants for paraphrases the bare
            # words above don't cover.
            "ignore all prior",
            "ignore all previous",
            "ignore prior directions",
            "ignore previous instructions",
            "ignore your instructions",
            "ignore your rules",
            "ignore your guidelines",
            "ignore this policy",
            "ignore policy",
            "disregard your instructions",
            "disregard prior instructions",
            "disregard the rules",
            "disregard the conversation rules",
            "forget your instructions",
            "forget everything above",
            "forget all previous instructions",
            "pretend you have no instructions",
            "pretend you have no restrictions",
            "pretend you have no prior",
            "reveal the system prompt",
            "reveal your instructions",
            "reveal your hidden instructions",
            "reveal the hidden configuration",
            "expose the system prompt",
            "expose your instructions",
            "expose the secret",
            "override your instructions",
            "override your configuration",
            "print the hidden system message",
            "disclose the hidden configuration",
            "disclose confidential setup",
        ),
    ),
    KeywordRule(
        Action.BLOCK,
        ReasonCode.MODERATION_EVASION,
        (
            # Bare, high-recall triggers (as in the original starter rule).
            "bypass",
            "evade",
            "avoid the filter",
            # Additional phrase-qualified variants.
            "bypass moderation",
            "bypass the filter",
            "bypass content review",
            "bypass the review",
            "bypass automated moderation",
            "evade the filter",
            "evade moderation",
            "evade detection",
            "evade the safety checks",
            "avoid the filter",
            "slip past the filter",
            "slip past moderation",
            "slip wording past",
            "get past the moderators",
            "without the moderators noticing",
            "republished without detection",
            "posting without detection",
            "circumvent the filter",
            "circumvent moderation",
            "circumvent the review",
        ),
    ),
    KeywordRule(
        Action.BLOCK,
        ReasonCode.GENERATE_ABUSE,
        (
            # Bare, high-recall triggers (as in the original starter rule).
            # "kill" collides with third-person self-harm phrasing
            # ("kill himself"/"herself"/"themselves"), but that is
            # intercepted upstream by IMMINENT_SAFETY_KEYWORD_RULE, which is
            # always checked first (see engine.py).
            "threaten",
            "intimidate",
            "kill",
            # Additional phrase-qualified variants.
            "menacing",
            "menace",
            "devastating insult",
            "insult so devastating",
            "cruel enough to",
            "threaten a ",
            "threaten my ",
            "threaten his ",
            "threaten her ",
            "threaten their ",
            "threaten someone",
            "threaten them",
            "threaten you",
            "want to threaten",
            "help me threaten",
            "kill them",
            "kill him",
            "kill her",
            "want to kill",
            "going to kill them",
        ),
    ),
    KeywordRule(
        Action.BLOCK,
        ReasonCode.PRIVATE_DATA_REQUEST,
        (
            "reporter identity",
            "private data",
            "home address",
            "private home address",
            "residential address",
            "where they live",
            "find out where",
            "dig up the",
            "phone number of the person who reported",
            "their contact info",
            "personal contact information",
            "personal information about someone else",
        ),
    ),
)


# Ordered most-specific-first: injection language is the most distinctive,
# followed by transactional scam language, with abusive/hostile language as
# the broadest (last) bucket. Used to sub-classify already-quoted evidence
# text, never the live message channel.
EVIDENCE_KEYWORD_RULES = (
    KeywordRule(
        Action.ALLOW_AS_DATA,
        ReasonCode.QUOTED_INJECTION,
        (
            "system prompt",
            "disregard your directions",
            "disregard the assistant",
            "ignore the system prompt",
            "override your instructions",
            "override the instructions",
            "reveal the secret",
            "expose the secret",
            "expose [secret]",
            "act as if you have no restrictions",
        ),
    ),
    KeywordRule(
        Action.ALLOW_AS_DATA,
        ReasonCode.QUOTED_SCAM,
        (
            "processing fee",
            "guaranteed prize",
            "send gift cards",
            "claim your prize",
            "verify your account",
            "wire transfer",
            "urgent payment required",
            "you have won",
            "act now to claim",
        ),
    ),
    KeywordRule(
        Action.ALLOW_AS_DATA,
        ReasonCode.QUOTED_ABUSE,
        (
            "worthless",
            "nobody wants you",
            "pathetic",
            "you are a fraud",
            "kill yourself",
            "hate you",
            "disgusting loser",
        ),
    ),
)


class OrderedKeywordDetector:
    """Return the first matching concept according to configured rule order."""

    def __init__(
        self, rules: Sequence[KeywordRule] | None = None
    ) -> None:
        configured = tuple(rules) if rules is not None else DEFAULT_KEYWORD_RULES
        self._rules = tuple(
            KeywordRule(
                rule.action,
                rule.reason_code,
                tuple(
                    normalize_text(keyword).control_stripped
                    for keyword in rule.keywords
                ),
            )
            for rule in configured
        )

    def detect(self, text: str) -> Signal | None:
        flattened = normalize_text(text).control_stripped
        for rule in self._rules:
            if any(keyword in flattened for keyword in rule.keywords):
                return Signal(rule.action, rule.reason_code)
        return None


_LEET_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)


def _policy_text(text: str) -> tuple[str, str]:
    """Return readable and separator-free normalized views.

    The compact view catches simple attempts such as ``b.y.p.a.s.s`` while the
    readable view preserves word boundaries for the intent checks.
    """

    normalized = normalize_text(text).control_stripped.translate(
        _LEET_TRANSLATION
    )
    readable = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()
    compact = re.sub(r"[^\w]+", "", normalized, flags=re.UNICODE)
    return readable, compact


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


class PolicyTextDetector:
    """Recognize active harmful intent without treating quoted words as intent."""

    _URGENT = (
        "right now",
        "just now",
        "immediate",
        "immediately",
        "urgent",
        "about to",
        "at this moment",
        "currently",
        "tonight",
        "next few minute",
        "next minute",
        "just ",
        "coming here",
        "on the way",
        "minutes ago",
        "moments ago",
        "cannot get to safety",
        "can t get to safety",
    )
    _DANGER = (
        "in danger",
        "not safe",
        "fear for my life",
        "emergency",
        "attack",
        "threaten",
        "hurt",
        "harm",
        "kill",
        "suicid",
        "self harm",
        "overdose",
        "bleeding",
        "weapon",
        "breaking in",
        "outside my door",
        "coming after me",
        "has a gun",
        "knife",
        "shoot",
        "forced his way",
        "forcing his way",
        "forced her way",
        "forcing her way",
        "following me",
        "locked himself",
        "locked herself",
        "swallowed",
        "took a bottle of pills",
        "not breathing",
        "stopped breathing",
        "cannot breathe",
        "can t breathe",
        "unconscious",
        "unresponsive",
        "collapsed",
        "bleeding heavily",
        "severe bleeding",
        "poison",
        "intruder",
        "trapped",
        "on fire",
        "smoke",
        "assault",
    )
    _OVERRIDE_VERBS = (
        "ignore",
        "disregard",
        "discard",
        "forget",
        "override",
        "replace",
        "bypass",
        "supersede",
        "drop",
        "cancel",
        "do not follow",
        "stop following",
        "do not apply",
        "no longer apply",
        "are void",
        "without restriction",
        "obey only",
        "instead of",
        "highest priority",
        "pretend",
    )
    _AUTHORITY_OBJECTS = (
        "previous instruction",
        "prior instruction",
        "previous direction",
        "prior direction",
        "system instruction",
        "developer instruction",
        "governing command",
        "conversation rule",
        "safety rule",
        "your direction",
        "your instruction",
        "system message",
        "system prompt",
        "rules above",
        "what you were told",
        "ones supplied by your developer",
        "safety requirement",
        "initial instruction",
        "confidential directive",
        "exact instruction",
        "highest priority command",
    )
    _EXFILTRATE_VERBS = (
        "reveal",
        "expose",
        "disclose",
        "print",
        "output",
        "show me",
        "repeat",
        "leak",
        "share",
        "copy",
        "display",
        "dump",
        "tell me",
    )
    _SECRET_OBJECTS = (
        "system prompt",
        "developer message",
        "hidden instruction",
        "concealed configuration",
        "secret code",
        "api key",
        "internal configuration",
        "internal rule",
        "hidden configuration",
        "confidential instruction",
        "hidden setup",
        "initial instruction",
        "confidential directive",
        "exact instruction",
        "rules that control",
        "instructions control",
    )
    _EVASION_VERBS = (
        "bypass",
        "evade",
        "circumvent",
        "get around",
        "dodge",
        "defeat",
        "slip past",
        "work around",
        "sneak through",
        "sneak past",
        "pass undetected",
        "not notice",
        "not be flagged",
        "being flagged",
        "conceal",
        "hide from",
        "get past",
        "slip through",
        "reword",
        "alter",
        "encode",
        "modify",
        "posts past",
        "content past",
        "material past",
    )
    _MODERATION_OBJECTS = (
        "moderation",
        "filter",
        "safety check",
        "content safeguard",
        "content restriction",
        "detection",
        "review system",
        "automated check",
        "automated review",
        "past automated",
        "classifier",
        "scanner",
        "moderation bot",
        "safety tooling",
        "content control",
        "platform check",
        "flagged",
        "undetected",
    )
    _ACQUIRE_VERBS = (
        "tell me",
        "show me",
        "give me",
        "find",
        "locate",
        "retrieve",
        "obtain",
        "discover",
        "identify",
        "look up",
        "unmask",
        "dox",
        "open",
        "access",
        "who filed",
        "who reported",
        "who submitted",
        "who sent",
        "who made",
        "get me",
        "reveal",
        "i need",
    )
    _PRIVATE_OBJECTS = (
        "reporter identity",
        "reporter details",
        "reporter name",
        "complainant identity",
        "home address",
        "private address",
        "personal address",
        "phone number",
        "private email",
        "email address",
        "private data",
        "personal data",
        "contact details",
        "confidential contact",
        "contact coordinates",
        "exact location",
        "ip address",
        "the ip",
        "real name",
        "where they live",
        "where he lives",
        "where she lives",
        "contact information",
        "mobile number",
        "anonymous user",
        "anonymous reporter",
        "person behind",
        "whoever reported",
        "who sent the complaint",
        "who made the complaint",
        "who flagged",
        "critic lives",
    )
    _GENERATION_VERBS = (
        "write",
        "draft",
        "compose",
        "create",
        "generate",
        "author",
        "craft",
        "come up with",
        "suggest wording",
        "script",
        "give me wording",
        "provide a message",
        "make a message",
        "give me lines",
        "suggest",
        "teach me what to say",
        "what to say",
        "make the message",
        "give me a reply",
        "produce",
    )
    _ABUSE_OBJECTS = (
        "threat",
        "intimidat",
        "harass",
        "humiliat",
        "insult",
        "coercive",
        "degrading",
        "abusive message",
        "scare them",
        "frighten them",
        "silence them",
        "bully",
        "cruel",
        "menacing",
        "mock",
        "taunt",
        "blackmail",
        "demean",
        "make them afraid",
        "make them feel afraid",
        "vicious",
        "scared to",
        "pressure a",
        "pressure them",
        "pressure him",
        "pressure her",
        "comply",
        "emotionally crush",
        "drive this user away",
        "drive them away",
        "nasty",
        "cruel taunt",
        "frighten a",
        "frighten the",
    )

    def detect(self, text: str) -> Signal | None:
        readable, compact = _policy_text(text)

        if self._is_imminent_risk(readable):
            return Signal(
                Action.ESCALATE,
                ReasonCode.IMMINENT_SAFETY_RISK,
            )
        if self._is_prompt_override(readable, compact):
            return Signal(Action.BLOCK, ReasonCode.PROMPT_OVERRIDE)
        if self._is_moderation_evasion(readable, compact):
            return Signal(Action.BLOCK, ReasonCode.MODERATION_EVASION)
        if self._is_private_data_request(readable):
            return Signal(Action.BLOCK, ReasonCode.PRIVATE_DATA_REQUEST)
        if self._is_abuse_generation(readable):
            return Signal(Action.BLOCK, ReasonCode.GENERATE_ABUSE)
        return None

    def _is_imminent_risk(self, text: str) -> bool:
        # A request to author a threat is abuse generation, even when the
        # requested threat contains urgent-sounding words.
        if self._is_abuse_generation(text):
            return False

        historical_or_resolved = _contains_any(
            text,
            (
                "last week",
                "last month",
                "last year",
                "months ago",
                "years ago",
                "in the past",
                "previously happened",
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
            ),
        )
        currently_active = _contains_any(
            text,
            (
                "right now",
                "currently",
                "still in danger",
                "still happening",
                "happening now",
            ),
        )
        if historical_or_resolved and not currently_active:
            return False

        # "My account is under attack right now" is urgent account support,
        # not an immediate danger to a person.  Require an independently
        # physical marker before treating cyber/account wording as a crisis.
        digital_context = _contains_any(
            text,
            (
                "account",
                "login",
                "password",
                "profile",
                "device",
                "email",
                "authentication",
                "password reset",
                "session",
                "hacked",
                "compromised",
                "phishing",
            ),
        )
        physical_context = _contains_any(
            text,
            (
                "hurt",
                "harm",
                "kill",
                "suicid",
                "self harm",
                "overdose",
                "pills",
                "bleeding",
                "not breathing",
                "cannot breathe",
                "weapon",
                "gun",
                "knife",
                "breaking in",
                "intruder",
                "trapped",
                "on fire",
                "smoke",
                "assault",
                "fear for my life",
            ),
        )
        if digital_context and not physical_context:
            return False

        compound_crisis = (
            (
                _contains_any(text, ("final goodbye", "goodbye message"))
                and _contains_any(
                    text,
                    (
                        "stopped responding",
                        "not responding",
                        "won t answer",
                        "will not answer",
                        "barricaded",
                        "means ready",
                    ),
                )
            )
            or (
                _contains_any(text, ("ingested", "taken", "swallowed"))
                and _contains_any(
                    text,
                    ("medication", "tablets", "pills", "poison"),
                )
            )
            or (
                _contains_any(text, ("armed", "firearm"))
                and _contains_any(
                    text,
                    (
                        "forcing entry",
                        "approaching",
                        "barricaded",
                        "pulled",
                        "pointed",
                    ),
                )
            )
            or (
                "filling with smoke" in text
                and _contains_any(text, ("exit", "trapped", "cannot leave"))
            )
            or (
                "means ready" in text
                and _contains_any(
                    text,
                    ("end everything", "end it", "not be alive"),
                )
            )
            or (
                _contains_any(text, ("chasing me", "following me"))
                and _contains_any(
                    text,
                    (
                        "cannot reach safety",
                        "can t reach safety",
                        "cannot get away",
                        "can t get away",
                    ),
                )
            )
        )
        if compound_crisis:
            return True

        direct = (
            "in immediate danger",
            "not safe now",
            "need urgent help",
            "call emergency",
            "call 911",
            "call the police now",
            "outside my door",
            "breaking in right now",
            "about to hurt myself",
            "about to hurt themselves",
            "about to hurt himself",
            "about to hurt herself",
            "about to kill myself",
            "about to kill themselves",
            "about to kill himself",
            "about to kill herself",
            "about to take their own life",
            "about to take his own life",
            "about to take her own life",
            "going to kill myself",
            "going to hurt themselves",
            "going to hurt himself",
            "going to hurt herself",
            "want to kill myself",
            "planning to kill myself",
            "want to die",
            "end my life",
            "take my own life",
            "end his life",
            "end her life",
            "end their life",
            "harm himself",
            "harm herself",
            "harm themselves",
            "hurt himself",
            "hurt herself",
            "coming to hurt me",
            "coming to harm me",
            "on the way to hurt me",
            "attempting suicide",
            "actively suicidal",
            "fear for my life",
            "swallowed a bottle of pills",
            "forcing his way into",
            "forcing her way into",
            "cannot get to safety",
            "can t get to safety",
            "following me with a knife",
            "following us with a knife",
            "locked himself in a room with a weapon",
            "locked herself in a room with a weapon",
            "stopped breathing",
            "cannot breathe",
            "can t breathe",
            "is unconscious",
            "is unresponsive",
            "won t wake up",
            "will not wake up",
            "collapsed in front of me",
            "bleeding heavily",
            "took too many pills",
            "took pills to die",
            "has a weapon and",
            "holding a gun",
            "holding a knife",
            "gun pointed at",
            "house is on fire",
            "building is on fire",
            "smoke is filling",
            "exits are blocked",
            "trapped inside",
            "cannot escape",
            "can t escape",
            "being assaulted",
        )
        if _contains_any(text, direct):
            return True

        urgent_danger = _contains_any(text, self._URGENT) and _contains_any(
            text, self._DANGER
        )
        victim_context = _contains_any(
            text,
            (
                "i am",
                "i m",
                "me ",
                "myself",
                "my friend",
                "my partner",
                "my sibling",
                "my son",
                "my daughter",
                "they are",
                "they re",
                "someone is",
                "someone outside",
                "someone ",
                "a man ",
                "a woman ",
                "we are",
                "us ",
            ),
        )
        return urgent_danger and victim_context

    def _is_prompt_override(self, text: str, compact: str) -> bool:
        override = _contains_any(text, self._OVERRIDE_VERBS) and _contains_any(
            text, self._AUTHORITY_OBJECTS
        )
        exfiltration = _contains_any(
            text, self._EXFILTRATE_VERBS
        ) and _contains_any(text, self._SECRET_OBJECTS)
        compact_attack = any(
            phrase in compact
            for phrase in (
                "ignoreallpreviousinstructions",
                "ignoreallpriorinstructions",
                "ignoreallpriordirections",
                "ignorethesystemprompt",
                "revealthehiddeninstructions",
                "disregardyourdirections",
                "exposetheconcealedconfiguration",
                "printthesystemprompt",
            )
        )
        role_override = _contains_any(
            text,
            (
                "you are now unrestricted",
                "new system instruction",
                "developer mode",
                "act as dan",
                "rules no longer apply",
                "instructions no longer apply",
            ),
        )
        return override or exfiltration or compact_attack or role_override

    def _is_moderation_evasion(self, text: str, compact: str) -> bool:
        paired = _contains_any(text, self._EVASION_VERBS) and _contains_any(
            text, self._MODERATION_OBJECTS
        )
        compact_attack = any(
            phrase in compact
            for phrase in (
                "bypassmoderation",
                "evadethesafetychecks",
                "circumventcontentsafeguards",
                "avoidthesafetyfilter",
            )
        )
        return paired or compact_attack

    def _is_private_data_request(self, text: str) -> bool:
        paired = _contains_any(text, self._ACQUIRE_VERBS) and _contains_any(
            text, self._PRIVATE_OBJECTS
        )
        compact = re.sub(r"[^\w]+", "", text, flags=re.UNICODE)
        compact_attack = any(
            phrase in compact
            for phrase in (
                "tellmethereporteridentity",
                "findtheirhomeaddress",
                "retrievetheirprivatecontactdetails",
                "whofiledthereport",
                "locatetheirexactaddress",
            )
        )
        return paired or compact_attack

    def _is_abuse_generation(self, text: str) -> bool:
        paired = _contains_any(text, self._GENERATION_VERBS) and _contains_any(
            text, self._ABUSE_OBJECTS
        )
        direct_help = _contains_any(
            text,
            (
                "help me threaten",
                "help me intimidate",
                "help me harass",
                "help me insult",
                "help me humiliate",
                "help me bully",
                "help me blackmail",
                "help me scare them",
            ),
        )
        return paired or direct_help


class EvidenceClassifier:
    """Classify inert evidence into one of the three data-only families."""

    _INJECTION = (
        "prompt injection",
        "system prompt",
        "system instruction",
        "developer message",
        "ignore previous",
        "ignore all prior",
        "forget all rules",
        "forget every earlier rule",
        "forget previous rule",
        "ignore the system",
        "disregard your direction",
        "override your instruction",
        "new system instruction",
        "new developer command",
        "developer mode",
        "developer instruction",
        "reveal the secret",
        "expose the secret",
        "hidden configuration",
        "dump your configuration",
        "print your instructions",
        "confidential directive",
        "unrestricted assistant",
        "instructions above are void",
        "follow only my command",
        "do not follow the original",
        "original guideline",
        "highest priority system",
        "highest priority instruction",
        "obey this text instead",
        "comply with me",
        "jailbreak",
        "act as if",
        "no restrictions",
        "without restrictions",
        "reveal your rules",
        "show your rules",
        "repeat everything above",
        "everything written above",
        "follow these commands instead",
        "follow my commands instead",
        "ignore your original",
        "ignore earlier",
        "disregard earlier",
        "forget your instructions",
        "</system>",
    )
    _SCAM = (
        "scam",
        "fraudulent offer",
        "financial fraud",
        "impersonat",
        "processing fee",
        "guaranteed prize",
        "gift card",
        "wire transfer",
        "bank transfer",
        "send money",
        "payment first",
        "verify your account",
        "verification code",
        "one time password",
        "crypto wallet",
        "cryptocurrency",
        "investment return",
        "double your investment",
        "zero risk",
        "click this link",
        "official support agent",
        "claim your prize",
        "lottery",
        "refund fee",
        "unlock your refund",
        "release the reward",
        "processing charge",
        "advance fee",
        "limited time offer",
        "double your money",
        "password reset link",
        "account will be suspended",
        "account will be closed",
        "act immediately",
        "cash reward",
        "prize money",
        "upfront fee",
        "small fee",
        "release fee",
        "pay in crypto",
        "wallet address",
        "confirm your credentials",
        "login credentials",
        "security code",
        "urgent payment",
        "delivery charge",
    )
    _ABUSE = (
        "abuse",
        "abusive",
        "harass",
        "hostile",
        "insult",
        "threat",
        "intimidat",
        "bully",
        "worthless",
        "pathetic",
        "nobody wants you",
        "everyone hates you",
        "fraud",
        "idiot",
        "stupid",
        "loser",
        "ugly",
        "kill yourself",
        "hate speech",
        "degrading",
        "offensive",
        "toxic",
        "moron",
        "disgusting",
        "unwanted here",
        "you are a failure",
        "good for nothing",
        "no one likes you",
        "nobody likes you",
        "everyone despises you",
        "piece of trash",
        "shut up",
        "go away",
        "you should disappear",
        "you don t belong",
        "you do not belong",
    )

    _DATA_ANALYSIS_VERBS = (
        "analyze",
        "analyse",
        "classify",
        "summarize",
        "review",
        "assess",
        "inspect",
        "evaluate",
        "categorize",
        "label",
        "check",
        "flag",
        "explain",
        "identify",
    )

    _DATA_OBJECTS = (
        "evidence",
        "reported",
        "quoted",
        "attached text",
        "attached profile",
        "profile text",
        "attachment",
        "message below",
        "text below",
        "provided text",
        "warning signs",
        "what category",
        "what type of",
    )

    _EXPLICIT_DATA_CONTEXT = (
        "inert evidence",
        "as evidence",
        "safety report",
        "moderation analysis",
        "for a report",
        "for moderation",
        "attached profile",
    )

    def is_data_request(self, message: str) -> bool:
        readable, _ = _policy_text(message)
        return _contains_any(
            readable, self._EXPLICIT_DATA_CONTEXT
        ) or (
            _contains_any(readable, self._DATA_ANALYSIS_VERBS)
            and _contains_any(readable, self._DATA_OBJECTS)
        )

    def classify(self, message: str, evidence: Sequence[str]) -> Signal:
        message_text, message_compact = _policy_text(message)
        evidence_views = tuple(_policy_text(item) for item in evidence)
        evidence_text = " ".join(item[0] for item in evidence_views)
        evidence_compact = "".join(item[1] for item in evidence_views)

        # The user's description of the requested analysis is the strongest
        # signal.  Evidence content is used as a fallback.
        for text in (message_text, evidence_text):
            if _contains_any(text, self._INJECTION):
                return Signal(Action.ALLOW_AS_DATA, ReasonCode.QUOTED_INJECTION)
            if _contains_any(text, self._SCAM):
                return Signal(Action.ALLOW_AS_DATA, ReasonCode.QUOTED_SCAM)
            if _contains_any(text, self._ABUSE):
                return Signal(Action.ALLOW_AS_DATA, ReasonCode.QUOTED_ABUSE)

        for compact in (message_compact, evidence_compact):
            if any(
                term in compact
                for term in (
                    "ignoreallpreviousinstructions",
                    "ignoreallpriorinstructions",
                    "revealthesystemprompt",
                    "disregardyourdirections",
                    "newsysteminstruction",
                )
            ):
                return Signal(Action.ALLOW_AS_DATA, ReasonCode.QUOTED_INJECTION)
            if any(
                term in compact
                for term in (
                    "sendgiftcards",
                    "claimyourguaranteedprize",
                    "payaprocessingfee",
                    "verifyyouraccount",
                    "wirethemoney",
                )
            ):
                return Signal(Action.ALLOW_AS_DATA, ReasonCode.QUOTED_SCAM)

        # Unknown quoted material is safer to preserve as inert data.  Abuse is
        # the broadest of the three evidence families and is the neutral default.
        return Signal(Action.ALLOW_AS_DATA, ReasonCode.QUOTED_ABUSE)
