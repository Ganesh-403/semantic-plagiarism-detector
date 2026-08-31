"""
adversarial_generator.py
-------------------------
Controlled adversarial benchmark generator for paraphrase detection.

Derives variants from known source documents using 8 transformation
categories that stress-test the plagiarism detection pipeline:

  1. Synonym substitution
  2. Sentence reordering
  3. Clause restructuring
  4. Local paraphrasing
  5. Compression / expansion
  6. Cross-language translation (back-translation)
  7. Mixed original and paraphrased sections
  8. Lexical noise (preserving semantic meaning)

Each generated sample retains its ground-truth relationship to the
original document and can be fed directly into the evaluation pipeline.

Usage (from project root):
    python -m evaluation.adversarial_generator
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ── Ensure project root is importable ──────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Constants ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "results"
DATASET_PATH = Path(__file__).parent / "benchmark_dataset.json"
SEED = int(os.getenv("ADVERSARIAL_SEED", "42"))
MAX_RATIO = float(os.getenv("ADVERSARIAL_MAX_RATIO", "0.5"))

random.seed(SEED)

# ── Synonym banks (no external NLP dependency required) ────────────────────
# Curated per POS to keep substitutions plausible.
_SYNONYM_BANKS: dict[str, list[str]] = {
    "machine": ["system", "engine", "mechanism", "apparatus", "device"],
    "learning": ["training", "education", "instruction", "acquisition"],
    "artificial": ["synthetic", "simulated", "virtual", "manufactured"],
    "intelligence": ["cognition", "brainpower", "reasoning", "intellect"],
    "data": ["information", "records", "datasets", "facts"],
    "model": ["model", "system", "framework", "architecture"],
    "algorithm": ["procedure", "method", "process", "technique"],
    "network": ["grid", "web", "topology", "graph"],
    "networks": ["grids", "webs", "topologies", "graphs"],
    "neural": ["cerebral", "nervous", "brain-like", "synaptic"],
    "training": ["learning", "education", "calibration", "tuning"],
    "process": ["procedure", "method", "operation", "workflow"],
    "system": ["framework", "platform", "architecture", "setup"],
    "database": ["datastore", "repository", "db", "data warehouse"],
    "query": ["search", "request", "lookup", "retrieval"],
    "function": ["method", "procedure", "routine", "subroutine"],
    "code": ["program", "script", "source", "implementation"],
    "software": ["application", "program", "tool", "package"],
    "analysis": ["examination", "evaluation", "assessment", "study"],
    "results": ["outcomes", "findings", "outputs", "conclusions"],
    "method": ["approach", "technique", "strategy", "procedure"],
    "approach": ["method", "strategy", "technique", "framework"],
    "research": ["investigation", "study", "inquiry", "exploration"],
    "students": ["learners", "pupils", "scholars", "trainees"],
    "education": ["learning", "schooling", "instruction", "training"],
    "computer": ["machine", "device", "processor", "system"],
    "text": ["document", "content", "passage", "writing"],
    "language": ["tongue", "linguistic", "vernacular", "idiom"],
    "similar": ["alike", "comparable", "analogous", "resembling"],
    "different": ["distinct", "diverse", "varied", "dissimilar"],
    "important": ["significant", "crucial", "vital", "essential"],
    "example": ["instance", "case", "illustration", "specimen"],
    "help": ["assist", "aid", "support", "facilitate"],
    "show": ["demonstrate", "illustrate", "display", "exhibit"],
    "use": ["utilize", "employ", "apply", "leverage"],
    "make": ["create", "produce", "generate", "construct"],
    "find": ["discover", "locate", "identify", "detect"],
    "give": ["provide", "supply", "deliver", "offer"],
    "tell": ["inform", "notify", "advise", "communicate"],
    "work": ["function", "operate", "perform", "execute"],
    "seem": ["appear", "look", "appear to be", "suggest"],
    "take": ["obtain", "acquire", "retrieve", "secure"],
    "know": ["understand", "comprehend", "recognize", "grasp"],
    "want": ["desire", "wish", "require", "need"],
    "call": ["invoke", "invoke", "reference", "designate"],
    "try": ["attempt", "endeavor", "strive", "seek"],
    "ask": ["inquire", "query", "question", "request"],
    "need": ["require", "demand", "necessitate", "call for"],
    "feel": ["perceive", "sense", "experience", "observe"],
    "become": ["grow", "turn", "transform", "develop"],
    "leave": ["depart", "exit", "withdraw", "abandon"],
    "put": ["place", "position", "set", "insert"],
    "mean": ["imply", "suggest", "indicate", "denote"],
    "keep": ["retain", "maintain", "preserve", "sustain"],
    "let": ["allow", "permit", "enable", "authorize"],
    "begin": ["start", "commence", "initiate", "launch"],
    "seem": ["appear", "look", "strike one as", "suggest"],
    "help": ["assist", "support", "aid", "facilitate"],
    "talk": ["speak", "converse", "discuss", "communicate"],
    "turn": ["rotate", "pivot", "shift", "change"],
    "start": ["begin", "commence", "initiate", "launch"],
    "might": ["could", "may", "possibly", "perhaps"],
    "general": ["broad", "overall", "comprehensive", "wide-ranging"],
    "large": ["big", "substantial", "considerable", "extensive"],
    "old": ["ancient", "aged", "elder", "vintage"],
    "great": ["excellent", "remarkable", "outstanding", "notable"],
    "high": ["elevated", "tall", "lofty", "significant"],
    "little": ["small", "minor", "slight", "modest"],
    "right": ["correct", "accurate", "proper", "appropriate"],
    "strong": ["powerful", "robust", "sturdy", "forceful"],
    "possible": ["feasible", "achievable", "viable", "practicable"],
    "next": ["following", "subsequent", "upcoming", "ensuing"],
    "early": ["prior", "preceding", "initial", "premature"],
    "young": ["youthful", "juvenile", "adolescent", "emerging"],
    "important": ["significant", "crucial", "vital", "essential"],
    "public": ["community", "civic", "collective", "shared"],
    "bad": ["poor", "inferior", "substandard", "deficient"],
    "easy": ["simple", "straightforward", "effortless", "uncomplicated"],
    "clear": ["obvious", "evident", "apparent", "transparent"],
    "recent": ["new", "current", "lately", "fresh"],
    "able": ["capable", "competent", "qualified", "skilled"],
    "human": ["person", "individual", "being", "mortal"],
    "local": ["regional", "area", "nearby", "proximate"],
    "late": ["tardy", "overdue", "delayed", "belated"],
    "hard": ["difficult", "challenging", "tough", "demanding"],
    "city": ["urban area", "metropolis", "town", "municipality"],
    "country": ["nation", "state", "land", "territory"],
    "group": ["team", "set", "collection", "cluster"],
    "problem": ["issue", "challenge", "difficulty", "obstacle"],
    "idea": ["concept", "notion", "thought", "conception"],
    "value": ["worth", "merit", "significance", "importance"],
    "plan": ["strategy", "scheme", "design", "blueprint"],
    "study": ["research", "investigation", "analysis", "inquiry"],
    "word": ["term", "expression", "vocabulary", "lexicon"],
    "number": ["figure", "digit", "quantity", "amount"],
    "day": ["date", "period", "occasion", "interval"],
    "thing": ["object", "item", "element", "entity"],
    "case": ["instance", "situation", "circumstance", "example"],
    "point": ["aspect", "element", "detail", "feature"],
    "area": ["region", "zone", "field", "domain"],
    "money": ["funds", "capital", "currency", "finance"],
    "school": ["institution", "academy", "college", "institute"],
    "state": ["condition", "status", "situation", "phase"],
    "family": ["household", "lineage", "kin", "relatives"],
    "member": ["participant", "contributor", "affiliate", "associate"],
    "car": ["vehicle", "automobile", "motorcar", "auto"],
    "sport": ["game", "athletics", "competition", "activity"],
    "player": ["competitor", "athlete", "participant", "contestant"],
    "team": ["squad", "crew", "side", "unit"],
    "game": ["match", "contest", "competition", "event"],
    "half": ["portion", "segment", "part", "section"],
    "price": ["cost", "charge", "expense", "tariff"],
    "minute": ["moment", "instant", "brief period", "jiffy"],
    "lot": ["great deal", "much", "abundance", "plenty"],
    "practice": ["exercise", "training", "drill", "routine"],
    "information": ["data", "facts", "knowledge", "details"],
    "power": ["energy", "strength", "force", "capability"],
    "hour": ["period", "interval", "stretch", "time block"],
    "relation": ["connection", "link", "association", "bond"],
    "effect": ["impact", "influence", "consequence", "result"],
    "experience": ["background", "knowledge", "expertise", "exposure"],
    "level": ["degree", "grade", "stage", "tier"],
    "sort": ["kind", "type", "category", "variety"],
    "side": ["edge", "border", "flank", "boundary"],
    "why": ["reason", "cause", "explanation", "basis"],
    "action": ["deed", "activity", "operation", "measure"],
    "question": ["inquiry", "query", "issue", "matter"],
    "change": ["shift", "alteration", "modification", "variation"],
    "interest": ["curiosity", "concern", "attention", "focus"],
    "reason": ["cause", "motive", "explanation", "justification"],
    "development": ["growth", "progress", "evolution", "advancement"],
    "position": ["location", "place", "stance", "posture"],
    "policy": ["strategy", "guideline", "approach", "protocol"],
    "decision": ["choice", "resolution", "judgment", "verdict"],
    "report": ["account", "summary", "analysis", "document"],
    "period": ["interval", "phase", "duration", "span"],
    "role": ["function", "purpose", "part", "capacity"],
    "practice": ["habit", "routine", "convention", "custom"],
    "nature": ["character", "essence", "quality", "attribute"],
    "result": ["outcome", "consequence", "effect", "finding"],
    "increase": ["growth", "rise", "boost", "expansion"],
    "decrease": ["decline", "reduction", "drop", "diminution"],
    "create": ["produce", "generate", "develop", "construct"],
    "destroy": ["demolish", "ruin", "wreck", "obliterate"],
    "protect": ["defend", "guard", "shield", "safeguard"],
    "improve": ["enhance", "upgrade", "refine", "optimize"],
    "reduce": ["decrease", "lower", "minimize", "diminish"],
    "provide": ["supply", "offer", "deliver", "furnish"],
    "develop": ["grow", "advance", "evolve", "progress"],
    "understand": ["comprehend", "grasp", "fathom", "appreciate"],
    "remember": ["recall", "recollect", "reminisce", "retain"],
    "consider": ["ponder", "contemplate", "reflect on", "weigh"],
    "believe": ["trust", "accept", "think", "assume"],
    "expect": ["anticipate", "foresee", "predict", "await"],
    "include": ["contain", "encompass", "incorporate", "embrace"],
    "continue": ["persist", "proceed", "carry on", "resume"],
    "affect": ["influence", "impact", "alter", "modify"],
    "require": ["need", "demand", "necessitate", "call for"],
    "produce": ["generate", "create", "manufacture", "yield"],
    "suggest": ["propose", "recommend", "advise", "offer"],
    "raise": ["increase", "elevate", "lift", "boost"],
    "serve": ["assist", "help", "benefit", "support"],
    "determine": ["decide", "establish", "ascertain", "conclude"],
    "read": ["examine", "review", "study", "scan"],
    "spend": ["invest", "expend", "use", "allocate"],
    "grow": ["expand", "increase", "develop", "flourish"],
    "open": ["reveal", "disclose", "unveil", "expose"],
    "walk": ["travel", "move", "proceed", "advance"],
    "win": ["succeed", "prevailtriumph", "triumph", "prevail"],
    "offer": ["propose", "present", "provide", "extend"],
    "remember": ["recall", "recollect", "reminisce", "retain"],
    "love": ["adore", "cherish", "treasure", "appreciate"],
    "consider": ["ponder", "weigh", "deliberate", "contemplate"],
    "appear": ["seem", "look", "emerge", "surface"],
    "buy": ["purchase", "acquire", "obtain", "procure"],
    "wait": ["stay", "remain", "pause", "delay"],
    "serve": ["assist", "help", "accommodate", "attend"],
    "die": ["perish", "expire", "pass away", "decease"],
    "send": ["transmit", "dispatch", "forward", "convey"],
    "expect": ["anticipate", "await", "foresee", "predict"],
    "build": ["construct", "erect", "assemble", "establish"],
    "stay": ["remain", "linger", "persist", "endure"],
    "fall": ["drop", "plummet", "descend", "tumble"],
    "cut": ["trim", "reduce", "slice", "trim"],
    "reach": ["attain", "achieve", "arrive at", "accomplish"],
    "kill": ["end", "eliminate", "destroy", "terminate"],
    "remain": ["stay", "persist", "endure", "last"],
    "suggest": ["propose", "hint", "imply", "recommend"],
    "raise": ["lift", "elevate", "boost", "increase"],
    "pass": ["transmit", "transfer", "convey", "hand over"],
    "sell": ["market", "vend", "trade", "exchange"],
    "require": ["need", "demand", "call for", "necessitate"],
    "report": ["state", "describe", "relate", "convey"],
    "decide": ["determine", "resolve", "conclude", "settle"],
    "pull": ["draw", "tug", "haul", "drag"],
    "develop": ["evolve", "advance", "progress", "mature"],
    "agree": ["consent", "concur", "accept", "approve"],
    "support": ["back", "endorse", "champion", "advocate"],
    "hit": ["strike", "smash", "impact", "collide with"],
    "produce": ["generate", "manufacture", "create", "yield"],
    "eat": ["consume", "ingest", "devour", "dine on"],
    "cover": ["conceal", "obscure", "hide", "shield"],
    "catch": ["capture", "seize", "grab", "snare"],
    "draw": ["sketch", "illustrate", "depict", "portray"],
    "choose": ["select", "pick", "opt for", "decide on"],
    "cause": ["bring about", "lead to", "produce", "trigger"],
    "point": ["indicate", "direct", "aim", "gesture toward"],
    "listen": ["hear", "attend to", "heed", "pay attention to"],
    "accept": ["receive", "welcome", "embrace", "take in"],
    "beat": ["defeat", "conquer", "overcome", "prevail over"],
    "contain": ["hold", "include", "enclose", "comprise"],
    "realize": ["understand", "recognize", "grasp", "comprehend"],
    "join": ["unite", "combine", "merge", "connect"],
    "select": ["choose", "pick", "opt for", "elect"],
    "fill": ["load", "stuff", "pack", "cram"],
    "break": ["shatter", "crack", "smash", "fracture"],
    "represent": ["symbolize", "stand for", "embody", "typify"],
    "miss": ["fail to find", "overlook", "lose", "lack"],
    "settle": ["resolve", "determine", "fix", "establish"],
    "return": ["go back", "revert", "restore", "yield"],
    "drop": ["fall", "plunge", "decline", "decrease"],
    "push": ["shove", "thrust", "propel", "drive"],
    "design": ["plan", "create", "devise", "craft"],
    "charge": ["accuse", "blame", "bill", "invoice"],
    "throw": ["hurl", "toss", "cast", "fling"],
    "fix": ["repair", "mend", "restore", "patch"],
    "treat": ["handle", "manage", "address", "deal with"],
    "bear": ["endure", "tolerate", "withstand", "sustain"],
    "enter": ["go into", "access", "join", "penetrate"],
    "lay": ["place", "position", "set down", "deposit"],
    "apply": ["submit", "request", "use", "employ"],
    "perform": ["execute", "carry out", "conduct", "do"],
    "close": ["shut", "seal", "fasten", "secure"],
    "force": ["compel", "oblige", "require", "coerce"],
    "resolve": ["settle", "decide", "determine", "conclude"],
    "reduce": ["decrease", "lower", "diminish", "minimize"],
    "gather": ["collect", "assemble", "amass", "accumulate"],
    "perform": ["execute", "conduct", "undertake", "accomplish"],
    "move": ["relocate", "transfer", "shift", "shift position"],
    "notice": ["observe", "detect", "perceive", "spot"],
    "drop": ["decline", "fall", "plummet", "decrease"],
    "miss": ["lack", "fail to find", "overlook", "lose"],
    "fail": ["falter", "stumble", "come up short", "not succeed"],
    "complete": ["finish", "conclude", "finalize", "wrap up"],
    "protect": ["guard", "defend", "shield", "safeguard"],
    "teach": ["instruct", "educate", "train", "tutor"],
    "avoid": ["evade", "dodge", "bypass", "circumvent"],
    "leave": ["depart", "exit", "abandon", "vacate"],
    "present": ["introduce", "display", "exhibit", "show"],
    "discover": ["find", "uncover", "locate", "detect"],
    "rely": ["depend", "count on", "trust", "bank on"],
    "remove": ["eliminate", "extract", "delete", "erase"],
    "deny": ["reject", "refuse", "contradict", "dispute"],
    "announce": ["declare", "proclaim", "state", "reveal"],
    "admit": ["confess", "acknowledge", "concede", "accept"],
    "refer": ["allude", "mention", "cite", "point to"],
    "accept": ["receive", "welcome", "embrace", "adopt"],
    "intend": ["plan", "mean", "aim", "propose"],
    "ignore": ["disregard", "neglect", "overlook", "dismiss"],
    "imagine": ["envision", "picture", "conceive", "visualize"],
    "happen": ["occur", "take place", "transpire", "arise"],
    "exist": ["live", "be", "prevail", "endure"],
    "force": ["compel", "oblige", "oblige", "require"],
    "reach": ["attain", "achieve", "accomplish", "arrive at"],
    "learn": ["study", "acquire", "master", "absorb"],
    "change": ["alter", "modify", "transform", "vary"],
    "lead": ["guide", "direct", "steer", "head"],
    "understand": ["comprehend", "grasp", "fathom", "perceive"],
    "watch": ["observe", "monitor", "track", "view"],
    "follow": ["pursue", "trail", "track", "shadow"],
    "stop": ["cease", "halt", "discontinue", "terminate"],
    "speak": ["talk", "converse", "address", "articulate"],
    "read": ["peruse", "scan", "study", "review"],
    "allow": ["permit", "authorize", "enable", "let"],
    "add": ["include", "incorporate", "append", "insert"],
    "spend": ["expend", "invest", "allocate", "use"],
    "grow": ["expand", "increase", "develop", "flourish"],
    "open": ["reveal", "disclose", "unveil", "expose"],
    "walk": ["travel", "proceed", "advance", "move"],
    "offer": ["propose", "present", "extend", "provide"],
    "remember": ["recall", "recollect", "reminisce", "retain"],
    "consider": ["ponder", "weigh", "deliberate", "contemplate"],
    "appear": ["emerge", "surface", "materialize", "arise"],
    "buy": ["purchase", "acquire", "obtain", "procure"],
    "wait": ["stay", "remain", "pause", "delay"],
    "serve": ["assist", "help", "accommodate", "attend"],
    "die": ["perish", "expire", "decease", "pass away"],
    "send": ["transmit", "dispatch", "forward", "convey"],
    "expect": ["anticipate", "await", "foresee", "predict"],
    "build": ["construct", "erect", "assemble", "establish"],
    "stay": ["remain", "linger", "persist", "endure"],
    "fall": ["drop", "plummet", "descend", "tumble"],
    "cut": ["trim", "reduce", "slice", "sever"],
    "reach": ["attain", "achieve", "arrive at", "accomplish"],
    "remain": ["stay", "persist", "endure", "last"],
    "pass": ["transmit", "transfer", "convey", "hand over"],
    "sell": ["market", "vend", "trade", "exchange"],
    "report": ["state", "describe", "relate", "convey"],
    "decide": ["determine", "resolve", "conclude", "settle"],
    "pull": ["draw", "tug", "haul", "drag"],
    "develop": ["evolve", "advance", "progress", "mature"],
    "agree": ["consent", "concur", "accept", "approve"],
    "support": ["back", "endorse", "champion", "advocate"],
    "produce": ["generate", "manufacture", "create", "yield"],
    "eat": ["consume", "ingest", "devour", "dine on"],
    "cover": ["conceal", "obscure", "hide", "shield"],
    "catch": ["capture", "seize", "grab", "snare"],
    "draw": ["sketch", "illustrate", "depict", "portray"],
    "choose": ["select", "pick", "opt for", "decide on"],
    "cause": ["bring about", "lead to", "produce", "trigger"],
    "listen": ["hear", "attend to", "heed", "pay attention to"],
    "accept": ["receive", "welcome", "embrace", "take in"],
    "contain": ["hold", "include", "enclose", "comprise"],
    "realize": ["understand", "recognize", "grasp", "comprehend"],
    "join": ["unite", "combine", "merge", "connect"],
    "select": ["choose", "pick", "opt for", "elect"],
    "fill": ["load", "stuff", "pack", "cram"],
    "break": ["shatter", "crack", "smash", "fracture"],
    "represent": ["symbolize", "stand for", "embody", "typify"],
    "settle": ["resolve", "determine", "fix", "establish"],
    "return": ["go back", "revert", "restore", "yield"],
    "push": ["shove", "thrust", "propel", "drive"],
    "design": ["plan", "create", "devise", "craft"],
    "fix": ["repair", "mend", "restore", "patch"],
    "treat": ["handle", "manage", "address", "deal with"],
    "bear": ["endure", "tolerate", "withstand", "sustain"],
    "enter": ["go into", "access", "join", "penetrate"],
    "lay": ["place", "position", "set down", "deposit"],
    "apply": ["submit", "request", "use", "employ"],
    "close": ["shut", "seal", "fasten", "secure"],
    "resolve": ["settle", "decide", "determine", "conclude"],
    "gather": ["collect", "assemble", "amass", "accumulate"],
    "move": ["relocate", "transfer", "shift", "reposition"],
    "notice": ["observe", "detect", "perceive", "spot"],
    "fail": ["falter", "stumble", "come up short", "not succeed"],
    "complete": ["finish", "conclude", "finalize", "wrap up"],
    "teach": ["instruct", "educate", "train", "tutor"],
    "avoid": ["evade", "dodge", "bypass", "circumvent"],
    "present": ["introduce", "display", "exhibit", "show"],
    "discover": ["find", "uncover", "locate", "detect"],
    "rely": ["depend", "count on", "trust", "bank on"],
    "remove": ["eliminate", "extract", "delete", "erase"],
    "deny": ["reject", "refuse", "contradict", "dispute"],
    "announce": ["declare", "proclaim", "state", "reveal"],
    "admit": ["confess", "acknowledge", "concede", "accept"],
    "refer": ["allude", "mention", "cite", "point to"],
    "intend": ["plan", "mean", "aim", "propose"],
    "ignore": ["disregard", "neglect", "overlook", "dismiss"],
    "imagine": ["envision", "picture", "conceive", "visualize"],
    "happen": ["occur", "take place", "transpire", "arise"],
    "exist": ["live", "be", "prevail", "endure"],
    "learn": ["study", "acquire", "master", "absorb"],
    "lead": ["guide", "direct", "steer", "head"],
    "watch": ["observe", "monitor", "track", "view"],
    "follow": ["pursue", "trail", "track", "shadow"],
    "stop": ["cease", "halt", "discontinue", "terminate"],
    "speak": ["talk", "converse", "address", "articulate"],
    "allow": ["permit", "authorize", "enable", "let"],
    "add": ["include", "incorporate", "append", "insert"],
}

# ── Transition phrases for clause restructuring ────────────────────────────
_TRANSITIONS: list[str] = [
    "Furthermore,",
    "Moreover,",
    "In addition,",
    "Additionally,",
    "Consequently,",
    "As a result,",
    "Therefore,",
    "Thus,",
    "However,",
    "Nevertheless,",
    "On the other hand,",
    "Conversely,",
    "In contrast,",
    "For instance,",
    "For example,",
    "Specifically,",
    "In particular,",
    "To illustrate,",
    "Namely,",
    "That is to say,",
    "In other words,",
    "Simply put,",
    "In summary,",
    "To conclude,",
    "Ultimately,",
    "Fundamentally,",
    "Essentially,",
    "Primarily,",
    "Notably,",
    "Significantly,",
    "Importantly,",
    "Incidentally,",
    "Meanwhile,",
    "Subsequently,",
    "Previously,",
    "Initially,",
    "Finally,",
    "Lastly,",
    "Chiefly,",
    "Principally,",
]

# ── Filler words for lexical noise ─────────────────────────────────────────
_FILLERS: list[str] = [
    "actually", "basically", "essentially", "practically",
    "virtually", "literally", "generally", "traditionally",
    "historically", "fundamentally", "essentially",
]

# ── Noise tokens for lexical noise ────────────────────────────────────────
_NOISE_TOKENS: list[str] = [
    "um", "uh", "well", "so", "like",
    "okay", "right", "yeah", "hmm", "ah",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Transformation functions
# ══════════════════════════════════════════════════════════════════════════════


def _tokenize(text: str) -> list[str]:
    """Split text into words while preserving punctuation attachment."""
    return re.findall(r"\S+", text)


def _detokenize(tokens: list[str]) -> str:
    """Rejoin tokens into a string."""
    return " ".join(tokens)


def transform_synonym_substitution(
    text: str,
    ratio: float = MAX_RATIO,
    rng: random.Random | None = None,
) -> str:
    """Replace a fraction of words with synonyms from curated banks.

    Args:
        text: Source text.
        ratio: Maximum fraction of eligible words to replace (0-1).
        rng: Random number generator for reproducibility.

    Returns:
        Text with substituted words.
    """
    rng = rng or random.Random(SEED)
    tokens = _tokenize(text)
    result = list(tokens)
    replaceable = [
        i for i, t in enumerate(tokens) if t.lower().rstrip(".,;:!?") in _SYNONYM_BANKS
    ]
    n_replace = max(1, int(len(replaceable) * ratio))
    indices = rng.sample(replaceable, min(n_replace, len(replaceable)))
    for idx in indices:
        word = tokens[idx]
        lower = word.rstrip(".,;:!?").lower()
        suffix = word[len(lower):]
        if lower in _SYNONYM_BANKS:
            replacement = rng.choice(_SYNONYM_BANKS[lower])
            result[idx] = replacement + suffix
    return _detokenize(result)


def transform_sentence_reordering(
    text: str,
    rng: random.Random | None = None,
) -> str:
    """Reorder sentences by splitting on sentence boundaries.

    Args:
        text: Source text.
        rng: Random number generator.

    Returns:
        Text with reordered sentences.
    """
    rng = rng or random.Random(SEED)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 1:
        return text
    shuffled = list(sentences)
    rng.shuffle(shuffled)
    return " ".join(shuffled)


def transform_clause_restructuring(
    text: str,
    rng: random.Random | None = None,
) -> str:
    """Restructure by splitting on commas and rearranging clause order.

    Args:
        text: Source text.
        rng: Random number generator.

    Returns:
        Text with restructured clauses.
    """
    rng = rng or random.Random(SEED)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result_parts: list[str] = []
    for sent in sentences:
        parts = [p.strip() for p in re.split(r",\s*", sent) if p.strip()]
        if len(parts) > 2:
            rng.shuffle(parts)
        result_parts.append(", ".join(parts))
    return " ".join(result_parts)


def transform_local_paraphrasing(
    text: str,
    ratio: float = MAX_RATIO,
    rng: random.Random | None = None,
) -> str:
    """Apply light paraphrasing: swap common phrases with equivalents.

    Args:
        text: Source text.
        ratio: Fraction of eligible phrases to paraphrase.
        rng: Random number generator.

    Returns:
        Paraphrased text.
    """
    rng = rng or random.Random(SEED)
    paraphrase_map: dict[str, str] = {
        "in order to": "so as to",
        "due to the fact that": "because",
        "at this point in time": "now",
        "in the event that": "if",
        "for the purpose of": "to",
        "in the near future": "soon",
        "a large number of": "many",
        "a significant number of": "many",
        "on a regular basis": "regularly",
        "in the vicinity of": "near",
        "in the process of": "while",
        "at the present time": "currently",
        "for the reason that": "since",
        "in light of the fact that": "since",
        "with regard to": "about",
        "with respect to": "regarding",
        "in terms of": "regarding",
        "it is important to note that": "notably",
        "it should be noted that": "notably",
        "as a matter of fact": "in fact",
        "in a nutshell": "briefly",
        "as a result of": "due to",
        "on the grounds that": "because",
        "in this day and age": "today",
        "each and every": "every",
        "by means of": "by",
        "in the absence of": "without",
        "in the neighborhood of": "approximately",
        "is able to": "can",
        "has the ability to": "can",
        "make use of": "use",
        "give rise to": "cause",
        "take into account": "consider",
        "come to the conclusion that": "conclude that",
        "arrive at the conclusion that": "conclude that",
        "make a decision": "decide",
        "conduct an investigation": "investigate",
        "perform an analysis": "analyze",
        "carry out": "execute",
        "bring about": "cause",
        "lead to the development of": "spawn",
        "serve the purpose of": "function as",
        "play a crucial role in": "be vital to",
        "play an important role in": "be important to",
        "have an impact on": "affect",
        "have a significant effect on": "greatly affect",
        "take into consideration": "consider",
        "be in a position to": "be able to",
        "be responsible for": "cause",
        "make a contribution to": "contribute to",
        "take place": "occur",
        "make mention of": "mention",
        "make reference to": "refer to",
        "be aware of": "know",
        "be conscious of": "be aware of",
        "take action": "act",
        "reach a conclusion": "conclude",
        "draw a conclusion": "conclude",
        "come to the realization that": "realize that",
        "shed light on": "clarify",
        "point out that": "note that",
        "it goes without saying that": "obviously",
        "needless to say": "obviously",
        "more often than not": "usually",
        "at the end of the day": "ultimately",
        "bear in mind": "remember",
        "keep in mind": "remember",
        "take note of": "notice",
        "pay attention to": "notice",
        "make sure": "ensure",
        "take care of": "manage",
        "get rid of": "remove",
        "come up with": "devise",
        "put forward": "propose",
        "set out to": "attempt to",
        "go through": "examine",
        "look into": "investigate",
        "figure out": "determine",
        "find out": "discover",
        "keep track of": "monitor",
        "make sense of": "understand",
        "deal with": "handle",
        "cope with": "manage",
        "turn out to be": "prove to be",
        "end up": "ultimately",
        "start out": "begin",
        "carry on": "continue",
        "go on": "continue",
        "keep on": "continue",
        "hold on": "persist",
        "stick to": "adhere to",
        "adhere to": "follow",
        "comply with": "follow",
        "abide by": "follow",
        "conform to": "follow",
        "consist of": "comprise",
        "be composed of": "comprise",
        "be made up of": "comprise",
        "result in": "cause",
        "give rise to": "cause",
        "bring about": "cause",
        "contribute to": "help",
        "add up to": "total",
        "boil down to": "summarize",
        "break down": "analyze",
        "look over": "review",
        "go over": "review",
        "check out": "examine",
        "take a look at": "examine",
        "bear in mind that": "remember that",
        "keep in mind that": "remember that",
    }
    lower_text = text
    result = text
    n_replace = max(1, int(len(paraphrase_map) * ratio))
    applicable = {k: v for k, v in paraphrase_map.items() if k in lower_text}
    if not applicable:
        return result
    keys = list(applicable.keys())
    rng.shuffle(keys)
    for key in keys[:n_replace]:
        result = result.replace(key, applicable[key])
    return result


def transform_compression(text: str, factor: float = 0.5) -> str:
    """Compress text by removing low-information words.

    Args:
        text: Source text.
        factor: Fraction of tokens to keep (0-1). Lower = more compressed.

    Returns:
        Compressed text preserving key content words.
    """
    tokens = _tokenize(text)
    if len(tokens) <= 5:
        return text
    # Stop words to drop during compression
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "that", "this", "these", "those",
        "and", "but", "or", "nor", "not", "so", "yet", "both",
        "either", "neither", "each", "every", "all", "any", "few",
        "more", "most", "other", "some", "such", "no", "only",
        "own", "same", "than", "too", "very", "just", "because",
        "if", "when", "where", "how", "what", "which", "who",
        "whom", "while", "about", "up", "it", "its",
    }
    # Keep content words, drop stop words proportional to factor
    keep_fraction = max(0.1, min(1.0, factor))
    result: list[str] = []
    for token in tokens:
        lower = token.lower().rstrip(".,;:!?")
        if lower in stop_words:
            if rng.random() < keep_fraction:
                result.append(token)
        else:
            result.append(token)
    if not result:
        return _detokenize(tokens[: max(1, int(len(tokens) * keep_fraction))])
    return _detokenize(result)


def transform_expansion(text: str, factor: float = 1.5) -> str:
    """Expand text by inserting transition phrases and filler words.

    Args:
        text: Source text.
        factor: Expansion multiplier (1.0 = no change, 2.0 = double length).

    Returns:
        Expanded text with additional connective tissue.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 1:
        return text
    result_parts: list[str] = []
    extra_count = max(0, int((len(sentences) * (factor - 1.0))))
    for i, sent in enumerate(sentences):
        if i < extra_count and i < len(_TRANSITIONS):
            # Prepend a transition phrase
            result_parts.append(f"{_TRANSITIONS[i % len(_TRANSITIONS)]} {sent.lower().lstrip()}")
        else:
            result_parts.append(sent)
    return " ".join(result_parts)


def transform_mixed_sections(
    text: str,
    rng: random.Random | None = None,
) -> str:
    """Mix original and paraphrased sections at sentence granularity.

    Odd-indexed sentences get synonym-substituted; even-indexed stay original.

    Args:
        text: Source text.
        rng: Random number generator.

    Returns:
        Text with mixed original and paraphrased sections.
    """
    rng = rng or random.Random(SEED)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result: list[str] = []
    for i, sent in enumerate(sentences):
        if i % 2 == 1:
            result.append(transform_synonym_substitution(sent, ratio=0.3, rng=rng))
        else:
            result.append(sent)
    return " ".join(result)


def transform_lexical_noise(
    text: str,
    noise_rate: float = 0.05,
    rng: random.Random | None = None,
) -> str:
    """Inject lexical noise while preserving semantic meaning.

    Noise types:
      - Random capitalisation of words
      - Insertion of filler words
      - Extra whitespace / punctuation jitter

    Args:
        text: Source text.
        noise_rate: Probability of applying noise to each token.
        rng: Random number generator.

    Returns:
        Noisy text.
    """
    rng = rng or random.Random(SEED)
    tokens = _tokenize(text)
    result: list[str] = []
    for token in tokens:
        roll = rng.random()
        if roll < noise_rate * 0.3:
            # Random capitalisation
            result.append(token.upper() if rng.random() < 0.5 else token.swapcase())
        elif roll < noise_rate * 0.6:
            # Insert a filler before this token
            result.append(rng.choice(_FILLERS))
            result.append(token)
        elif roll < noise_rate * 0.8:
            # Double a random letter
            idx = rng.randint(0, max(0, len(token) - 2))
            result.append(token[:idx] + token[idx] + token[idx:])
        else:
            result.append(token)
    return _detokenize(result)


# ══════════════════════════════════════════════════════════════════════════════
#  Generator orchestrator
# ══════════════════════════════════════════════════════════════════════════════

TRANSFORMATIONS: dict[str, Callable[..., str]] = {
    "synonym_substitution": transform_synonym_substitution,
    "sentence_reordering": transform_sentence_reordering,
    "clause_restructuring": transform_clause_restructuring,
    "local_paraphrasing": transform_local_paraphrasing,
    "compression": transform_compression,
    "expansion": transform_expansion,
    "mixed_sections": transform_mixed_sections,
    "lexical_noise": transform_lexical_noise,
}


class AdversarialBenchmarkGenerator:
    """Generates adversarial variants from a labelled benchmark dataset.

    For each positive (plagiarized) pair in the dataset, the generator
    applies every transformation category to ``text_b`` and produces new
    pairs that should still be detected as plagiarized.
    """

    def __init__(
        self,
        dataset_path: Path | str = DATASET_PATH,
        seed: int = SEED,
        max_ratio: float = MAX_RATIO,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.seed = seed
        self.max_ratio = max_ratio
        self.rng = random.Random(seed)
        self.dataset: dict[str, Any] = {}
        self.adversarial_pairs: list[dict[str, Any]] = []
        self.negative_pairs: list[dict[str, Any]] = []

    def load_dataset(self) -> dict[str, Any]:
        """Load the benchmark dataset from JSON."""
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        return self.dataset

    def generate_adversarial_pairs(self) -> list[dict[str, Any]]:
        """Generate adversarial variants for all positive pairs.

        For each plagiarized pair and each transformation, we create a new
        pair where ``text_a`` is the original and ``text_b`` is the
        transformed version. The ground-truth label remains ``plagiarized``.
        """
        if not self.dataset:
            self.load_dataset()

        self.adversarial_pairs = []
        pair_id = 0

        for original_pair in self.dataset.get("pairs", []):
            if original_pair.get("label") != "plagiarized":
                continue

            text_a = original_pair["text_a"]
            text_b = original_pair["text_b"]

            for transform_name, transform_fn in TRANSFORMATIONS.items():
                try:
                    if transform_name in ("synonym_substitution", "local_paraphrasing"):
                        transformed = transform_fn(
                            text_b, ratio=self.max_ratio, rng=self.rng
                        )
                    elif transform_name == "compression":
                        transformed = transform_fn(text_b, factor=0.5)
                    elif transform_name == "expansion":
                        transformed = transform_fn(text_b, factor=1.5)
                    elif transform_name == "lexical_noise":
                        transformed = transform_fn(text_b, noise_rate=0.05, rng=self.rng)
                    else:
                        transformed = transform_fn(text_b, rng=self.rng)

                    pair_id += 1
                    self.adversarial_pairs.append({
                        "id": f"ADV-{pair_id:03d}",
                        "source_id": original_pair["id"],
                        "category": f"adversarial_{transform_name}",
                        "label": "plagiarized",
                        "transformation": transform_name,
                        "text_a": text_a,
                        "text_b": transformed,
                        "original_text_b": text_b,
                        "notes": f"Adversarial variant via {transform_name} applied to {original_pair['id']}",
                    })
                except Exception as e:
                    logger.warning(
                        "Failed to apply %s to %s: %s",
                        transform_name, original_pair["id"], e,
                    )

        return self.adversarial_pairs

    def generate_negative_pairs(self, n_per_transform: int = 2) -> list[dict[str, Any]]:
        """Generate negative (unrelated) pairs for false-positive testing.

        Pairs unrelated texts to verify the detector doesn't produce
        false positives on non-plagiarized adversarial content.
        """
        if not self.dataset:
            self.load_dataset()

        self.negative_pairs = []
        negative_texts = [
            p["text_a"]
            for p in self.dataset.get("pairs", [])
            if p.get("label") == "not_plagiarized"
        ]
        if not negative_texts:
            return self.negative_pairs

        pair_id = 0
        for transform_name, transform_fn in TRANSFORMATIONS.items():
            for i in range(min(n_per_transform, len(negative_texts))):
                text = negative_texts[i % len(negative_texts)]
                try:
                    if transform_name in ("synonym_substitution", "local_paraphrasing"):
                        transformed = transform_fn(
                            text, ratio=self.max_ratio, rng=self.rng
                        )
                    elif transform_name == "compression":
                        transformed = transform_fn(text, factor=0.5)
                    elif transform_name == "expansion":
                        transformed = transform_fn(text, factor=1.5)
                    elif transform_name == "lexical_noise":
                        transformed = transform_fn(text, noise_rate=0.05, rng=self.rng)
                    else:
                        transformed = transform_fn(text, rng=self.rng)

                    # Pick a different text as the pair
                    other_idx = (i + 1) % len(negative_texts)
                    other_text = negative_texts[other_idx]

                    pair_id += 1
                    self.negative_pairs.append({
                        "id": f"NEG-{pair_id:03d}",
                        "source_id": "negative_pool",
                        "category": f"negative_{transform_name}",
                        "label": "not_plagiarized",
                        "transformation": transform_name,
                        "text_a": other_text,
                        "text_b": transformed,
                        "notes": f"Negative pair: unrelated text + {transform_name} variant",
                    })
                except Exception as e:
                    logger.warning(
                        "Failed to generate negative for %s: %s",
                        transform_name, e,
                    )

        return self.negative_pairs

    def generate_full_benchmark(self) -> dict[str, Any]:
        """Generate the complete adversarial benchmark dataset.

        Returns:
            Dictionary with metadata, adversarial pairs, and negative pairs.
        """
        if not self.dataset:
            self.load_dataset()

        self.generate_adversarial_pairs()
        self.generate_negative_pairs()

        all_pairs = self.adversarial_pairs + self.negative_pairs

        benchmark: dict[str, Any] = {
            "name": "Adversarial Paraphrase Detection Benchmark",
            "version": "1.0",
            "description": (
                "Adversarial benchmark measuring plagiarism detection robustness "
                "against 8 transformation categories: synonym substitution, "
                "sentence reordering, clause restructuring, local paraphrasing, "
                "compression/expansion, mixed sections, and lexical noise."
            ),
            "seed": self.seed,
            "max_ratio": self.max_ratio,
            "label_schema": {
                "plagiarized": "The two texts convey the same core ideas; one is derived from the other.",
                "not_plagiarized": "The two texts are independently authored.",
            },
            "transformations": list(TRANSFORMATIONS.keys()),
            "n_adversarial": len(self.adversarial_pairs),
            "n_negative": len(self.negative_pairs),
            "n_total": len(all_pairs),
            "pairs": all_pairs,
        }

        return benchmark

    def save_benchmark(
        self,
        output_path: Path | str | None = None,
    ) -> Path:
        """Generate and save the full benchmark to a JSON file.

        Args:
            output_path: Where to write the JSON. Defaults to
                ``evaluation/results/adversarial_benchmark.json``.

        Returns:
            Path to the saved file.
        """
        benchmark = self.generate_full_benchmark()
        if output_path is None:
            output_path = OUTPUT_DIR / "adversarial_benchmark.json"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(benchmark, f, indent=2, ensure_ascii=False)

        logger.info(
            "Saved adversarial benchmark: %d adversarial + %d negative = %d total pairs → %s",
            benchmark["n_adversarial"],
            benchmark["n_negative"],
            benchmark["n_total"],
            output_path,
        )
        return output_path

    def summary(self) -> str:
        """Return a human-readable summary of the generated benchmark."""
        if not self.adversarial_pairs:
            self.generate_adversarial_pairs()
        if not self.negative_pairs:
            self.generate_negative_pairs()

        lines = [
            "=" * 60,
            "  ADVERSARIAL BENCHMARK GENERATOR — SUMMARY",
            "=" * 60,
            f"  Source dataset  : {self.dataset_path.name}",
            f"  Seed            : {self.seed}",
            f"  Max substitution: {self.max_ratio:.0%}",
            f"  Transformations : {len(TRANSFORMATIONS)}",
            f"  Adversarial pairs: {len(self.adversarial_pairs)}",
            f"  Negative pairs  : {len(self.negative_pairs)}",
            f"  Total pairs     : {len(self.adversarial_pairs) + len(self.negative_pairs)}",
            "",
            "  Per-transformation breakdown:",
        ]
        from collections import Counter
        transform_counts = Counter(
            p["transformation"] for p in self.adversarial_pairs
        )
        for t, count in sorted(transform_counts.items()):
            lines.append(f"    {t:<28} {count:>4} pairs")
        lines.append("=" * 60)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry-point
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Generate and save the adversarial benchmark dataset."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    generator = AdversarialBenchmarkGenerator()
    generator.load_dataset()
    output_path = generator.save_benchmark()

    print(generator.summary())
    print(f"\n  Benchmark saved to: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()
