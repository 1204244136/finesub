"""What a scoring run is measuring, and what it is therefore allowed to conclude.

Two of this branch's wrong conclusions, and both of the ruler defects the 2026-09-20 audit found,
have the same shape: **a property of the measurement that nobody wrote down, so every script
assumed a different answer.**

    Q12 "cross-episode matching works"        87% of the "same person" pairs were the same game
                                              clip played twice -- hina's truth names ROLES
    "merging by similarity is closed off"     same contaminated labels, opposite direction
    edit cost "abstention is free"            nobody had said which rows are scored
    crossfit "the fitting bonus is ~0"        nobody had said what the fold actually holds out

Four questions, declared here once and consumed rather than re-answered:

1. **Whose truth is it** -- people, or roles? (`TruthKind`)
2. **Is it per-line?** A spot-check cannot support a per-line conclusion.
3. **Which rows get scored**, and what does a row nobody labelled cost? (`ScoringPolicy`)
4. **What does a crossfit fold hold out?** (`ScoringPolicy.fold_seconds`)

⚠ This module deliberately does NOT try to unify how the 30-odd exploration scripts load their
truth. That would be a rewrite, and `CLAUDE.md` says tools are maintained on demand. It declares
the properties, and the scripts that still produce numbers tracked documents quote consume it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TruthKind(str, Enum):
    """Whether a reference label names a human being or something else."""

    #: One label, one human. Only this supports a conclusion about identity.
    PEOPLE = "people"
    #: A label is a role, a character or a slot -- `嘉宾`, `男主`, `派蒙`. Two episodes' `嘉宾`
    #: are different humans; one `派蒙` is the same recording played twice.
    ROLES = "roles"
    #: Nobody has established which, so neither may be assumed.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Corpus:
    """One body of material and what its reference can support."""

    name: str
    truth: TruthKind
    #: Why it is classified that way. ⚠ Required: a classification with no evidence behind it is
    #: the thing this module exists to prevent, not a smaller version of it.
    why: str
    #: Is there a reference label on every line, or only a spot check?
    line_level: bool


CORPORA: dict[str, Corpus] = {
    "onetonton": Corpus(
        "onetonton", TruthKind.PEOPLE,
        "姐姐咚咚: fifteen episodes, the same three NAMED humans throughout, line-level truth "
        "(docs/diarization-findings.md 8 and 11.3)", line_level=True),
    "hina": Corpus(
        "hina", TruthKind.ROLES,
        "the cast list holds role placeholders (`嘉宾`, `男主`) shared by different voices, and "
        "game characters (`派蒙`, `钟离`) that are one recording replayed -- so a 'same person' "
        "population built from it is contaminated in both directions "
        "(docs/diarization-findings.md 11.3)", line_level=True),
    "mocai": Corpus(
        "mocai", TruthKind.ROLES,
        "魔裁: labels are the drama's 14-17 CHARACTERS, not the people voicing them",
        line_level=True),
    "nepolabo": Corpus(
        "nepolabo", TruthKind.UNKNOWN,
        "four real people, but the reference is a 70% spot check rather than line-level truth "
        "(docs/diarization-findings.md 11.3) -- not enough to settle an identity question",
        line_level=False),
    "alimeeting": Corpus(
        "alimeeting", TruthKind.PEOPLE,
        "public diarization corpus: labels are speaker ids", line_level=True),
    "ami": Corpus(
        "ami", TruthKind.PEOPLE,
        "public diarization corpus: labels are speaker ids", line_level=True),
    "callhome": Corpus(
        "callhome", TruthKind.PEOPLE,
        "public diarization corpus: labels are speaker ids", line_level=True),
    "voxconverse": Corpus(
        "voxconverse", TruthKind.PEOPLE,
        "public diarization corpus: labels are speaker ids", line_level=True),
}


def corpus(name: str) -> Corpus:
    """The declared properties of `name`, or a failure that says to declare them.

    ⚠ An undeclared corpus raises rather than defaulting to `UNKNOWN`. A silent default is how
    the property goes back to being invisible, which is the whole problem.
    """
    try:
        return CORPORA[name]
    except KeyError:
        raise KeyError(
            f"corpus {name!r} has no declared truth kind. Add it to eval_contract.CORPORA with "
            f"the evidence for the classification; known: {', '.join(sorted(CORPORA))}"
        ) from None


def announce(name: str) -> Corpus:
    """Print what this corpus's reference can support, and return it.

    Every script that reports an aggregate over a corpus should call this, so the reading and the
    caveat leave the machine together. A number in a terminal outlives the context it was read in.
    """
    found = corpus(name)
    detail = "line-level" if found.line_level else "⚠ SPOT-CHECKED, not line-level"
    print(f"corpus {found.name}: truth names {found.truth.value.upper()} ({detail})")
    print(f"  why: {found.why}")
    return found


def require_people_truth(name: str, question: str, *, roles_ok: str | None = None) -> Corpus:
    """Refuse to answer an identity question on truth that does not name people.

    "Are these two tracks the same person" is only answerable where a label IS a person. Pass
    `roles_ok` with the reason when the run is deliberately measuring what the answer looks like
    on role labels -- that is a legitimate thing to do, and it was how the contamination got
    characterised -- but it has to be said out loud, in the script, next to the question.
    """
    found = corpus(name)
    if found.truth is TruthKind.PEOPLE:
        return found
    if roles_ok:
        print(f"⚠ {name} truth names {found.truth.value}, answering {question!r} anyway: {roles_ok}")
        print(f"  ⚠ this reading is NOT about identity. why: {found.why}")
        return found
    raise ValueError(
        f"{question!r} is an identity question and {name}'s truth names {found.truth.value}, "
        f"so the answer would not be about identity. {found.why}. Use a corpus whose truth names "
        f"people, or pass roles_ok='<why this reading is still worth taking>'."
    )


@dataclass(frozen=True)
class ScoringPolicy:
    """The answers the edit-cost ruler needs before it can price anything.

    ⚠ `price_unlabelled` is the one that was never asked. `refined_multi.strong_cost` skipped a
    row the arm left unlabelled, which is sound for the theta arms it was written for -- one fixed
    segmentation, the same rows unlabelled in every arm -- and wrong for comparing two backends
    whose coverage differs, because then deleting every label scores zero (2026-09-20 audit).
    """

    #: ⚠⚠ **`False` -- owner 2026-09-20: "对于这种无关痛痒的，记 0 分".**
    #:
    #: The rows in question are ones the reference covers and the timeline left with no speaker
    #: at all. Measured on why that happens: 72% of them have NO track active in that span, 28%
    #: have one covering only ~12% of the line against a 0.20 threshold -- and they are SHORT,
    #: median 0.67 s against 1.75 s for all rows. They are interjections. A proofreader reading
    #: the text does not care that 「はい」 carries no speaker tag, so they cost nothing.
    #:
    #: ⚠ That makes the number **the edit cost over LABELLED rows**, and it has to be called that
    #: wherever it is quoted -- because with abstention free, an arm can still lower it by
    #: labelling less. `edit_cost` therefore reports `abstained` beside the total ALWAYS, and a
    #: comparison that does not show both is not reporting the ruler it used.
    price_unlabelled: bool = False
    #: Owner 2026-09-14/15: delete 1, move 2, a missed second speaker `add`, `merge` per split.
    add: float = 2.0
    merge: float = 3.0
    #: Crossfit interleaves by this many seconds. ⚠ Held out at FRAME level -- assigning whole
    #: segments by their start let a segment crossing the boundary carry held-out truth into the
    #: fit (2026-09-20 audit, finding 4).
    fold_seconds: float = 60.0


#: What every comparison in the closeout used after 2026-09-20. Scripts take their defaults from
#: here rather than repeating the literals, so there is one place to change and one to read.
DEFAULT_POLICY = ScoringPolicy()
