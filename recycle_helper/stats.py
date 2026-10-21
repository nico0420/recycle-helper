"""Running totals for one browser session."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .catalog import BIN_ORDER, BINS
from .result import Advice


@dataclass
class Record:
    item_name: str
    bin: str | None
    guess: str | None

    @property
    def guessed(self) -> bool:
        return self.guess is not None and self.bin is not None

    @property
    def correct(self) -> bool:
        return self.guessed and self.guess == self.bin


@dataclass
class Stats:
    records: list[Record] = field(default_factory=list)

    def add(self, advice: Advice, guess: str | None) -> Record:
        record = Record(
            item_name=advice.item.name if advice.item else "無法判斷",
            bin=advice.bin,
            guess=guess,
        )
        self.records.append(record)
        return record

    @property
    def total(self) -> int:
        return len(self.records)

    def by_bin(self) -> dict[str, int]:
        counts = Counter(r.bin for r in self.records if r.bin)
        return {BINS[b]["name"]: counts.get(b, 0) for b in BIN_ORDER}

    @property
    def guess_count(self) -> int:
        return sum(1 for r in self.records if r.guessed)

    @property
    def accuracy(self) -> float | None:
        guessed = [r for r in self.records if r.guessed]
        if not guessed:
            return None
        return sum(r.correct for r in guessed) / len(guessed)

    def most_missed(self, limit: int = 3) -> list[tuple[str, int]]:
        misses = Counter(r.item_name for r in self.records if r.guessed and not r.correct)
        return misses.most_common(limit)
