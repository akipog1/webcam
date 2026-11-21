"""X01 game logic (301/501/701), double-out, busts, averages, multi-player."""

from __future__ import annotations

from dataclasses import dataclass, field

from .scorer import Score


@dataclass
class Player:
    name: str
    remaining: int
    darts_thrown: int = 0
    points_scored: int = 0

    @property
    def three_dart_average(self) -> float:
        return 3.0 * self.points_scored / self.darts_thrown if self.darts_thrown else 0.0


@dataclass
class TurnResult:
    status: str  # ok | bust | win | turn_full
    message: str


@dataclass
class GameState:
    """Tracks scores. Busted turns revert to the turn-start score."""

    start: int = 501
    double_out: bool = True
    players: list[Player] = field(default_factory=list)
    current: int = 0
    turn_darts: list[Score] = field(default_factory=list)
    turn_start_remaining: int = 0
    busted: bool = False
    winner: str | None = None

    @classmethod
    def new(cls, names: list[str], start: int = 501, double_out: bool = True) -> "GameState":
        g = cls(start=start, double_out=double_out, players=[Player(n, start) for n in names])
        g.turn_start_remaining = start
        return g

    @property
    def player(self) -> Player:
        return self.players[self.current]

    @property
    def turn_total(self) -> int:
        return 0 if self.busted else sum(s.points for s in self.turn_darts)

    def _recompute(self) -> TurnResult:
        p = self.player
        rem = self.turn_start_remaining
        self.busted = False
        self.winner = None
        result = TurnResult("ok", "")
        for s in self.turn_darts:
            new = rem - s.points
            if (
                new < 0
                or (self.double_out and new == 1)
                or (new == 0 and self.double_out and not s.is_double)
            ):
                self.busted = True
                rem = self.turn_start_remaining
                result = TurnResult("bust", f"BUST - {p.name} stays on {rem}")
                break
            rem = new
            if rem == 0:
                self.winner = p.name
                result = TurnResult("win", f"{p.name} wins!")
                break
        p.remaining = rem
        return result

    def add_dart(self, score: Score) -> TurnResult:
        if self.winner:
            return TurnResult("win", f"{self.winner} already won")
        if len(self.turn_darts) >= 3 or self.busted:
            return TurnResult("turn_full", "Turn over - press n")
        self.turn_darts.append(score)
        return self._recompute()

    def undo(self) -> bool:
        if not self.turn_darts:
            return False
        self.turn_darts.pop()
        self._recompute()
        return True

    def override_last(self, score: Score) -> TurnResult:
        if not self.turn_darts:
            return TurnResult("ok", "nothing to override")
        self.turn_darts[-1] = score
        return self._recompute()

    @property
    def turn_over(self) -> bool:
        return self.busted or len(self.turn_darts) >= 3 or self.winner is not None

    def next_turn(self) -> None:
        """Commit stats and advance to the next player."""
        p = self.player
        p.darts_thrown += len(self.turn_darts)
        p.points_scored += self.turn_total
        self.turn_darts = []
        self.busted = False
        if self.winner is None:
            self.current = (self.current + 1) % len(self.players)
        self.turn_start_remaining = self.player.remaining
