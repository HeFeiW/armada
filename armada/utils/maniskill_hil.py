import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence


@dataclass
class InterventionDecision:
    action: str
    reason: str
    timestamp: float


class ManiSkillHumanInLoopController:
    """Decision controller for simulation-only human-in-loop flow.

    Supported decisions:
    - continue: keep policy control
    - teleop: hand over to simulated teleop loop
    - discard: drop current env trajectory
    - finish: finalize current env trajectory
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.mode = str(cfg.get("mode", "auto")).lower()
        self.on_failure = str(cfg.get("on_failure", "teleop")).lower()
        self.on_timeout = str(cfg.get("on_timeout", "teleop")).lower()
        self.prompt_timeout_s = float(cfg.get("prompt_timeout_s", 20.0))
        self.max_teleop_steps = int(cfg.get("max_teleop_steps", 64))
        self.max_blocked_rounds = int(cfg.get("max_blocked_rounds", 32))

        self._blocked_rounds: Dict[int, int] = {}

    def decide(
        self,
        env_idx: int,
        reason: str,
        timestep: int,
        key_provider: Optional[Callable[[Sequence[str], int], Optional[str]]] = None,
    ) -> InterventionDecision:
        reason = reason.lower()
        default_action = self.on_timeout if reason == "timeout" else self.on_failure

        print(
            f"[HIL] decide env={env_idx} step={timestep} reason={reason} "
            f"mode={self.mode} default={default_action}",
            flush=True,
        )

        if self.mode != "manual":
            print(f"[HIL] auto decision env={env_idx} -> {default_action}", flush=True)
            return InterventionDecision(action=default_action, reason=reason, timestamp=time.time())

        prompt = (
            f"[HIL][env={env_idx}][t={timestep}] reason={reason}. "
            "Choose [C]ontinue/[T]eleop/[D]iscard/[F]inish: "
        )
        print(prompt, flush=True)

        action = self._read_user_choice_with_timeout(default_action, key_provider=key_provider)
        print(f"[HIL] manual decision env={env_idx} -> {action}", flush=True)
        return InterventionDecision(action=action, reason=reason, timestamp=time.time())

    def _read_user_choice_with_timeout(
        self,
        default_action: str,
        key_provider: Optional[Callable[[Sequence[str], int], Optional[str]]] = None,
    ) -> str:
        mapping = {
            "c": "continue",
            "t": "teleop",
            "d": "discard",
            "f": "finish",
        }

        if key_provider is not None:
            deadline = time.time() + self.prompt_timeout_s
            while time.time() < deadline:
                key = key_provider(tuple(mapping.keys()), 1)
                if key is None:
                    continue
                key = key.lower()
                if key in mapping:
                    print(f"[HIL] dashboard key input -> {mapping[key]}")
                    return mapping[key]
                if key in mapping.values():
                    print(f"[HIL] dashboard key input -> {key}")
                    return key

            print(f"[HIL] dashboard input timeout after {self.prompt_timeout_s:.1f}s")

        if sys.stdin.isatty():
            try:
                import select

                if select.select([sys.stdin], [], [], self.prompt_timeout_s)[0]:
                    text = sys.stdin.readline().strip().lower()
                    if text in mapping:
                        print(f"[HIL] stdin input -> {mapping[text]}")
                        return mapping[text]
                    if text in mapping.values():
                        print(f"[HIL] stdin input -> {text}")
                        return text
            except Exception:
                pass

        print(f"[HIL] No valid input in {self.prompt_timeout_s:.1f}s, fallback to {default_action}.")
        return default_action

    def mark_blocked_round(self, env_idx: int) -> bool:
        rounds = self._blocked_rounds.get(env_idx, 0) + 1
        self._blocked_rounds[env_idx] = rounds
        print(f"[HIL] blocked round env={env_idx} count={rounds}/{self.max_blocked_rounds}")
        return rounds > self.max_blocked_rounds

    def clear_blocked(self, env_idx: int):
        if env_idx in self._blocked_rounds:
            print(f"[HIL] clear blocked state env={env_idx}")
        self._blocked_rounds.pop(env_idx, None)
