"""Sorting decision + actuator interface.

The decision is derived only from EcoGuard: an ACCEPTed item gets its bin action;
a REVIEW item gets NO bin and is held for the supervisor (never auto-sorted).

Backends
  simulation  default; records what WOULD happen. No hardware is touched.
  gpio        pulses one GPIO output per bin (gpiozero, BCM pin numbers from config/hardware.yaml).
              Requires BOTH `actuator.backend: gpio` in config/hardware.yaml AND the station's
              --enable-actuator flag. Not validated on hardware by the Zyqora authors.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .schemas import ACCEPT, MANUAL_REVIEW

log = logging.getLogger("zyqora.sorter")


class ActuatorError(RuntimeError):
    pass


@dataclass(frozen=True)
class SortDecision:
    action: str                 # RECYCLE / COMPOST / ... or MANUAL_REVIEW
    actuate: bool               # True only for an ACCEPTed bin action
    reason: str


def decide_sort(ecoguard_status: str, disposal_action: str) -> SortDecision:
    if ecoguard_status == ACCEPT and disposal_action != MANUAL_REVIEW:
        return SortDecision(disposal_action, True, "accepted by EcoGuard")
    return SortDecision(MANUAL_REVIEW, False, "held for supervisor review (EcoGuard REVIEW)")


@dataclass
class SimulatedSorter:
    name: str = "simulation"
    hardware: bool = False
    history: list[str] = field(default_factory=list)

    def execute(self, decision: SortDecision) -> str:
        msg = (f"[SIMULATION] would open the {decision.action} bin" if decision.actuate
               else "[SIMULATION] no bin opened; item held for review")
        self.history.append(msg)
        log.info(msg)
        return msg

    def close(self) -> None:
        pass


class GpioSorter:
    """One output pin per bin action, pulsed for `pulse_s` seconds (e.g. a relay or servo driver input)."""

    name = "gpio"
    hardware = True

    def __init__(self, pins: dict[str, int], pulse_s: float = 0.5, active_high: bool = True, device_cls=None):
        if not pins:
            raise ActuatorError("gpio backend needs at least one action -> pin entry")
        if not 0 < pulse_s <= 5:
            raise ActuatorError("pulse_s must be in (0, 5] seconds")
        if len(set(pins.values())) != len(pins):
            raise ActuatorError(f"duplicate GPIO pins in {pins}")
        if device_cls is None:
            try:
                from gpiozero import OutputDevice as device_cls  # type: ignore
            except ImportError as exc:
                raise ActuatorError("gpiozero not available (apt install python3-gpiozero)") from exc
        try:
            self.devices = {a: device_cls(int(p), active_high=active_high, initial_value=False) for a, p in pins.items()}
        except Exception as exc:
            raise ActuatorError(f"could not claim GPIO pins {pins}: {exc}") from exc
        self.pulse_s = pulse_s

    def execute(self, decision: SortDecision) -> str:
        if not decision.actuate:
            return "no bin opened; item held for review"
        dev = self.devices.get(decision.action)
        if dev is None:
            raise ActuatorError(f"no GPIO pin configured for action {decision.action}")
        dev.on()
        try:
            time.sleep(self.pulse_s)
        finally:
            dev.off()
        log.info("GPIO pulse for %s (%.2fs)", decision.action, self.pulse_s)
        return f"opened {decision.action} bin (GPIO pulse {self.pulse_s}s)"

    def close(self) -> None:
        for dev in self.devices.values():
            try:
                dev.off()
                dev.close()
            except Exception as exc:
                log.warning("GPIO close failed: %s", exc)


def make_sorter(cfg: dict, enable_actuator: bool, device_cls=None):
    """cfg = config/hardware.yaml 'actuator' section. Hardware only with config AND explicit flag."""
    backend = (cfg or {}).get("backend", "simulation")
    if backend == "simulation" or not enable_actuator:
        if backend == "gpio" and not enable_actuator:
            log.warning("actuator backend 'gpio' configured but --enable-actuator not given: using simulation")
        return SimulatedSorter()
    if backend == "gpio":
        return GpioSorter({str(k): int(v) for k, v in (cfg.get("pins") or {}).items()},
                          float(cfg.get("pulse_s", 0.5)), bool(cfg.get("active_high", True)), device_cls)
    raise ActuatorError(f"unknown actuator backend {backend!r}")

