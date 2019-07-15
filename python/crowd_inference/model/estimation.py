from dataclasses import dataclass


@dataclass(frozen=True)
class Estimation:
    task: str
    value: str
