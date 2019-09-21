from dataclasses import dataclass


@dataclass(frozen=True)
class Annotation:
    annotator: str
    task: str
    value: str
