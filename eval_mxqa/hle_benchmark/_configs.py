from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Config:
    dataset: str
    provider: str
    base_url: str
    model: str
    max_completion_tokens: int
    reasoning: bool
    num_workers: int
    max_samples: Optional[int] = None
    question_range: Optional[List[int]] = None
    judge: str = ""