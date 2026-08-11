from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from app.ui.keys import Key

@dataclass(frozen=True)
class Go:
    screen: str
    params: Dict[str, Any] = field(default_factory=dict)

QUIT = Go("quit")
MENU = Go("menu")

class Screen:
    needs_camera: bool = False

    def enter(self) -> None:
        pass

    def leave(self) -> None:
        pass

    def update(self, dt: float) -> Optional[Go]:
        return None

    def on_key(self, key: Key) -> Optional[Go]:
        return None

    def draw(self, canvas: np.ndarray) -> None:
        raise NotImplementedError