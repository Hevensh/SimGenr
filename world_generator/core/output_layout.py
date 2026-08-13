from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorldDataLayout:
    root: Path

    @property
    def weather(self) -> Path:
        return self.root / "stage_05_weather"

    @property
    def energy(self) -> Path:
        return self.root / "stage_08_energy_sites"

    @property
    def buses(self) -> Path:
        return self.root / "stage_09_grid_buses"

    @property
    def topology(self) -> Path:
        return self.root / "stage_10_grid_topology"

    @property
    def operation(self) -> Path:
        return self.root / "stage_11_operation"

    @property
    def grid_update(self) -> Path:
        return self.root / "stage_12_grid_update"

    @property
    def storage_planning(self) -> Path:
        return self.root / "stage_13_storage_planning"

    @property
    def storage_dispatch(self) -> Path:
        return self.root / "stage_14_storage_dispatch"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def config_snapshot(self) -> Path:
        return self.root / "config_snapshot.yaml"

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in self.stage_directories():
            directory.mkdir(parents=True, exist_ok=True)

    def stage_directories(self) -> tuple[Path, ...]:
        return (
            self.weather,
            self.energy,
            self.buses,
            self.topology,
            self.operation,
            self.grid_update,
            self.storage_planning,
            self.storage_dispatch,
        )

    def existing(self, directory: Path, filename: str, *legacy_paths: Path) -> Path:
        preferred = directory / filename
        if preferred.exists():
            return preferred
        for path in legacy_paths:
            if path.exists():
                return path
        return preferred
