from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from cognityx_jobs import JobRepository

from cognityx_dataforge.dataset import deterministic_id

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_identifier(name: str, value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{name} must start with a letter or number and contain only "
            "letters, numbers, '.', '_' or '-'."
        )
    return value


@dataclass(frozen=True, slots=True)
class BuildIdentity:
    experiment_id: str
    variant_id: str
    run_id: str
    job_id: str
    dataset_id: str
    dataset_version: str

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        recipe: str,
        configuration_checksum: str,
        source_checksum: str,
    ) -> "BuildIdentity":
        experiment_id = validate_identifier("experiment_id", experiment_id)
        variant_id = deterministic_id(recipe, configuration_checksum)
        run_id = f"run-{uuid4().hex}"
        return cls(
            experiment_id=experiment_id,
            variant_id=variant_id,
            run_id=run_id,
            job_id=deterministic_id(run_id, "dataforge-job"),
            dataset_id=deterministic_id(experiment_id, variant_id, source_checksum),
            dataset_version=deterministic_id(
                experiment_id,
                variant_id,
                source_checksum,
                configuration_checksum,
            ),
        )

    def fields(self) -> dict[str, str]:
        return {
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
        }

    @property
    def run_root(self) -> str:
        return (
            f"dataforge/experiments/{self.experiment_id}/variants/{self.variant_id}"
            f"/runs/{self.run_id}"
        )

    @property
    def dataset_root(self) -> str:
        return (
            f"{self.run_root}/datasets/{self.dataset_id}/{self.dataset_version}"
        )


def default_jobs_database() -> Path:
    configured = os.environ.get("COGNITYX_DATAFORGE_JOBS_DB")
    if configured:
        return Path(configured).expanduser()
    state_root = Path(
        os.environ.get(
            "XDG_STATE_HOME",
            Path.home() / ".local" / "state",
        )
    )
    return state_root / "cognityx" / "dataforge" / "jobs.sqlite3"


def load_job_repository(database: str | Path | None = None) -> JobRepository:
    path = Path(database).expanduser() if database is not None else default_jobs_database()
    path.parent.mkdir(parents=True, exist_ok=True)
    return JobRepository(str(path))
