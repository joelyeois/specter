from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._job import _resolve_base_dir


class JobDatabase:
    """
    Read-only view over all job folders under a base directory.

    Parameters
    ----------
    base_dir : str or Path, optional
        Root directory to scan. Defaults to ``~/specter-data/`` or
        the ``SPECTER_JOBS_DIR`` environment variable.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = _resolve_base_dir(base_dir)

    def list(self, project: str | None = None) -> list[dict[str, Any]]:
        """
        Return all job records, optionally filtered by project.

        Parameters
        ----------
        project : str, optional
            If given, only return jobs from this project folder.

        Returns
        -------
        list[dict]
            Each element is the parsed contents of a ``job.json`` file,
            sorted by ``created_at`` ascending.
        """
        results = []
        search_root = self._base_dir / project if project else self._base_dir
        if not search_root.exists():
            return []
        for job_json in sorted(search_root.glob("**/job.json")):
            try:
                results.append(json.loads(job_json.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return results

    def get(self, project: str | None, job_id: str) -> dict[str, Any]:
        """
        Return a single job record by project and ID.

        Parameters
        ----------
        project : str, optional
            Project folder name. ``None`` for a job created without one
            (living directly under ``base_dir``, not a named project).
        job_id : str
            Job ID, e.g. ``"J001"``. Unique per project (shared across every
            job_type in it -- see `Job`), so the caller doesn't need to name
            the job_type subfolder it happens to live in.

        Returns
        -------
        dict
            Parsed ``job.json`` contents.

        Raises
        ------
        FileNotFoundError
            If no job with this id exists anywhere under the project.
        """
        project_dir = self._base_dir if project is None else self._base_dir / project
        matches = sorted(project_dir.glob(f"*/{job_id}/job.json"))
        if not matches:
            raise FileNotFoundError(
                f"No job {job_id!r} found under {project_dir} (searched every job-type subfolder)"
            )
        return json.loads(matches[0].read_text())

    def diff(
        self, project: str | None, job_id_a: str, job_id_b: str
    ) -> dict[str, tuple[Any, Any]]:
        """
        Compare the ``params`` dicts of two jobs.

        Parameters
        ----------
        project : str, optional
            Project folder name. ``None`` for jobs created without one.
        job_id_a : str
            First job ID.
        job_id_b : str
            Second job ID.

        Returns
        -------
        dict
            Keys where values differ, mapped to ``(value_in_a, value_in_b)``.
            Keys present in one job but not the other have ``None`` on the
            missing side.
        """
        params_a = self.get(project, job_id_a).get("params", {})
        params_b = self.get(project, job_id_b).get("params", {})
        all_keys = set(params_a) | set(params_b)
        return {
            k: (params_a.get(k), params_b.get(k))
            for k in all_keys
            if params_a.get(k) != params_b.get(k)
        }
