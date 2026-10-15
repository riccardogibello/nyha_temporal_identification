from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

from nyha_temporal_identification.data_model.dspy_model import NYHAMention
from nyha_temporal_identification.evaluation.constants import (
    CLASS_ERROR_TYPES,
    ERROR_TYPES,
    TEMP_ERROR_TYPES,
)


@dataclass
class ErrorRecord:
    old_note_id: str
    gold: List[Optional[NYHAMention]]
    pred: List[Optional[NYHAMention]]
    error_types: List[str] = field(default_factory=list)
    details: str = ""


@dataclass
class ErrorSummary:
    """
    Structured error summary split by task from the start.

    Attributes:
        class_error_counts:     {error_type: count} for class-relevant errors.
        temp_error_counts:      {error_type: count} for temporality-relevant errors.
        class_per_label_errors: {"NYHA_{class}": {error_type: count}}.
        temp_per_label_errors:  {"current"|"historical": {error_type: count}}.
        total_errors:           Total number of errors across all types.
    """

    class_error_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    temp_error_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    class_per_label_errors: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    temp_per_label_errors: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    total_errors: int = 0

    def record(self, etype: str, gold_m: NYHAMention) -> None:
        """Register one error, routing it into the class and/or temp partition."""
        self.total_errors += 1
        nyha_class = gold_m.get_nyha_class() or "unknown"
        temporality = gold_m.temporality or "unknown"

        if etype in CLASS_ERROR_TYPES:
            self.class_error_counts[etype] += 1
            self.class_per_label_errors[nyha_class][etype] += 1

        if etype in TEMP_ERROR_TYPES:
            self.temp_error_counts[etype] += 1
            self.temp_per_label_errors[temporality][etype] += 1

    def _build_partition(
        self,
        counts: dict[str, int],
        per_label: dict[str, dict[str, int]],
    ) -> dict:
        """Shared formatting logic for a single task partition."""
        sorted_counts = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
        return {
            "total": sum(sorted_counts.values()),
            "counts": sorted_counts,
            "percentages": {
                etype: round(100.0 * count / self.total_errors, 2)
                if self.total_errors > 0
                else 0.0
                for etype, count in sorted_counts.items()
            },
            "descriptions": {
                etype: ERROR_TYPES.get(etype, "unknown") for etype in sorted_counts
            },
            "per_label": {
                label: dict(errs) for label, errs in sorted(per_label.items()) if errs
            },
        }

    def as_class_partition(self) -> dict:
        return self._build_partition(
            dict(self.class_error_counts),
            dict(self.class_per_label_errors),
        )

    def as_temp_partition(self) -> dict:
        return self._build_partition(
            dict(self.temp_error_counts),
            dict(self.temp_per_label_errors),
        )

    def to_json(self) -> dict:
        """
        Serialise the full ErrorSummary into a structured dict with both
        task partitions and a combined overview.

        Structure:
            {
                "total_errors": int,
                "class_errors": {          # see as_class_partition()
                    "total": int,
                    "counts": {error_type: count},
                    "percentages": {error_type: float},
                    "descriptions": {error_type: str},
                    "per_label": {"NYHA_{class}": {error_type: count}},
                },
                "temp_errors": {           # see as_temp_partition()
                    "total": int,
                    "counts": {error_type: count},
                    "percentages": {error_type: float},
                    "descriptions": {error_type: str},
                    "per_label": {"current"|"historical": {error_type: count}},
                },
                "overview": {
                    "all_error_counts": {error_type: count},   # union of both partitions
                    "by_type": {                               # per error_type detail
                        error_type: {
                            "description": str,
                            "total": int,
                            "in_class_partition": bool,
                            "in_temp_partition": bool,
                        }
                    },
                },
            }
        """
        class_partition = self.as_class_partition()
        temp_partition = self.as_temp_partition()

        # Union of counts across both partitions; wrong_both appears in both
        # but must be counted once in the overview.
        all_types = set(class_partition["counts"]) | set(temp_partition["counts"])
        all_error_counts = {
            etype: (
                class_partition["counts"].get(etype, 0)
                or temp_partition["counts"].get(etype, 0)
            )
            for etype in sorted(all_types)
        }

        by_type = {
            etype: {
                "description": ERROR_TYPES.get(etype, "unknown"),
                "total": count,
                "in_class_partition": etype in CLASS_ERROR_TYPES,
                "in_temp_partition": etype in TEMP_ERROR_TYPES,
            }
            for etype, count in sorted(
                all_error_counts.items(), key=lambda x: (-x[1], x[0])
            )
        }

        return {
            "total_errors": self.total_errors,
            "class_errors": class_partition,
            "temp_errors": temp_partition,
            "overview": {
                "all_error_counts": all_error_counts,
                "by_type": by_type,
            },
        }
