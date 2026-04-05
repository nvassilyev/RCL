"""Core data structures for RCL optimization."""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PlaybookEntry:
    """A single entry in the playbook."""
    content: str
    section: str = "others"
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    helpful_count: int = 0
    harmful_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "section": self.section,
            "entry_id": self.entry_id,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlaybookEntry":
        return cls(
            content=data["content"],
            section=data.get("section", "others"),
            entry_id=data.get("entry_id", str(uuid.uuid4())[:8]),
            helpful_count=data.get("helpful_count", 0),
            harmful_count=data.get("harmful_count", 0),
        )


class Playbook:
    """A collection of playbook entries organized by section."""

    def __init__(self, entries: Optional[List[PlaybookEntry]] = None, allowed_sections: Optional[set] = None):
        self.entries: List[PlaybookEntry] = entries or []
        self.allowed_sections: set = allowed_sections or set()
        self._prompt_suffix: str = ""
        self._content_hashes: set = {
            hashlib.md5(e.content.lower().strip().encode()).hexdigest()
            for e in self.entries
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def copy(self) -> "Playbook":
        pb = self.__class__(
            [
                PlaybookEntry(
                    content=e.content,
                    section=e.section,
                    entry_id=e.entry_id,
                    helpful_count=e.helpful_count,
                    harmful_count=e.harmful_count,
                )
                for e in self.entries
            ],
            allowed_sections=self.allowed_sections,
        )
        pb._prompt_suffix = self._prompt_suffix
        pb._content_hashes = set(self._content_hashes)
        return pb

    def add_entry(self, content: str, section: str = "others", check_duplicate: bool = True) -> Optional[PlaybookEntry]:
        if self.allowed_sections and section not in self.allowed_sections:
            section = "others"

        if check_duplicate:
            content_hash = hashlib.md5(content.lower().strip().encode()).hexdigest()
            if content_hash in self._content_hashes:
                return None
        else:
            content_hash = hashlib.md5(content.lower().strip().encode()).hexdigest()

        entry = PlaybookEntry(content=content, section=section)
        self.entries.append(entry)
        self._content_hashes.add(content_hash)
        return entry

    def remove_entry(self, entry_id: str) -> bool:
        for i, entry in enumerate(self.entries):
            if entry.entry_id == entry_id:
                removed = self.entries.pop(i)
                h = hashlib.md5(removed.content.lower().strip().encode()).hexdigest()
                self._content_hashes.discard(h)
                return True
        return False

    def update_entry(self, entry_id: str, new_content: str) -> bool:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                old_hash = hashlib.md5(entry.content.lower().strip().encode()).hexdigest()
                self._content_hashes.discard(old_hash)
                entry.content = new_content
                new_hash = hashlib.md5(new_content.lower().strip().encode()).hexdigest()
                self._content_hashes.add(new_hash)
                return True
        return False

    def get_entry(self, entry_id: str) -> Optional[PlaybookEntry]:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def get_entries_by_section(self, section: str) -> List[PlaybookEntry]:
        return [e for e in self.entries if e.section == section]

    def to_prompt(self) -> str:
        if not self.entries and not self._prompt_suffix:
            return ""

        sections: Dict[str, List[str]] = {}
        for entry in self.entries:
            if entry.section not in sections:
                sections[entry.section] = []
            sections[entry.section].append(f"- {entry.content}")

        lines = ["## Playbook\n"]
        for section, items in sorted(sections.items()):
            title = section.replace("_", " ").title()
            lines.append(f"### {title}")
            lines.extend(items)
            lines.append("")

        if self._prompt_suffix:
            lines.append(self._prompt_suffix)

        return "\n".join(lines)

    def to_prompt_with_ids(self) -> str:
        if not self.entries:
            return ""

        sections: Dict[str, List[str]] = {}
        for entry in self.entries:
            if entry.section not in sections:
                sections[entry.section] = []
            sections[entry.section].append(f"- [{entry.entry_id}] {entry.content}")

        lines = ["## Playbook\n"]
        for section, items in sorted(sections.items()):
            title = section.replace("_", " ").title()
            lines.append(f"### {title}")
            lines.extend(items)
            lines.append("")

        return "\n".join(lines)

    def to_prompt_with_counts(self, recently_assessed: Optional[set] = None) -> str:
        """Render playbook with entry IDs and helpful/harmful counts.

        Args:
            recently_assessed: Set of entry_ids assessed in the current round.
                These are marked with * to highlight them for the mutator.
        """
        if not self.entries:
            return ""

        recently_assessed = recently_assessed or set()

        sections: Dict[str, List[str]] = {}
        for entry in self.entries:
            if entry.section not in sections:
                sections[entry.section] = []
            marker = " *" if entry.entry_id in recently_assessed else ""
            count_str = f"(+{entry.helpful_count} -{entry.harmful_count})"
            sections[entry.section].append(
                f"- [{entry.entry_id}] {count_str}{marker} {entry.content}"
            )

        lines = ["## Playbook\n"]
        for section, items in sorted(sections.items()):
            title = section.replace("_", " ").title()
            lines.append(f"### {title}")
            lines.extend(items)
            lines.append("")

        return "\n".join(lines)

    def update_counts(
        self, assessments: List[Dict[str, str]],
        iteration: Optional[int] = None,
    ) -> set:
        """Update helpful/harmful counts from reflector assessments.

        Args:
            assessments: List of {"entry_id": str, "tag": "helpful"|"harmful",
                         optional "task_summary": str}
            iteration: Current iteration number (for memory tracking).

        Returns:
            Set of entry_ids that were assessed (for highlighting in mutator prompt).
        """
        assessed_ids = set()
        entry_map = {e.entry_id: e for e in self.entries}
        for a in assessments:
            entry_id = a.get("entry_id", "")
            tag = a.get("tag", "").lower()
            entry = entry_map.get(entry_id)
            if not entry:
                continue
            assessed_ids.add(entry_id)
            if tag == "helpful":
                entry.helpful_count += 1
            elif tag == "harmful":
                entry.harmful_count += 1
        return assessed_ids

    def prune(self, threshold: int = 2) -> List[PlaybookEntry]:
        """Remove entries where harmful exceeds helpful by threshold.

        Args:
            threshold: Remove if harmful_count - helpful_count >= threshold.

        Returns:
            List of pruned entries (for logging).
        """
        pruned = []
        remaining = []
        for entry in self.entries:
            if entry.harmful_count - entry.helpful_count >= threshold:
                pruned.append(entry)
            else:
                remaining.append(entry)
        self.entries = remaining
        self._content_hashes = {
            hashlib.md5(e.content.lower().strip().encode()).hexdigest()
            for e in remaining
        }
        return pruned

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "version": "1.0",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], allowed_sections: Optional[set] = None) -> "Playbook":
        entries = [PlaybookEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(entries, allowed_sections=allowed_sections)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, allowed_sections: Optional[set] = None) -> "Playbook":
        with open(path) as f:
            return cls.from_dict(json.load(f), allowed_sections=allowed_sections)


@dataclass
class ExecutionTrace:
    """Captures the execution of a single task."""
    task_id: str
    input_query: str
    system_output: Any
    trace: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_afc_trace_str(self, max_output_chars: int = 500) -> str:
        """Get the raw AFC trace as a JSON string with tool outputs truncated.

        This is the preferred format for passing to the reflector/mutator —
        models work better with the raw structured trace than a reformatted version.

        Args:
            max_output_chars: Max chars per tool output (0 = no truncation).

        Returns:
            JSON string of the truncated afc_trace, or trace.trace as fallback.
        """
        afc_trace = self.metadata.get("afc_trace")
        if not afc_trace:
            return self.trace

        truncated = []
        for tc in afc_trace:
            tc_copy = dict(tc)
            if max_output_chars > 0 and tc_copy.get("type") == "tool_call":
                output = tc_copy.get("output", "")
                if isinstance(output, str) and len(output) > max_output_chars:
                    tc_copy["output"] = output[:max_output_chars] + "\n[TRUNCATED]"
            truncated.append(tc_copy)

        return json.dumps(truncated, indent=2, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "input_query": self.input_query,
            "system_output": self.system_output,
            "trace": self.trace[:1000] if len(self.trace) > 1000 else self.trace,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a candidate on tasks."""
    score: float
    tgc: float
    per_instance_scores: List[float] = field(default_factory=list)
    traces: List[ExecutionTrace] = field(default_factory=list)
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "tgc": self.tgc,
            "per_instance_scores": self.per_instance_scores,
            "feedback": self.feedback,
            "metadata": self.metadata,
        }
