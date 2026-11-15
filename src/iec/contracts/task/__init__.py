"""Contracts of the task of a point: vocabulary, rule list, source text, prompt, reference."""

from iec.contracts.task.prompt import Prompt
from iec.contracts.task.reference import Reference
from iec.contracts.task.rule_table import Rule, RuleTable
from iec.contracts.task.source_text import SourceText
from iec.contracts.task.vocab import Vocab

__all__ = ["Prompt", "Reference", "Rule", "RuleTable", "SourceText", "Vocab"]
