"""Assembling the prompt of one task: instruction, replacement table and text."""

from __future__ import annotations

from iec.contracts import Prompt, RuleTable, SourceText

_MODULE_NAME = "prompt_builder"

# The prompt template is a code constant, not a config: the wording of the task is part of
# the measurement protocol, the instrument has one mode of operation. Changing the wording
# is a new protocol version, not a turn of a knob. The draft was finalized after the pilot
# and frozen.
PROMPT_TEMPLATE = """\
Rewrite the text below by applying the substitution rules.

Rules (replace every occurrence of the left word with the right word):
{rules_block}

Strict requirements:
- Replace only the words listed in the rules.
- Do not change, add, or remove any other words.
- Keep the exact word order, spaces, and line breaks.
- Output only the rewritten text, nothing else.

Text:
{text_block}
"""


class PromptBuilder:
    """The prompt builder: three blocks in a fixed order - instruction, table, text.

    The rules are shown line by line ``source -> target`` exactly in table order (the order
    of lines is a measured axis of the experiment). A payload task is not assembled:
    ``payload_position`` is always ``None``, the contract field is reserved for weighing
    tasks. Assembly is deterministic: the same inputs give the same prompt to the byte.
    """

    def build(self, rule_table: RuleTable, source_text: SourceText) -> Prompt:
        """Assemble the prompt from the replacement table and the word stream.

        Args:
            rule_table: The replacement table; the order of rules is kept in the Rules block.
            source_text: The word stream; its text form goes into the Text block verbatim.

        Returns:
            ``Prompt`` with the ready request text and an empty payload position.
        """
        rules_block = "\n".join(f"{rule.source} -> {rule.target}" for rule in rule_table.rules)
        return Prompt(
            text=PROMPT_TEMPLATE.format(rules_block=rules_block, text_block=source_text.text),
            payload_position=None,
        )
