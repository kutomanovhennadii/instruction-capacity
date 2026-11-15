"""The lines of an answer against the lines of the source: anchors and wrapper lines.

An anchor is a pair of a reference line and an answer line of equal canonical form: the
right-hand word of every rule is replaced by its left-hand word, so a right replacement and an
unreplaced word look the same, and a line rewritten with any such outcomes equals the line of the
source text. The anchors form the longest common subsequence of lines. A wrapper line is a line at
an edge of the answer without a single task word, such as a line of introduction before the text.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def anchor_lines(
    reference_signatures: Sequence[tuple[str, ...]],
    answer_signatures: Sequence[tuple[str, ...]],
) -> list[tuple[int, int]]:
    """Return the anchors: pairs (reference line, answer line) of equal canonical form.

    The pairs form the longest common subsequence of lines. Of equal ones the one where earlier
    answer lines stand against the reference is taken: when the model goes back, the first pass
    stands against the reference and the repeat stays an insertion. The table is built on the
    reversed sequences, so the backtrack walks the answer from its start and takes a match as
    soon as there is one. Without a match an answer line is skipped first, then a reference line:
    so on an intact answer a damaged line that matches a foreign reference line by chance does
    not move the anchors off the diagonal. A row of the table is computed at once with numpy:
    ``d[i][j] = max(d[i-1][j], d[i-1][j-1] + e, d[i][j-1])``, where the last term is the running
    maximum along the row.

    The equal start of the two sequences is taken without the table: the backtrack starts there
    and takes every such pair at once, and the cells it reads next depend only on the rest. The
    equal end cannot be taken so: a repeat at the end must stay an insertion. An answer line that
    equals no reference line is left out of the table: the backtrack always skips it. A reference
    line cannot be left out so: the backtrack may skip an answer line against it.
    """
    shared = 0
    for reference_line, answer_line in zip(reference_signatures, answer_signatures, strict=False):
        if reference_line != answer_line:
            break
        shared += 1
    pairs = [(index, index) for index in range(shared)]
    if shared == len(reference_signatures) or shared == len(answer_signatures):
        return pairs
    ids: dict[tuple[str, ...], int] = {}
    reference_ids = [
        ids.setdefault(line, len(ids)) for line in reversed(reference_signatures[shared:])
    ]
    kept = [
        (index, ids[line])
        for index, line in zip(
            range(len(answer_signatures) - 1, shared - 1, -1),
            reversed(answer_signatures[shared:]),
            strict=True,
        )
        if line in ids
    ]
    if not kept:
        return pairs
    answer_ids = [number for _, number in kept]
    answer_row = np.array(answer_ids, dtype=np.int32)
    present = frozenset(answer_ids)
    rows, columns = len(reference_ids), len(answer_ids)
    table = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    for i in range(1, rows + 1):
        if reference_ids[i - 1] not in present:
            table[i] = table[i - 1]
            continue
        equal = (answer_row == reference_ids[i - 1]).astype(np.int32)
        candidate = np.maximum(table[i - 1, 1:], table[i - 1, :-1] + equal)
        table[i, 1:] = np.maximum.accumulate(candidate)
    i, j = rows, columns
    while i > 0 and j > 0:
        if reference_ids[i - 1] == answer_ids[j - 1]:
            pairs.append((shared + rows - i, kept[j - 1][0]))
            i -= 1
            j -= 1
        elif table.item(i, j - 1) == table.item(i, j):
            j -= 1
        else:
            i -= 1
    return pairs


def wrapper_lines(
    answer_lines: Sequence[tuple[str, ...]],
    anchors: Sequence[tuple[int, int]],
    task_words: frozenset[str],
) -> frozenset[int]:
    """Return the wrapper lines: lines at the edges of the answer without a single task word.

    An edge is the start of the answer before the first anchor and the end after the last one;
    wrapper lines run in a row from the very edge. An empty line has no task words. A line with
    at least one task word is never a wrapper: it is rewritten text with errors, and it goes into
    the alignment.
    """
    first = anchors[0][1] if anchors else len(answer_lines)
    last = anchors[-1][1] if anchors else -1
    wrapper: set[int] = set()
    index = 0
    while index < first and not task_words.intersection(answer_lines[index]):
        wrapper.add(index)
        index += 1
    index = len(answer_lines) - 1
    floor = max(last, max(wrapper, default=-1))
    while index > floor and not task_words.intersection(answer_lines[index]):
        wrapper.add(index)
        index -= 1
    return frozenset(wrapper)
