# anchors — the anchors of an answer by the background words of its lines

An anchor is a pair (line of the source text, line of the answer) that the validator takes for one and the same line; the
words between two neighbouring anchors are decided by a strategy of `gaps/`.

## What is here

| Module | Role |
|---|---|
| `anchor_finder.py` | `AnchorFinder`: the interface of a strategy that finds the anchors |
| `marker_anchor_finder.py` | `MarkerAnchorFinder` and `DEFAULT_ANCHOR_FINDER`: the recognizers in their order, then the longest chain |
| `line_recognizers.py` | `LineRecognizer` and its three strategies: `EqualLineRecognizer`, `SkeletonRecognizer`, `MarkerRecognizer` |
| `text_line_index.py` | `TextLineIndex`, `build_text_line_index`, `AnswerLines`, `encode_answer_lines`: the lines as numbers and their hashes |
| `longest_chain.py` | `longest_chain`: the longest chain of candidate pairs increasing in both lines, by the patience method |

## How a line is recognized

Lines are compared in the canonical form. A marker of an answer line is a background word of the text standing in it, with its
place; the skeleton of a line has the left-hand words of the rules masked. An answer line can stand on the text lines equal to
it; otherwise on the text lines with the same skeleton where more than half of the places agree; otherwise on the text lines of
the same length that keep every marker of the line and more than half of its places. Of all such pairs the anchors are the
longest chain going forward in both the text and the answer; of equal chains, the one with the earlier answer lines.

## Key signatures

    AnchorFinder.find(reference_signatures, answer_signatures, sources) -> list[tuple[int, int]]
    MarkerAnchorFinder(recognizers: Sequence[LineRecognizer])
    LineRecognizer.recognize(index, answer, open_lines) -> Candidates
    build_text_line_index(reference_signatures, sources) -> TextLineIndex
    encode_answer_lines(index, answer_signatures) -> AnswerLines
    longest_chain(answer_lines, text_lines) -> list[tuple[int, int]]

## Boundaries

The folder finds the anchors of one answer: it decides no gaps and builds no places. The checks of candidates and the chain are
compiled loops (`iec.infrastructure.acceleration`): without numba they run as plain Python, slower, with the same anchors.
