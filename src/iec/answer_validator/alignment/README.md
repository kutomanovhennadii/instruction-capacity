# alignment — the alignment of an answer with the reference by places

A place is a reference word or an answer word without a place in the reference: matched, deleted or inserted. `align_answer`
goes in steps, and every step is a module of its own.

## What is here

| Module or folder | Role |
|---|---|
| `line_alignment.py` | `AnswerAligner`, `DEFAULT_ALIGNER` and `align_answer`: the steps of the alignment; the strategies of anchors and gaps come through the constructor |
| `aligned_answer.py` | the result: the kinds and origins of places, `Place` and `AlignedAnswer` with its arrays |
| `answer_text.py` | `normalize_output` and `split_lines`: the text forms of an answer |
| `wrapper_lines.py` | `wrapper_lines`: the wrapper lines at the edges of an answer |
| `anchors/` | the anchors of lines by their background words: the recognizers and the longest chain behind one interface |
| `place_builder.py` | `PlaceBuilder`: the anchors and the gap decisions turned into the arrays of places |
| `edit_alignment.py` | `edit_alignment`: the words of a pair of lines aligned with the fewest edits |
| `conservation_check.py` | `check_conservation` and `check_positions`: every reference word and every core word in exactly one place |
| `gaps/` | the decision of the gaps between anchors: two strategies behind one interface |

## The steps

1. The lines of the reference, of the source and of the answer; the answer lines get their canonical form, in which the
   right-hand word of every rule is replaced by its left-hand word.
2. The anchors: an answer line is recognized by an equal text line, the same skeleton or the same background words; of all
   recognized pairs, the longest chain going forward in both texts.
3. The wrapper lines: lines without a single task word at the edges of the answer; they are not places.
4. The gaps between anchors, each decided by a strategy of `gaps/`; the anchors and the decisions go into the accumulator of
   places in the order of the lines. In a pair of lines of equal length the words stand by positions, as in the article,
   otherwise by the alignment with the fewest edits.
5. The arrays of places, the check of their cover and the result `AlignedAnswer`.

## The result

The places are kept as numeric arrays: `kinds`, `reference_positions`, `answer_positions` and `groups`, with `core_words` and
`core_line_sizes` next to them. The `places` property builds `Place` objects on request.

**A bad answer is data, not an exception.** An empty answer, a fragment or garbage give places with deletions and insertions.
`ContractViolationError(ALIGNMENT_CONSERVATION)` is raised only if the places do not cover every word of the reference and of
the answer core: that is a fault of the instrument. `check_conservation(places, reference_words, answer_words)` runs the same
check on a list of `Place` objects; the alignment runs `check_positions` on its own arrays.

## Boundaries

The folder aligns one answer: it reads no records and counts no trials.
