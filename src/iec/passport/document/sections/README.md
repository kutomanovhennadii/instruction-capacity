# sections — the text of every quantity in the passport

One renderer per quantity: `sync_text` for P_sync, `rule_word_text` for H, `background_text` for F. A renderer takes the section of
its quantity and the names of its figures by the plan and returns the lines of its part of `passport.md`: the fit, the spread, the
laws, the level, the capacities and the links to the figures. A quantity without laws gets a sentence instead of the laws.

## Key signatures

    sync_lines(section, figures) -> list[str]
    rule_word_lines(section, figures) -> list[str]
    background_lines(section, figures) -> list[str]

## Boundaries

The renderers compute nothing; the cells shared by the quantities live in `../text_cells.py`.
