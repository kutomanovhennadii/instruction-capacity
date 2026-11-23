# sections — the text of every quantity in the passport

One renderer per quantity: `sync_text` for P_sync, `rule_word_text` for H, `background_text` for F. A renderer takes the section of
its quantity and the names of its figures by the plan and returns the lines of its part of `passport.md`: the fit, the spread, the
laws, the level and the links to the figures. A quantity without laws gets a sentence instead of the laws. Every renderer keeps
the title of its quantity in `TITLE`.
`capacities_text` writes the part that opens the document: the zero point and the level 0.5 of the capacities, then for every
quantity under its title the table of its capacities and what they are, or a sentence for a quantity without laws.
`stress_test_text` writes the standard stress test, which follows the capacities: the seven numbers of the test at its point,
without figures, or a sentence with the status of a test without a point.

## Key signatures

    capacities_lines(passport) -> list[str]
    stress_test_lines(section) -> list[str]
    sync_lines(section, figures) -> list[str]
    rule_word_lines(section, figures) -> list[str]
    background_lines(section, figures) -> list[str]

## Boundaries

The renderers compute nothing; the cells shared by the quantities live in `../text_cells.py`.
