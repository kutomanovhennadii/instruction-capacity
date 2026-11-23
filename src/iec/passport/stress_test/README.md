# stress_test — the standard stress test of a passport

The package computes the section `stress_test` of a passport from its three published sections: the task of typical
proportions at which the median request errs at the last word in half of the cases, and what its errors consist of
([stress test](../../../../docs/08-stress-test.md)).

| Module | Role |
|---|---|
| `stress_test` | the point of the test, the structure of the errors at it, the influence of the loads, the errors over the text and their 95% intervals |

## The test

The test describes how a model behaves as the complexity of a task with fixed proportions of the load increases step by
step: the density of replacements 0.5 and one rule per 50 words of the text, `T = 50·m` words and `D = 2m` distinct words
for `m` rules, the length of the text and the number of rules increasing together. The complexity increases until the
combined error of the last word of the median request, `E = 1 − P_sync·H`, reaches 50%. The false replacements of
background words `F` do not enter the definition of the point and are measured at the point found. The price of a load at a word is the fall of its law
from the zero of its axis: 20 words and the start of the answer, the density at the shift of its law (the top of its law
for `F`), one rule and the first place of the list; the price of rules at a word is the mean price of the places over the
list. The influence of a load per 100 errors of a kind is the share of its price in the logit of the last word: negative
when the quality grows with the load, the three shares of a kind summing to 100, and not defined (`null`) when the prices
of the kind sum to zero or less.

## Errors

Every number goes with the half-width of its 95% interval by the delta method over `law_covariance` of every quantity,
the three quantities taken independent: their fits are separate.

## Statuses

`measured` - the point lies inside the measured area, `support`: the longest text and the largest number of rules of the
bins of `H`; `extrapolated` - past it, a continuation of the laws; `below_one_rule` - the last word fails in half of the
cases already at one rule; `beyond_range` - it does not up to `2²⁰` rules; `no_laws` - a quantity has no laws.

## Interface

    stress_test_section(passport, assignment) -> dict[str, Any]
