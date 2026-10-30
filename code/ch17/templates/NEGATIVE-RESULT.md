# Negative result: `<hypothesis>`

A negative result records a registered hypothesis that its decision rule did not support, so that the next researcher does not have to find it again. Section 17.3 explains the fields; Section 16.3.3 and `../results/contribution-summary.txt` contain worked examples.

| Field | Entry |
|---|---|
| Hypothesis | as registered, word for word, with its decision rule |
| Registered | where (file, release) and when, and that it was before the first run |
| Configurations | the configuration tested and its comparator, with the one module that differs |
| Family of worlds | its rules and its seeds; whether it was used before |
| Code | the release or commit, and the command that reproduces the result |

## Outcome

- **Decision.** Not supported: the cases examined and every exception, with the run that produced it.
- **What did hold.** The statements of the hypothesis that held, with their cases.

## Diagnosis

What was found when the exceptions were examined, marked as found after the run. A diagnosis is an explanation, not a result: it says what the runs did, and it becomes a result only when a new hypothesis that it suggests is registered and tested on worlds that did not suggest it.

## What follows

What the negative result rules out, what it does not, and the hypotheses that it suggests, untested.
