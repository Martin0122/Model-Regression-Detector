# Archived prompt versions

Superseded prompts, kept so a run's `prompt_version` can always be traced back to the text that
produced it. Only the current prompt lives in `prompts/`.

| Version | Pass rate | Why it was replaced |
|---|---|---|
| v1 | 66–74% | Bare instruction with no category precedence or summary guidance. Summaries described content but never tone, language mixing, or absence of detail, which the reference summaries call out. |
| v2 | 80–84% | Fixed the above, but "note how the message is written" overcorrected into editorializing — "poorly written", "very unclear", "impatient". Its billing rule was also too broad and pulled account-access cases into billing. |
| v3 | 82% | Stopped the editorializing, but euphemised tone: sarcasm, insults, profanity and accusations all collapsed into "expresses frustration" — in five separate cases, despite the prompt explicitly forbidding that phrase. Also sent bare fault words like "error" to general. |

The v1 and v2 figures span runs recorded before `temperature` was pinned, when the judge
re-rolled its score on every call; they are noisier than later numbers and not directly
comparable. See `src/runs/archive/README.md`.

The lesson carried into v4: a negative instruction naming the unwanted phrase did not suppress
it. Worked examples showing the desired output did.
