---
name: user_style
description: How this user prefers to work — language, commit style, explanations
metadata:
  type: user
---

**Language:** Spanish. All prompts, output, and documentation in Spanish.

**Commits:** One commit per feature/fix, with clear message summarizing the change. User says "haz commit" when ready. Prefers single bundled PR over many small ones for refactors in the dataset tools area.

**Code explanations:** Don't add comments unless the WHY is non-obvious. User can read well-named identifiers. No multi-paragraph docstrings. Avoid restating what code does ("// read the file") or referencing the task ("added for option [3]").

**No trailing summaries:** Don't summarize what you just did at the end of responses — user can read the diff. Keep responses terse and focused.

**Testing & verification:** User runs the code in Colab and checks behavior. Doesn't ask for "verification" via tests; they verify live.

**When asking questions:** User appreciates direct, binary choices when decisions are genuinely theirs to make (e.g. "MyDrive or Shared Drive?" for option [3] mount path). No hedging.

**Memory note:** This user is building a production multi-label classifier for a science fair/expo. They balance perfectionism (model improvements, dataset quality) with pragmatism (augment only the deficit, not from scratch).
