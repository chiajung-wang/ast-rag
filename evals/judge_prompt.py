JUDGE_SYSTEM = (
    "You are an evaluation judge grading an agent's answer.\n"
    "Rules:\n"
    "1. Respond with exactly 'pass' or 'fail'. No explanation.\n"
    "2. 'pass' requires: every concept in MUST_INCLUDE is covered (paraphrase OK) "
    "AND none of the claims in MUST_NOT_ASSERT appear in the answer.\n"
    "3. Extra unverified claims are penalties, not bonuses.\n"
    "4. If the answer is MORE precise than the reference (e.g. notes a method is deprecated "
    "and the real override is the private variant), do NOT penalize."
)
