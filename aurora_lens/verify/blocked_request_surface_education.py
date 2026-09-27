"""Regex-free surfaces: academic integrity violation (ghostwriting)."""

from __future__ import annotations


def surface_academic_integrity_violation(norm: str, tokens: tuple[str, ...]) -> bool:
    """Token / phrase cues for ghostwriting or exam-answer provision for submission."""
    w = f" {norm} "
    tok_f = frozenset(tokens)

    # Core action verbs
    verbs = {"write", "complete", "finish", "do", "take", "sit", "answer"}
    if tok_f.isdisjoint(verbs):
        return False

    # Academic subjects / objects
    subjects = {
        "essay",
        "assignment",
        "thesis",
        "dissertation",
        "homework",
        "coursework",
        "paper",
        "report",
        "exam",
        "test",
        "quiz",
        "assessment",
        "midterm",
        "final",
    }
    if tok_f.isdisjoint(subjects):
        return False

    # Submission / intent cues
    intent = {"submit", "submission", "behalf", "instead"}
    has_intent = not tok_f.isdisjoint(intent) or " for me " in w or " hand in " in w

    if not has_intent:
        return False

    # "write my essay", "complete the assignment", etc.
    # We look for verb ... (my|the) ... subject
    n = len(tokens)
    for i in range(n):
        if tokens[i] in verbs:
            # Look ahead for subject within a reasonable window
            limit = min(i + 15, n)
            window = tokens[i:limit]
            window_f = frozenset(window)
            
            if not window_f.isdisjoint(subjects):
                # Check for "my" or "the" or "for me" etc.
                if any(x in window_f for x in {"my", "the", "me"}):
                    return True
                if " for me " in w or " on my behalf " in w:
                    return True

    return False


def surface_student_record_exposure(norm: str, tokens: frozenset[str], user_input: str) -> bool:
    """Token / phrase cues for student record exposure requests."""
    w = f" {norm} "
    
    # Data fields: grade, gpa, transcript, standing, etc.
    data_fields = {
        " grade ", " grades ", " gpa ", " transcript ", " disciplinary ", " student id ",
        " enrollment ", " academic record ", " ferpa ", " financial aid ",
        " disability ", " iep ", " 504 plan ", " academic standing "
    }
    
    has_data_field = any(x in w for x in data_fields)
    if not has_data_field:
        return False

    # Person signal: name (Capitalized) or pronoun or "student"
    if any(x in w for x in (" his ", " her ", " their ", " student ", " pupil ", " learner ")):
        return True
        
    # Simple check for capitalized name (e.g. "Tyler Johnson") in original input
    words = user_input.split()
    sentence_starters = {"I", "The", "This", "That", "We", "They", "It", "What", "How", "When", "Where", "Who", "Which"}
    for i in range(len(words) - 1):
        if len(words[i]) > 1 and words[i][0].isupper() and len(words[i+1]) > 1 and words[i+1][0].isupper():
            # Exclude common sentence starters
            if words[i] not in sentence_starters:
                return True
    
    return False


def surface_personalized_academic_advice(norm: str, tokens: frozenset[str]) -> bool:
    """Token / phrase cues for personalized academic advice requests."""
    w = f" {norm} "
    
    # Personal binding: my application, my chances, my gpa, etc.
    personal = {
        "my application", "my chances", "my gpa", "my grades", "my results",
        "will i get in", "will i be accepted", "am i eligible", "should i apply",
        "what are my chances", "what are my odds"
    }
    if not any(x in w for x in personal):
        return False
        
    # Vocab: university, college, admission, scholarship, degree, etc.
    vocab = {
        "university", "college", "admission", "admissions", "scholarship",
        "degree", "qualification", "credential", "program", "course"
    }
    if tokens.isdisjoint(vocab):
        return False
        
    return True
