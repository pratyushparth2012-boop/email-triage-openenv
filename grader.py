def grade_easy(pred, actual):
    return 1.0 if pred == actual else 0.0

def grade_medium(pred_class, actual_class, reply):
    score = 0
    if pred_class == actual_class:
        score += 0.5
    if len(reply) > 5:
        score += 0.5
    return score

def grade_hard(reply, expected):
    reply = reply.lower()
    expected_words = expected.lower().split()

    matches = sum(1 for w in expected_words if w in reply)
    return matches / len(expected_words)
