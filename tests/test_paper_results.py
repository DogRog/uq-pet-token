from utils.paper_results import match_entities

GOLD = ["B-Action", "O", "B-Entity", "I-Entity", "O", "B-Condition"]


def test_exact_match_needs_same_words_and_type():
    predicted = ["B-Action", "O", "B-Entity", "O", "O", "B-Entity"]
    gold, spans = match_entities(GOLD, predicted, "exact")
    assert spans == [(0, 1, "Action", True), (2, 3, "Entity", False), (5, 6, "Entity", False)]
    assert gold == [(0, 1, "Action", True), (2, 4, "Entity", False), (5, 6, "Condition", False)]


def test_partial_match_needs_one_shared_word_and_each_gold_matches_once():
    predicted = ["B-Action", "O", "B-Entity", "B-Entity", "O", "B-Entity"]
    gold, spans = match_entities(GOLD, predicted, "partial")
    assert spans == [
        (0, 1, "Action", True),
        (2, 3, "Entity", True),
        (3, 4, "Entity", False),
        (5, 6, "Entity", False),
    ]
    assert gold == [(0, 1, "Action", True), (2, 4, "Entity", True), (5, 6, "Condition", False)]
