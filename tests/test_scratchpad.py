from src.cdcl.scratchpad import CDCLScratchpad
from src.model.registry import CommandRegistry


def test_backtrack_logic(cfg, tokenizer):
    registry = CommandRegistry(cfg, tokenizer)
    input_clauses = tokenizer.encode("[ x1 x2 x3 ] [ -x1 x2 -x3 ]")
    scratchpad = CDCLScratchpad(input_clauses, registry)

    def tok(text: str) -> list[int]:
        return tokenizer.encode(text).ids

    scratchpad._saved_tokens_dict["level"] = tok("I I I")
    scratchpad._saved_tokens_dict["decision_levels"] = [
        tok("I"),
        tok("I I"),
        tok("I I I"),
    ]
    scratchpad._saved_tokens_dict["assignments"] = [
        tok("x1"),
        tok("-x2"),
        tok("x3"),
    ]
    scratchpad._saved_tokens_dict["reason_clauses"] = [
        tok("None"),
        tok("None"),
        tok("x1 x2 x3"),
    ]
    scratchpad._saved_tokens_dict["backtrack_level"] = tok("I")
    scratchpad._saved_tokens_dict["conflict_clause"] = tok("-x1 x2 -x3")

    # Before backtrack
    assert len(scratchpad._saved_tokens_dict["decision_levels"]) == 3

    scratchpad._backtrack()

    # After backtrack
    assert scratchpad._saved_tokens_dict["level"] == tok("I")
    assert len(scratchpad._saved_tokens_dict["decision_levels"]) == 1
    assert len(scratchpad._saved_tokens_dict["assignments"]) == 1
    assert len(scratchpad._saved_tokens_dict["reason_clauses"]) == 1
    assert "backtrack_level" not in scratchpad._saved_tokens_dict
    assert "conflict_clause" not in scratchpad._saved_tokens_dict


def test_read_op(cfg, tokenizer):
    registry = CommandRegistry(cfg, tokenizer)

    def tok(text: str) -> list[int]:
        return tokenizer.encode(text).ids

    input_clauses = tok("[ x1 x2 x3 ] [ -x1 x2 -x3 ]")
    scratchpad = CDCLScratchpad(input_clauses, registry)

    scratchpad._saved_tokens_dict["decision_levels"] = [
        tok("I"),
        tok("I I"),
        tok("I I I"),
    ]
    scratchpad._saved_tokens_dict["assignments"] = [
        tok("x1"),
        tok("-x2"),
        tok("x3"),
    ]
    scratchpad._saved_tokens_dict["reason_clauses"] = [
        tok("None"),
        tok("None"),
        tok("x1 x2 x3"),
    ]
    scratchpad._saved_tokens_dict["backtrack_level"] = tok("I")
    scratchpad._saved_tokens_dict["conflict_clause"] = tok("-x1 x2 -x3")

    read_dc_levels = scratchpad.apply("READ_DECISION_LEVELS", payload=[])
    read_assignments = scratchpad.apply("READ_ASSIGNMENTS", payload=[])
    read_reason_clauses = scratchpad.apply("READ_REASON_CLAUSES", payload=[])
    read_bt_level = scratchpad.apply("READ_BACKTRACK_LEVEL", payload=[])
    read_conflict_clause = scratchpad.apply("READ_CONFLICT_CLAUSE", payload=[])
    read_clauses = scratchpad.apply("READ_CLAUSES", payload=[])

    decoded_read_dc_levels = tokenizer.decode(read_dc_levels)
    decoded_read_assignments = tokenizer.decode(read_assignments)
    decoded_read_reason_clauses = tokenizer.decode(read_reason_clauses)
    decoded_read_bt_level = tokenizer.decode(read_bt_level)
    decoded_read_conflict_clause = tokenizer.decode(read_conflict_clause)
    decoded_read_clauses = tokenizer.decode(read_clauses)

    assert decoded_read_dc_levels == "READ_BEGIN [ I ] [ I I ] [ I I I ] READ_END"
    assert decoded_read_assignments == "READ_BEGIN x1 -x2 x3 READ_END"
    assert decoded_read_reason_clauses == "READ_BEGIN [ None ] [ None ] [ x1 x2 x3 ] READ_END"
    assert decoded_read_bt_level == "READ_BEGIN I READ_END"
    assert decoded_read_conflict_clause == "READ_BEGIN -x1 x2 -x3 READ_END"
    assert decoded_read_clauses == "READ_BEGIN [ x1 x2 x3 ] [ -x1 x2 -x3 ] READ_END"

def test_write_op(cfg, tokenizer):
    registry = CommandRegistry(cfg, tokenizer)

    def tok(text: str) -> list[int]:
        return tokenizer.encode(text).ids

    scratchpad = CDCLScratchpad(tokenized_input_clauses=[], registry=registry)

    # Write to all writable fields
    scratchpad.apply("WRITE_ASSIGNMENTS", payload=tok("x1"))
    scratchpad.apply("WRITE_ASSIGNMENTS", payload=tok("-x2"))
    scratchpad.apply("WRITE_REASON_CLAUSES", payload=tok("x1 x2"))
    scratchpad.apply("WRITE_DECISION_LEVELS", payload=tok("I"))
    scratchpad.apply("WRITE_LEARNED_CLAUSES", payload=tok("x3 x4"))
    scratchpad.apply("WRITE_CONFLICT_CLAUSE", payload=tok("-x1 x2 -x3"))
    scratchpad.apply("WRITE_BACKTRACK_LEVEL", payload=tok("I I"))

    # Check that data was written correctly
    assert scratchpad._saved_tokens_dict["assignments"] == [tok("x1"), tok("-x2")]
    assert scratchpad._saved_tokens_dict["reason_clauses"] == [tok("x1 x2")]
    assert scratchpad._saved_tokens_dict["decision_levels"] == [tok("I")]
    assert scratchpad._saved_tokens_dict["learned_clauses"] == [tok("x3 x4")]
    assert scratchpad._saved_tokens_dict["conflict_clause"] == tok("-x1 x2 -x3")
    assert scratchpad._saved_tokens_dict["backtrack_level"] == tok("I I")

    # Optional: test that rendering returns the correct string
    assert tokenizer.decode(scratchpad.apply("READ_ASSIGNMENTS", [])) == "READ_BEGIN x1 -x2 READ_END"
    assert tokenizer.decode(scratchpad.apply("READ_REASON_CLAUSES", [])) == "READ_BEGIN [ x1 x2 ] READ_END"
    assert tokenizer.decode(scratchpad.apply("READ_DECISION_LEVELS", [])) == "READ_BEGIN [ I ] READ_END"
    assert tokenizer.decode(scratchpad.apply("READ_LEARNED_CLAUSES", [])) == "READ_BEGIN [ x3 x4 ] READ_END"
    assert tokenizer.decode(scratchpad.apply("READ_CONFLICT_CLAUSE", [])) == "READ_BEGIN -x1 x2 -x3 READ_END"
    assert tokenizer.decode(scratchpad.apply("READ_BACKTRACK_LEVEL", [])) == "READ_BEGIN I I READ_END"
