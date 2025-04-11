from src.model.registry import CommandRegistry
from src.model.parser import CommandParser


def test_read_command_parsing(cfg, tokenizer):
    registry = CommandRegistry(cfg, tokenizer)
    parser = CommandParser(registry)

    # Simulate token sequence: [READ_ASSIGNMENTS]
    read_token = registry.read_cmd_tokens["READ_ASSIGNMENTS"]
    result = parser.step(read_token)

    assert result == ("READ_ASSIGNMENTS", [])
    assert parser.state == "idle"
    assert parser.command is None


def test_call_up_parsing(cfg, tokenizer):
    registry = CommandRegistry(cfg, tokenizer)
    parser = CommandParser(registry)

    # Simulate token sequence: [CALL_UNIT_PROPAGATION]
    read_token = registry.action_cmd_tokens["CALL_UNIT_PROPAGATION"]
    result = parser.step(read_token)

    assert result == ("CALL_UNIT_PROPAGATION", [])
    assert parser.state == "idle"
    assert parser.command is None


def test_write_command_parsing(cfg, tokenizer):
    registry = CommandRegistry(cfg, tokenizer)
    parser = CommandParser(registry)

    def tok(text): return tokenizer.encode(text).ids

    # Simulate: WRITE_ASSIGNMENTS -> WRITE_BEGIN -> x1 x2 -> WRITE_END
    tokens = [
        registry.write_cmd_tokens["WRITE_ASSIGNMENTS"],
        registry.write_block_markers[0],  # WRITE_BEGIN
        *tok("x1 x2"),
        registry.write_block_markers[1],  # WRITE_END
    ]

    result = None
    for t in tokens:
        r = parser.step(t)
        if r is not None:
            result = r

    assert result is not None
    command, payload = result
    assert command == "WRITE_ASSIGNMENTS"
    assert payload == tok("x1 x2")
    assert parser.state == "idle"
    assert parser.command is None
    assert parser.payload == []


def test_incomplete_write_command_returns_none(cfg, tokenizer):
    registry = CommandRegistry(cfg, tokenizer)
    parser = CommandParser(registry)

    write_tok = registry.write_cmd_tokens["WRITE_ASSIGNMENTS"]
    begin = registry.write_block_markers[0]

    # Start WRITE but never end it
    assert parser.step(write_tok) is None
    assert parser.step(begin) is None
    assert parser.step(123) is None  # payload token
    assert parser.command == "WRITE_ASSIGNMENTS"
    assert parser.payload == [123]
    assert parser.state == "payload"
