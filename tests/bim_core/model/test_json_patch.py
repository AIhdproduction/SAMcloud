"""RFC 6902 JSON Patch and RFC 6901 pointers (examples from the RFC appendices)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from samcloud.bim.core.model.json_patch import (
    JsonPatchError,
    apply_patch,
    apply_patch_with_inverse,
    format_pointer,
    parse_pointer,
    resolve_pointer,
)

# ---------------------------------------------------------------- RFC 6901 pointers

RFC6901_DOC: dict[str, Any] = {
    "foo": ["bar", "baz"],
    "": 0,
    "a/b": 1,
    "c%d": 2,
    "e^f": 3,
    "g|h": 4,
    "i\\j": 5,
    'k"l': 6,
    " ": 7,
    "m~n": 8,
}


@pytest.mark.parametrize(
    ("pointer", "expected"),
    [
        ("", RFC6901_DOC),
        ("/foo", ["bar", "baz"]),
        ("/foo/0", "bar"),
        ("/", 0),
        ("/a~1b", 1),
        ("/c%d", 2),
        ("/e^f", 3),
        ("/g|h", 4),
        ("/i\\j", 5),
        ('/k"l', 6),
        ("/ ", 7),
        ("/m~0n", 8),
    ],
)
def test_rfc6901_examples(pointer: str, expected: Any) -> None:
    assert resolve_pointer(RFC6901_DOC, pointer) == expected


def test_pointer_escaping_roundtrip() -> None:
    tokens = ["a/b", "m~n", "~1", ""]
    assert format_pointer(tokens) == "/a~1b/m~0n/~01/"
    assert parse_pointer(format_pointer(tokens)) == tokens
    assert parse_pointer("/~01") == ["~1"]


@pytest.mark.parametrize("pointer", ["foo", "/a~2b", "/a~"])
def test_invalid_pointers(pointer: str) -> None:
    with pytest.raises(JsonPatchError):
        parse_pointer(pointer)


# ------------------------------------------------------------ RFC 6902 appendix A


def check(document: Any, operations: list[dict[str, Any]], expected: Any) -> None:
    original = copy.deepcopy(document)
    result, inverse = apply_patch_with_inverse(document, operations)
    assert result == expected
    assert document == original  # the input is never modified
    assert apply_patch(result, inverse) == original  # exact inverse


def test_a1_add_object_member() -> None:
    check(
        {"foo": "bar"},
        [{"op": "add", "path": "/baz", "value": "qux"}],
        {"baz": "qux", "foo": "bar"},
    )


def test_a2_add_array_element() -> None:
    check(
        {"foo": ["bar", "baz"]},
        [{"op": "add", "path": "/foo/1", "value": "qux"}],
        {"foo": ["bar", "qux", "baz"]},
    )


def test_a3_remove_object_member() -> None:
    check({"baz": "qux", "foo": "bar"}, [{"op": "remove", "path": "/baz"}], {"foo": "bar"})


def test_a4_remove_array_element() -> None:
    check(
        {"foo": ["bar", "qux", "baz"]},
        [{"op": "remove", "path": "/foo/1"}],
        {"foo": ["bar", "baz"]},
    )


def test_a5_replace_value() -> None:
    check(
        {"baz": "qux", "foo": "bar"},
        [{"op": "replace", "path": "/baz", "value": "boo"}],
        {"baz": "boo", "foo": "bar"},
    )


def test_a6_move_value() -> None:
    check(
        {"foo": {"bar": "baz", "waldo": "fred"}, "qux": {"corge": "grault"}},
        [{"op": "move", "from": "/foo/waldo", "path": "/qux/thud"}],
        {"foo": {"bar": "baz"}, "qux": {"corge": "grault", "thud": "fred"}},
    )


def test_a7_move_array_element() -> None:
    check(
        {"foo": ["all", "grass", "cows", "eat"]},
        [{"op": "move", "from": "/foo/1", "path": "/foo/3"}],
        {"foo": ["all", "cows", "eat", "grass"]},
    )


def test_a8_a9_test_operation() -> None:
    document = {"baz": "qux", "foo": ["a", 2, "c"]}
    check(document, [{"op": "test", "path": "/baz", "value": "qux"}], document)
    check(document, [{"op": "test", "path": "/foo/1", "value": 2}], document)


def test_a10_test_failure_is_an_error() -> None:
    with pytest.raises(JsonPatchError, match="test failed"):
        apply_patch({"baz": "qux"}, [{"op": "test", "path": "/baz", "value": "bar"}])


def test_a11_add_nested_object() -> None:
    check(
        {"foo": "bar"},
        [{"op": "add", "path": "/child", "value": {"grandchild": {}}}],
        {"foo": "bar", "child": {"grandchild": {}}},
    )


def test_a12_add_to_nonexistent_parent_fails() -> None:
    with pytest.raises(JsonPatchError):
        apply_patch({"foo": "bar"}, [{"op": "add", "path": "/baz/bat", "value": "qux"}])


def test_a14_tilde_escape_ordering() -> None:
    document = {"/": 9, "~1": 10}
    check(document, [{"op": "test", "path": "/~01", "value": 10}], document)


def test_a15_comparing_strings_and_numbers() -> None:
    with pytest.raises(JsonPatchError):
        apply_patch({"/": 9, "~1": 10}, [{"op": "test", "path": "/~01", "value": "10"}])


def test_a16_add_array_value() -> None:
    check(
        {"foo": ["bar"]},
        [{"op": "add", "path": "/foo/-", "value": ["abc", "def"]}],
        {"foo": ["bar", ["abc", "def"]]},
    )


# ------------------------------------------------------------------ further cases


def test_copy_and_inverse() -> None:
    check(
        {"a": {"x": [1, 2]}, "b": 0},
        [{"op": "copy", "from": "/a/x", "path": "/b"}],
        {"a": {"x": [1, 2]}, "b": [1, 2]},
    )


def test_copy_does_not_alias() -> None:
    result = apply_patch(
        {"a": [1]},
        [
            {"op": "copy", "from": "/a", "path": "/b"},
            {"op": "add", "path": "/b/-", "value": 2},
        ],
    )
    assert result == {"a": [1], "b": [1, 2]}


def test_add_replaces_existing_object_member_and_inverse_restores() -> None:
    check({"a": 1}, [{"op": "add", "path": "/a", "value": 2}], {"a": 2})


def test_move_onto_existing_target_restores_both() -> None:
    check({"a": 1, "b": 2}, [{"op": "move", "from": "/a", "path": "/b"}], {"b": 1})


def test_append_with_dash_inverse_removes_last() -> None:
    check({"l": [1]}, [{"op": "add", "path": "/l/-", "value": 2}], {"l": [1, 2]})


def test_multi_operation_inverse_is_applied_in_reverse() -> None:
    check(
        {"a": [1, 2, 3], "n": {"k": "v"}},
        [
            {"op": "remove", "path": "/a/0"},
            {"op": "replace", "path": "/n/k", "value": "w"},
            {"op": "add", "path": "/a/1", "value": 9},
            {"op": "move", "from": "/n", "path": "/m"},
        ],
        {"a": [2, 9, 3], "m": {"k": "w"}},
    )


def test_root_replace_and_inverse() -> None:
    check({"a": 1}, [{"op": "replace", "path": "", "value": [1]}], [1])


def test_test_distinguishes_bool_and_number() -> None:
    with pytest.raises(JsonPatchError):
        apply_patch({"a": True}, [{"op": "test", "path": "/a", "value": 1}])
    apply_patch({"a": 1.0}, [{"op": "test", "path": "/a", "value": 1}])


def test_failed_patch_leaves_input_unchanged() -> None:
    document = {"a": 1}
    with pytest.raises(JsonPatchError):
        apply_patch(
            document,
            [{"op": "replace", "path": "/a", "value": 2}, {"op": "remove", "path": "/zzz"}],
        )
    assert document == {"a": 1}


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "bogus", "path": "/a"},
        {"op": "add", "path": "/a"},
        {"op": "move", "path": "/a"},
        {"op": "remove"},
        {"op": "replace", "path": "/missing", "value": 1},
        {"op": "remove", "path": "/l/5"},
        {"op": "remove", "path": "/l/01"},
        {"op": "add", "path": "/l/3", "value": 1},
        {"op": "add", "path": "/a/x", "value": 1},
        {"op": "remove", "path": ""},
        {"op": "move", "from": "/n", "path": "/n/inner"},
    ],
)
def test_invalid_operations(operation: dict[str, Any]) -> None:
    with pytest.raises(JsonPatchError):
        apply_patch({"a": 1, "l": [1, 2], "n": {}}, [operation])
