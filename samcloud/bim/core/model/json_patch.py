"""RFC 6902 JSON Patch and RFC 6901 JSON Pointer, with inverse operations.

:func:`apply_patch_with_inverse` applies a patch to a copy of a JSON document and also returns
the operations that undo it exactly. The inverse is computed from the document state at the
moment each operation runs, so it restores replaced and removed values. ``move`` and ``copy``
are decomposed into ``remove`` and ``add`` for the inverse. Restoring a removed object key
puts it at the end of the object (JSON objects are unordered); array positions are restored
exactly.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any

Operation = dict[str, Any]
OPERATIONS = ("add", "remove", "replace", "move", "copy", "test")


class JsonPatchError(ValueError):
    """A patch is malformed or cannot be applied to the document."""


# ------------------------------------------------------------------------------ pointer


def parse_pointer(pointer: str) -> list[str]:
    """Split an RFC 6901 pointer into unescaped reference tokens (``""`` gives ``[]``)."""
    if not isinstance(pointer, str):
        raise JsonPatchError(f"pointer must be a string, got {type(pointer).__name__}")
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise JsonPatchError(f"invalid JSON pointer {pointer!r}: must be empty or start with '/'")
    tokens = pointer[1:].split("/")
    result: list[str] = []
    for token in tokens:
        # "~" must only appear as the escapes "~0" and "~1"
        index = 0
        while True:
            index = token.find("~", index)
            if index < 0:
                break
            if token[index + 1 : index + 2] not in ("0", "1"):
                raise JsonPatchError(f"invalid escape in JSON pointer {pointer!r}")
            index += 2
        result.append(token.replace("~1", "/").replace("~0", "~"))
    return result


def format_pointer(tokens: Iterable[str]) -> str:
    """Build an RFC 6901 pointer from reference tokens (escaping ``~`` and ``/``)."""
    return "".join("/" + str(token).replace("~", "~0").replace("/", "~1") for token in tokens)


def _array_index(token: str, length: int, pointer: str, *, allow_end: bool) -> int:
    if token == "-" and allow_end:
        return length
    if not token.isascii() or not token.isdigit() or (len(token) > 1 and token[0] == "0"):
        raise JsonPatchError(f"invalid array index {token!r} in {pointer!r}")
    index = int(token)
    limit = length if allow_end else length - 1
    if index > limit:
        raise JsonPatchError(f"array index {index} out of range in {pointer!r}")
    return index


def _child(container: Any, token: str, pointer: str) -> Any:
    if isinstance(container, dict):
        if token not in container:
            raise JsonPatchError(f"path {pointer!r} does not exist (missing key {token!r})")
        return container[token]
    if isinstance(container, list):
        return container[_array_index(token, len(container), pointer, allow_end=False)]
    raise JsonPatchError(f"path {pointer!r} does not exist (cannot descend into a scalar)")


def resolve_pointer(document: Any, pointer: str) -> Any:
    """Return the value at ``pointer`` (the live object, not a copy)."""
    current = document
    for token in parse_pointer(pointer):
        current = _child(current, token, pointer)
    return current


def _parent(document: Any, tokens: list[str], pointer: str) -> Any:
    current = document
    for token in tokens[:-1]:
        current = _child(current, token, pointer)
    return current


# --------------------------------------------------------------------------- primitives


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_json_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, bool | list | Mapping) or isinstance(right, bool | list | Mapping):
        return False
    return bool(left == right)


def _add(document: Any, pointer: str, value: Any, inverse: list[Operation]) -> Any:
    tokens = parse_pointer(pointer)
    value = copy.deepcopy(value)
    if not tokens:
        inverse.insert(0, {"op": "replace", "path": "", "value": copy.deepcopy(document)})
        return value
    parent = _parent(document, tokens, pointer)
    last = tokens[-1]
    if isinstance(parent, dict):
        if last in parent:
            inverse.insert(
                0, {"op": "replace", "path": pointer, "value": copy.deepcopy(parent[last])}
            )
        else:
            inverse.insert(0, {"op": "remove", "path": pointer})
        parent[last] = value
    elif isinstance(parent, list):
        index = _array_index(last, len(parent), pointer, allow_end=True)
        parent.insert(index, value)
        inverse.insert(0, {"op": "remove", "path": format_pointer([*tokens[:-1], str(index)])})
    else:
        raise JsonPatchError(f"cannot add at {pointer!r}: parent is a scalar")
    return document


def _remove(document: Any, pointer: str, inverse: list[Operation]) -> tuple[Any, Any]:
    tokens = parse_pointer(pointer)
    if not tokens:
        raise JsonPatchError("cannot remove the document root")
    parent = _parent(document, tokens, pointer)
    last = tokens[-1]
    if isinstance(parent, dict):
        if last not in parent:
            raise JsonPatchError(f"path {pointer!r} does not exist (missing key {last!r})")
        removed = parent.pop(last)
    elif isinstance(parent, list):
        removed = parent.pop(_array_index(last, len(parent), pointer, allow_end=False))
    else:
        raise JsonPatchError(f"path {pointer!r} does not exist (parent is a scalar)")
    inverse.insert(0, {"op": "add", "path": pointer, "value": copy.deepcopy(removed)})
    return document, removed


def _replace(document: Any, pointer: str, value: Any, inverse: list[Operation]) -> Any:
    tokens = parse_pointer(pointer)
    value = copy.deepcopy(value)
    if not tokens:
        inverse.insert(0, {"op": "replace", "path": "", "value": copy.deepcopy(document)})
        return value
    parent = _parent(document, tokens, pointer)
    last = tokens[-1]
    if isinstance(parent, dict):
        if last not in parent:
            raise JsonPatchError(f"path {pointer!r} does not exist (missing key {last!r})")
        old = parent[last]
        parent[last] = value
    elif isinstance(parent, list):
        index = _array_index(last, len(parent), pointer, allow_end=False)
        old = parent[index]
        parent[index] = value
    else:
        raise JsonPatchError(f"path {pointer!r} does not exist (parent is a scalar)")
    inverse.insert(0, {"op": "replace", "path": pointer, "value": copy.deepcopy(old)})
    return document


def validate_operation(operation: Any) -> Operation:
    """Check the shape of one operation and return it."""
    if not isinstance(operation, Mapping):
        raise JsonPatchError(f"operation must be an object, got {type(operation).__name__}")
    name = operation.get("op")
    if name not in OPERATIONS:
        raise JsonPatchError(f"unknown operation {name!r}")
    if not isinstance(operation.get("path"), str):
        raise JsonPatchError(f"operation {name!r} needs a string 'path'")
    if name in ("add", "replace", "test") and "value" not in operation:
        raise JsonPatchError(f"operation {name!r} needs a 'value'")
    if name in ("move", "copy") and not isinstance(operation.get("from"), str):
        raise JsonPatchError(f"operation {name!r} needs a string 'from'")
    return dict(operation)


def _apply_one(document: Any, operation: Operation, inverse: list[Operation]) -> Any:
    name = operation["op"]
    path = operation["path"]
    if name == "add":
        return _add(document, path, operation["value"], inverse)
    if name == "remove":
        return _remove(document, path, inverse)[0]
    if name == "replace":
        return _replace(document, path, operation["value"], inverse)
    if name == "test":
        actual = resolve_pointer(document, path)
        if not _json_equal(actual, operation["value"]):
            raise JsonPatchError(
                f"test failed at {path!r}: expected {operation['value']!r}, found {actual!r}"
            )
        return document
    source = operation["from"]
    if name == "copy":
        value = copy.deepcopy(resolve_pointer(document, source))
        return _add(document, path, value, inverse)
    # move
    if source == path:
        resolve_pointer(document, source)
        return document
    source_tokens = parse_pointer(source)
    if parse_pointer(path)[: len(source_tokens)] == source_tokens:
        raise JsonPatchError(f"cannot move {source!r} into its own child {path!r}")
    document, value = _remove(document, source, inverse)
    return _add(document, path, value, inverse)


def apply_patch_with_inverse(
    document: Any, operations: Iterable[Mapping[str, Any]]
) -> tuple[Any, list[Operation]]:
    """Apply a patch to a deep copy of ``document``; return the new document and the inverse.

    The patch is atomic: any failing operation raises :class:`JsonPatchError` and the input is
    untouched (it is never modified). Applying the returned inverse to the returned document
    gives back a document equal to the input.
    """
    result = copy.deepcopy(document)
    inverse: list[Operation] = []
    for raw in operations:
        operation = validate_operation(raw)
        try:
            result = _apply_one(result, operation, inverse)
        except JsonPatchError:
            raise
        except (KeyError, IndexError, TypeError) as error:  # pragma: no cover - safety net
            raise JsonPatchError(f"cannot apply {operation!r}: {error}") from error
    return result, inverse


def apply_patch(document: Any, operations: Iterable[Mapping[str, Any]]) -> Any:
    """Apply a patch to a deep copy of ``document`` and return the new document."""
    return apply_patch_with_inverse(document, operations)[0]


__all__ = [
    "JsonPatchError",
    "Operation",
    "apply_patch",
    "apply_patch_with_inverse",
    "format_pointer",
    "parse_pointer",
    "resolve_pointer",
    "validate_operation",
]
