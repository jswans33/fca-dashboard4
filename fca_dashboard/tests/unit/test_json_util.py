import os
import tempfile

import pytest

from fca_dashboard.utils.json_utils import (
    json_deserialize,
    json_is_valid,
    json_load,
    json_save,
    json_serialize,
    pretty_print_json,
    safe_get,
    safe_get_nested,
)


def test_json_load_and_save():
    data = {"key": "value", "number": 42}
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        path = tmp.name
        json_save(data, path)

    loaded_data = json_load(path)
    assert loaded_data == data

    os.unlink(path)


def test_json_serialize():
    data = {"name": "Alice", "age": 30}
    json_str = json_serialize(data)
    assert json_str == '{"name": "Alice", "age": 30}'


def test_json_deserialize_valid():
    json_str = '{"valid": true, "value": 10}'
    result = json_deserialize(json_str)
    assert result == {"valid": True, "value": 10}


def test_json_deserialize_invalid():
    json_str = '{invalid json}'
    default = {"default": True}
    result = json_deserialize(json_str, default=default)
    assert result == default


def test_json_is_valid():
    assert json_is_valid('{"valid": true}') is True
    assert json_is_valid('{invalid json}') is False


def test_pretty_print_json():
    data = {"key": "value"}
    expected = '{\n  "key": "value"\n}'
    assert pretty_print_json(data) == expected


def test_safe_get():
    data = {"a": 1, "b": None}
    assert safe_get(data, "a") == 1
    assert safe_get(data, "b", default="default") is None
    assert safe_get(data, "missing", default="default") == "default"


def test_safe_get_nested():
    data = {"a": {"b": {"c": 42}}}
    assert safe_get_nested(data, "a", "b", "c") == 42
    assert safe_get_nested(data, "a", "x", default="missing") == "missing"
    assert safe_get_nested(data, "a", "b", "c", "d", default=None) is None


def test_json_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        json_load("nonexistent_file.json")


def test_json_load_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("{invalid json}")
        path = tmp.name

    with pytest.raises(Exception):
        json_load(path)

    os.unlink(path)


def test_json_save_and_load_unicode():
    data = {"message": "こんにちは世界"}
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        path = tmp.name
        json_save(data, path)

    loaded_data = json_load(path)
    assert loaded_data == data

    os.unlink(path)
