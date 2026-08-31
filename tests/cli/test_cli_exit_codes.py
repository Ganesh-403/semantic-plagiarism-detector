"""
tests/test_cli_exit_codes.py
----------------------------
Tests for the CLI exit-code registry (issue #3783).

`src/cli_exit_codes.py` had no test coverage at all, which is how it came to
be committed in a state where it could not even be imported: `sys`, `datetime`
and `typing.Any` were all used but never imported, and the `Any` reference sat
in a method annotation inside a class body, so it blew up at definition time
rather than lazily.

These tests exercise the three names that were missing, then pin the public
surface so the module cannot silently rot again.
"""

import json
import subprocess
import sys

import pytest

from src.cli_exit_codes import (
    CliExitCodes,
    ExitCodeManager,
    ExitHandler,
    ExitInfo,
    exit_with_code,
    get_exit_code,
    list_all_exit_codes,
)

# (code, category) pairs, one per documented band.
CATEGORY_BANDS = [
    (0, "Success"),
    (1, "Argument/Input Error"),
    (19, "Argument/Input Error"),
    (20, "File/Path Error"),
    (39, "File/Path Error"),
    (40, "Runtime Error"),
    (59, "Runtime Error"),
    (60, "Network Error"),
    (79, "Network Error"),
    (80, "System/Environment Error"),
    (99, "System/Environment Error"),
    (100, "Configuration Error"),
    (119, "Configuration Error"),
    (120, "Data/Processing Error"),
    (139, "Data/Processing Error"),
    (140, "Permission/Security Error"),
    (159, "Permission/Security Error"),
    (160, "External Service Error"),
    (179, "External Service Error"),
]


# ── the regression itself ──────────────────────────────────────────────────────


def test_module_imports_in_a_clean_interpreter():
    """`Any` was undefined in a class-body annotation, so import raised.

    Run it out of process: an in-process import would be satisfied by the
    already-imported module object and prove nothing.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import src.cli_exit_codes"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_sys_is_available_to_the_exit_helpers():
    import src.cli_exit_codes as module

    assert module.sys is sys


def test_datetime_is_available_to_exit_info():
    from datetime import datetime

    import src.cli_exit_codes as module

    assert module.datetime is datetime


# ── ExitInfo ───────────────────────────────────────────────────────────────────


def test_exit_info_from_code_fills_every_field():
    info = ExitInfo.from_code(CliExitCodes.FILE_NOT_FOUND)

    assert info.code == CliExitCodes.FILE_NOT_FOUND
    assert info.message == CliExitCodes.get_description(CliExitCodes.FILE_NOT_FOUND)
    assert info.category == "File/Path Error"
    assert info.timestamp


def test_exit_info_timestamp_is_iso8601():
    """`datetime` was undefined, so this used to raise NameError."""
    from datetime import datetime

    info = ExitInfo.from_code(1)

    # Round-trips only if it is genuinely ISO-8601.
    assert isinstance(datetime.fromisoformat(info.timestamp), datetime)


def test_exit_info_custom_message_overrides_the_description():
    info = ExitInfo.from_code(1, "you passed --frobnicate twice")

    assert info.message == "you passed --frobnicate twice"


def test_exit_info_blank_message_falls_back_to_the_description():
    info = ExitInfo.from_code(1, "")

    assert info.message == CliExitCodes.get_description(1)


def test_exit_info_to_dict_round_trips_through_json():
    """`to_dict`'s `Any` annotation is what broke the import."""
    payload = ExitInfo.from_code(21, "missing").to_dict()

    assert set(payload) == {"code", "message", "category", "timestamp"}
    assert json.loads(json.dumps(payload)) == payload


# ── exit_with_code ─────────────────────────────────────────────────────────────


def test_exit_with_code_raises_system_exit_with_that_code():
    """`sys.exit` was undefined, so this used to raise NameError instead."""
    with pytest.raises(SystemExit) as excinfo:
        exit_with_code(CliExitCodes.FILE_NOT_FOUND)

    assert excinfo.value.code == CliExitCodes.FILE_NOT_FOUND


def test_exit_with_code_writes_the_message_to_stderr(capsys):
    with pytest.raises(SystemExit):
        exit_with_code(2, "missing --input")

    captured = capsys.readouterr()
    assert "missing --input" in captured.err
    assert captured.out == ""


def test_exit_with_code_falls_back_to_the_description(capsys):
    with pytest.raises(SystemExit):
        exit_with_code(CliExitCodes.FILE_NOT_FOUND)

    assert CliExitCodes.get_description(CliExitCodes.FILE_NOT_FOUND) in (
        capsys.readouterr().err
    )


# ── ExitHandler ────────────────────────────────────────────────────────────────


def test_handler_handle_delegates_to_exit_with_code():
    with pytest.raises(SystemExit) as excinfo:
        ExitHandler.handle(40, "boom")

    assert excinfo.value.code == 40


def test_handler_handle_with_info_reports_to_stderr(capsys):
    with pytest.raises(SystemExit) as excinfo:
        ExitHandler.handle_with_info(21, "no such file")

    assert excinfo.value.code == 21
    assert "no such file" in capsys.readouterr().err


def test_handler_success_exits_zero_on_stdout(capsys):
    with pytest.raises(SystemExit) as excinfo:
        ExitHandler.success("done")

    assert excinfo.value.code == CliExitCodes.SUCCESS
    captured = capsys.readouterr()
    assert "done" in captured.out
    assert captured.err == ""


def test_handler_error_exits_with_the_code_on_stderr(capsys):
    with pytest.raises(SystemExit) as excinfo:
        ExitHandler.error(60, "connection refused")

    assert excinfo.value.code == 60
    assert "connection refused" in capsys.readouterr().err


# ── classification ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("code,category", CATEGORY_BANDS)
def test_get_category_maps_each_band(code, category):
    assert CliExitCodes.get_category(code) == category


@pytest.mark.parametrize("code", [180, 999, -1])
def test_get_category_is_unknown_outside_the_bands(code):
    assert CliExitCodes.get_category(code) == "Unknown Category"


def test_success_and_error_are_complementary():
    assert CliExitCodes.is_success(0)
    assert not CliExitCodes.is_error(0)
    assert CliExitCodes.is_error(1)
    assert not CliExitCodes.is_success(1)


@pytest.mark.parametrize(
    "predicate,inside,outside",
    [
        (CliExitCodes.is_invalid_args, 1, 20),
        (CliExitCodes.is_file_error, 20, 40),
        (CliExitCodes.is_runtime_error, 40, 60),
        (CliExitCodes.is_network_error, 60, 80),
        (CliExitCodes.is_system_error, 80, 100),
        (CliExitCodes.is_config_error, 100, 120),
        (CliExitCodes.is_data_error, 120, 140),
        (CliExitCodes.is_permission_error, 140, 160),
        (CliExitCodes.is_external_error, 160, 180),
    ],
)
def test_band_predicates_have_the_right_boundaries(predicate, inside, outside):
    assert predicate(inside)
    assert not predicate(outside)


def test_every_member_lands_in_exactly_one_band():
    """A member falling in no band, or two, means the ranges have drifted."""
    predicates = [
        CliExitCodes.is_invalid_args,
        CliExitCodes.is_file_error,
        CliExitCodes.is_runtime_error,
        CliExitCodes.is_network_error,
        CliExitCodes.is_system_error,
        CliExitCodes.is_config_error,
        CliExitCodes.is_data_error,
        CliExitCodes.is_permission_error,
        CliExitCodes.is_external_error,
    ]

    for member in CliExitCodes:
        if member is CliExitCodes.SUCCESS:
            continue
        matches = [p for p in predicates if p(member.value)]
        assert len(matches) == 1, f"{member.name} matched {len(matches)} bands"


# ── lookup helpers ─────────────────────────────────────────────────────────────


def test_get_exit_code_resolves_a_known_name():
    assert get_exit_code("FILE_NOT_FOUND") == CliExitCodes.FILE_NOT_FOUND.value


def test_get_exit_code_returns_none_for_an_unknown_name():
    assert get_exit_code("NOT_A_REAL_CODE") is None


def test_get_exit_code_classmethod_resolves_a_member():
    assert CliExitCodes.get_exit_code(0) is CliExitCodes.SUCCESS


def test_get_exit_code_classmethod_returns_none_for_an_unused_value():
    assert CliExitCodes.get_exit_code(9999) is None


def test_get_exit_code_name_falls_back_to_unknown():
    assert CliExitCodes.get_exit_code_name(9999) == "UNKNOWN"


def test_get_description_is_non_empty_for_every_member():
    for member in CliExitCodes:
        assert CliExitCodes.get_description(member.value).strip()


# ── serialisation ──────────────────────────────────────────────────────────────


def test_to_dict_covers_every_member():
    mapping = CliExitCodes.to_dict()

    assert len(mapping) == len(list(CliExitCodes))
    assert mapping[0] == "SUCCESS"


def test_to_json_is_valid_json_matching_to_dict():
    parsed = json.loads(CliExitCodes.to_json())

    assert {int(k): v for k, v in parsed.items()} == CliExitCodes.to_dict()


def test_exit_code_values_are_unique():
    """Aliased IntEnum members would silently collapse into one."""
    names = [m.name for m in CliExitCodes]
    assert len(names) == len(set(names))


def test_print_all_writes_a_row_per_code(capsys):
    CliExitCodes.print_all()

    lines = capsys.readouterr().out.splitlines()
    # header + separator + one row per member
    assert len(lines) == len(list(CliExitCodes)) + 2


def test_list_all_exit_codes_produces_output(capsys):
    list_all_exit_codes()

    assert capsys.readouterr().out.strip()


# ── ExitCodeManager ────────────────────────────────────────────────────────────


def test_manager_agrees_with_the_classmethods():
    manager = ExitCodeManager()
    code = CliExitCodes.FILE_NOT_FOUND.value

    assert manager.get_code("FILE_NOT_FOUND") == code
    assert manager.get_name(code) == "FILE_NOT_FOUND"
    assert manager.get_description(code) == CliExitCodes.get_description(code)
    assert manager.get_category(code) == CliExitCodes.get_category(code)


def test_manager_predicates_agree_with_the_classmethods():
    manager = ExitCodeManager()

    assert manager.is_success(0) is True
    assert manager.is_error(0) is False
    assert manager.is_file_error(21) is True
    assert manager.is_network_error(21) is False
