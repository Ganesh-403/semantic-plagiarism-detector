"""Regression tests for #4081.

``create_jwt_token`` read two names it never bound::

    alg = (algorithm or get_jwt_algorithm()).upper()

    is_test_env = (os.getenv("APP_ENV") == "test") or _IS_TEST
    if len(resolved_secret) < 32 and not is_test_env:   # <-- unbound
        ...
    if alg == "RS256":
        signature = _sign_rs256(signing_input, private_key)   # <-- unbound
    else:
        signature = hmac.new(resolved_secret.encode("utf-8"), ...)   # <-- unbound

Commit ``1bbceaeb`` ("feat: Support asymmetric RS256/ES256 signatures") had the
per-algorithm key-resolution block those checks depend on; the merge commit
``edde67d1`` deleted the block and kept the checks. The length check sits before
any branch, so *every* call raised ``NameError`` regardless of algorithm,
secret, or environment — no token could be minted at all, while
``verify_jwt_token`` kept working and made the breakage one-sided.

The tests below pin three things: that issuance works again, that the failure
modes raise the documented ``ValueError`` rather than ``NameError``, and — the
part that actually prevents a recurrence — that no function in the module reads
a local name it never binds.
"""

from __future__ import annotations

import ast
import importlib
import json
import pathlib
import time

import pytest

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "security" / "jwt_utils.py"
)

STRONG_SECRET = "s" * 48
SHORT_SECRET = "tooshort"


@pytest.fixture()
def jwt_mod(monkeypatch):
    """The module with a clean, production-shaped environment."""
    module = importlib.import_module("src.security.jwt_utils")
    monkeypatch.setenv("JWT_SECRET_KEY", STRONG_SECRET)
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.setattr(module, "_IS_TEST", False, raising=False)
    monkeypatch.setattr(module, "JWT_SECRET_KEY", STRONG_SECRET, raising=False)
    return module


def _claims(mod, token: str) -> dict:
    return json.loads(mod.base64url_decode(token.split(".")[1]).decode("utf-8"))


# ── The defect, pinned structurally ──────────────────────────────────────────


def _defined_names(func: ast.FunctionDef) -> set[str]:
    """Every name ``func`` itself binds: parameters, assignments, imports, ..."""
    names: set[str] = {a.arg for a in func.args.args}
    names |= {a.arg for a in func.args.posonlyargs}
    names |= {a.arg for a in func.args.kwonlyargs}
    for extra in (func.args.vararg, func.args.kwarg):
        if extra is not None:
            names.add(extra.arg)

    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node is not func:
                names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def _module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.Assign):
            names.update(
                t.id for t in node.targets if isinstance(t, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Try):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    names.update(a.asname or a.name.split(".")[0] for a in sub.names)
                elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    names.add(sub.id)
    return names


def _undefined_names(func: ast.FunctionDef, tree: ast.Module) -> set[str]:
    """Names read inside ``func`` that resolve to nothing at all.

    This is deliberately order-insensitive. A conditional binding
    (``x`` assigned in one branch, read in another) is legitimate Python and
    is not flagged; what #4081 was is different and much starker —
    ``resolved_secret`` and ``private_key`` were assigned *nowhere* in the
    function and are not module globals or builtins either, so every path that
    reached them raised ``NameError``.
    """
    import builtins

    known = _defined_names(func) | _module_level_names(tree) | set(dir(builtins))
    read = {
        n.id
        for n in ast.walk(func)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    return read - known


def test_create_jwt_token_binds_every_name_it_reads() -> None:
    """The exact shape of #4081: ``resolved_secret`` and ``private_key``."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    func = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "create_jwt_token"
    )
    assert _undefined_names(func, tree) == set()


def test_no_module_function_reads_an_undefined_name() -> None:
    """Same check across the module, so the next bad merge is caught here."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    offenders = {
        n.name: sorted(_undefined_names(n, tree))
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and _undefined_names(n, tree)
    }
    assert offenders == {}, f"functions reading undefined names: {offenders}"


def test_the_undefined_name_detector_catches_the_original_defect() -> None:
    """Guard the guard against the code that actually shipped."""
    broken = ast.parse(
        "import os\n"
        "JWT_SECRET_KEY = None\n"
        "def create_jwt_token(data, algorithm=None):\n"
        "    alg = (algorithm or 'HS256').upper()\n"
        "    if len(resolved_secret) < 32:\n"
        "        raise ValueError('short')\n"
        "    return resolved_secret\n"
    )
    func = next(
        n for n in broken.body
        if isinstance(n, ast.FunctionDef) and n.name == "create_jwt_token"
    )
    assert _undefined_names(func, broken) == {"resolved_secret"}


def test_the_undefined_name_detector_allows_a_conditional_binding() -> None:
    """A name bound in only one branch is legitimate and must not be flagged."""
    ok = ast.parse(
        "def f(alg):\n"
        "    if alg == 'RS256':\n"
        "        key = 'private'\n"
        "    else:\n"
        "        key = 'secret'\n"
        "    return key\n"
    )
    func = ok.body[0]
    assert _undefined_names(func, ok) == set()


# ── HS256 issuance ───────────────────────────────────────────────────────────


def test_create_jwt_token_returns_a_three_part_token(jwt_mod) -> None:
    token = jwt_mod.create_jwt_token({"sub": "alice"})
    assert isinstance(token, str)
    assert len(token.split(".")) == 3


def test_created_token_verifies(jwt_mod) -> None:
    """Issue and verify must agree — the two halves had drifted apart."""
    token = jwt_mod.create_jwt_token({"sub": "alice", "type": "access"})
    payload = jwt_mod.verify_access_token(token)
    assert payload["sub"] == "alice"


def test_created_token_carries_the_registered_claims(jwt_mod) -> None:
    before = int(time.time())
    token = jwt_mod.create_jwt_token({"sub": "alice"}, expires_in_seconds=120)
    claims = _claims(jwt_mod, token)

    assert claims["sub"] == "alice"
    assert claims["iat"] >= before
    assert claims["exp"] == claims["iat"] + 120
    assert claims["jti"]


def test_jti_is_unique_per_token(jwt_mod) -> None:
    a = _claims(jwt_mod, jwt_mod.create_jwt_token({"sub": "alice"}))
    b = _claims(jwt_mod, jwt_mod.create_jwt_token({"sub": "alice"}))
    assert a["jti"] != b["jti"]


def test_header_names_the_algorithm(jwt_mod) -> None:
    token = jwt_mod.create_jwt_token({"sub": "alice"})
    header = json.loads(jwt_mod.base64url_decode(token.split(".")[0]).decode("utf-8"))
    assert header == {"alg": "HS256", "typ": "JWT"}


def test_access_and_refresh_helpers_work(jwt_mod) -> None:
    """Both were collateral damage — they route through create_jwt_token."""
    access = _claims(jwt_mod, jwt_mod.create_access_token(sub="alice"))
    refresh = _claims(jwt_mod, jwt_mod.create_refresh_token(sub="alice"))

    assert access["type"] == "access"
    assert refresh["type"] == "refresh"
    assert refresh["exp"] - refresh["iat"] > access["exp"] - access["iat"]


def test_explicit_secret_key_overrides_the_environment(jwt_mod, monkeypatch) -> None:
    other = "o" * 48
    token = jwt_mod.create_jwt_token({"sub": "alice"}, secret_key=other)
    # Verifiable with the key it was signed with, not with the environment one.
    assert jwt_mod.verify_access_token(token, secret_key=other)["sub"] == "alice"
    with pytest.raises(jwt_mod.JWTSignatureError):
        jwt_mod.verify_access_token(token, secret_key=STRONG_SECRET)


# ── HS256 failure modes: ValueError, never NameError ─────────────────────────


def test_missing_secret_raises_a_useful_valueerror(jwt_mod, monkeypatch) -> None:
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.setattr(jwt_mod, "JWT_SECRET_KEY", None, raising=False)

    with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
        jwt_mod.create_jwt_token({"sub": "alice"})


def test_short_secret_is_refused_in_production(jwt_mod, monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", SHORT_SECRET)
    monkeypatch.setattr(jwt_mod, "JWT_SECRET_KEY", SHORT_SECRET, raising=False)

    with pytest.raises(ValueError, match="at least 32 characters"):
        jwt_mod.create_jwt_token({"sub": "alice"})


def test_short_secret_is_allowed_under_app_env_test(jwt_mod, monkeypatch) -> None:
    """The test-environment escape hatch must survive the repair."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("JWT_SECRET_KEY", SHORT_SECRET)
    monkeypatch.setattr(jwt_mod, "JWT_SECRET_KEY", SHORT_SECRET, raising=False)

    assert len(jwt_mod.create_jwt_token({"sub": "alice"}).split(".")) == 3


def test_short_secret_is_allowed_when_is_test_flag_is_set(jwt_mod, monkeypatch) -> None:
    monkeypatch.setattr(jwt_mod, "_IS_TEST", True, raising=False)
    monkeypatch.setenv("JWT_SECRET_KEY", SHORT_SECRET)
    monkeypatch.setattr(jwt_mod, "JWT_SECRET_KEY", SHORT_SECRET, raising=False)

    assert len(jwt_mod.create_jwt_token({"sub": "alice"}).split(".")) == 3


@pytest.mark.parametrize("bad", [None, ""])
def test_empty_environment_secret_is_refused(jwt_mod, monkeypatch, bad) -> None:
    if bad is None:
        monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("JWT_SECRET_KEY", bad)
    monkeypatch.setattr(jwt_mod, "JWT_SECRET_KEY", bad, raising=False)

    with pytest.raises(ValueError):
        jwt_mod.create_jwt_token({"sub": "alice"})


# ── RS256 ────────────────────────────────────────────────────────────────────


@pytest.fixture()
def rsa_keys():
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = (
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    assert crypto  # silence the unused-name linter
    return private_pem, public_pem


def test_rs256_round_trip(jwt_mod, monkeypatch, rsa_keys) -> None:
    private_pem, public_pem = rsa_keys
    monkeypatch.setenv("JWT_ALGORITHM", "RS256")
    monkeypatch.setenv("JWT_PRIVATE_KEY", private_pem)
    monkeypatch.setattr(jwt_mod, "JWT_PRIVATE_KEY", private_pem, raising=False)
    monkeypatch.setattr(jwt_mod, "JWT_PUBLIC_KEY", public_pem, raising=False)
    monkeypatch.setenv("JWT_PUBLIC_KEY", public_pem)

    token = jwt_mod.create_jwt_token({"sub": "alice", "type": "access"})
    assert jwt_mod.verify_access_token(token)["sub"] == "alice"


def test_rs256_missing_private_key_raises_valueerror(jwt_mod, monkeypatch) -> None:
    monkeypatch.setenv("JWT_ALGORITHM", "RS256")
    monkeypatch.delenv("JWT_PRIVATE_KEY", raising=False)
    monkeypatch.setattr(jwt_mod, "JWT_PRIVATE_KEY", None, raising=False)

    with pytest.raises(ValueError, match="JWT_PRIVATE_KEY"):
        jwt_mod.create_jwt_token({"sub": "alice"})


def test_rs256_does_not_apply_the_hmac_length_rule(jwt_mod, monkeypatch) -> None:
    """A PEM private key is not a shared secret.

    Counting its characters says nothing about brute-force resistance, so the
    ">= 32 characters" rule must not gate the RS256 branch. A too-short key
    should fail at *signing* with an RS256-specific message, not be rejected
    up front as a weak HMAC secret.
    """
    monkeypatch.setenv("JWT_ALGORITHM", "RS256")
    monkeypatch.setenv("JWT_PRIVATE_KEY", "not-a-pem")
    monkeypatch.setattr(jwt_mod, "JWT_PRIVATE_KEY", "not-a-pem", raising=False)

    with pytest.raises(ValueError) as excinfo:
        jwt_mod.create_jwt_token({"sub": "alice"})

    assert "at least 32 characters" not in str(excinfo.value)
    assert "RS256" in str(excinfo.value)


def test_rs256_ignores_a_missing_hmac_secret(jwt_mod, monkeypatch, rsa_keys) -> None:
    """Signing asymmetrically must not require JWT_SECRET_KEY to be set."""
    private_pem, public_pem = rsa_keys
    monkeypatch.setenv("JWT_ALGORITHM", "RS256")
    monkeypatch.setenv("JWT_PRIVATE_KEY", private_pem)
    monkeypatch.setattr(jwt_mod, "JWT_PRIVATE_KEY", private_pem, raising=False)
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.setattr(jwt_mod, "JWT_SECRET_KEY", None, raising=False)

    assert len(jwt_mod.create_jwt_token({"sub": "alice"}).split(".")) == 3


def test_algorithm_argument_overrides_the_environment(jwt_mod, monkeypatch) -> None:
    monkeypatch.setenv("JWT_ALGORITHM", "RS256")
    token = jwt_mod.create_jwt_token({"sub": "alice"}, algorithm="HS256")
    header = json.loads(jwt_mod.base64url_decode(token.split(".")[0]).decode("utf-8"))
    assert header["alg"] == "HS256"


def test_algorithm_argument_is_case_insensitive(jwt_mod) -> None:
    token = jwt_mod.create_jwt_token({"sub": "alice"}, algorithm="hs256")
    header = json.loads(jwt_mod.base64url_decode(token.split(".")[0]).decode("utf-8"))
    assert header["alg"] == "HS256"
