"""Tests for the skill install endpoint validation and command building.

Tests the package name validators, OS filtering, command building,
and URL validation used by POST /api/v1/install.
"""

import pytest

from nanobot.channels.http_api import (
    InstallSpec,
    _build_argv,
    _os_matches,
    _validate_pkg,
    _validate_url,
)


# ── Package name validation ────────────────────────────────────────────────


class TestValidatePkg:
    def test_node_valid_scoped(self):
        assert _validate_pkg("node", "@steipete/oracle") is None

    def test_node_valid_simple(self):
        assert _validate_pkg("node", "clawhub") is None

    def test_node_valid_versioned(self):
        assert _validate_pkg("node", "@steipete/oracle@1.0.0") is None

    def test_node_rejects_shell_meta(self):
        assert _validate_pkg("node", "foo; rm -rf /") is not None

    def test_node_rejects_backtick(self):
        assert _validate_pkg("node", "foo`whoami`") is not None

    def test_go_valid_module(self):
        assert _validate_pkg("go", "github.com/steipete/eightctl/cmd/eightctl@latest") is None

    def test_go_rejects_no_slash(self):
        assert _validate_pkg("go", "eightctl@latest") is not None

    def test_go_rejects_shell_injection(self):
        assert _validate_pkg("go", "github.com/foo/bar;rm -rf /") is not None

    def test_uv_valid_simple(self):
        assert _validate_pkg("uv", "yfinance") is None

    def test_uv_valid_with_extras(self):
        assert _validate_pkg("uv", "nano-pdf[cli]") is None

    def test_uv_rejects_pipe(self):
        assert _validate_pkg("uv", "foo | cat /etc/passwd") is not None

    def test_apt_valid(self):
        assert _validate_pkg("apt", "jq") is None

    def test_apt_rejects_uppercase(self):
        assert _validate_pkg("apt", "JQ") is not None

    def test_empty_returns_error(self):
        assert _validate_pkg("node", "") is not None

    def test_whitespace_only_returns_error(self):
        assert _validate_pkg("node", "   ") is not None


# ── URL validation ─────────────────────────────────────────────────────────


class TestValidateUrl:
    def test_valid_https(self):
        assert _validate_url("https://github.com/releases/v1.0/tool.tar.gz") is None

    def test_valid_http(self):
        assert _validate_url("http://example.com/archive.zip") is None

    def test_rejects_missing_url(self):
        assert _validate_url(None) is not None

    def test_rejects_empty(self):
        assert _validate_url("") is not None

    def test_rejects_ftp(self):
        assert _validate_url("ftp://files.example.com/tool.tar.gz") is not None

    def test_rejects_localhost(self):
        assert _validate_url("http://localhost:8080/tool") is not None

    def test_rejects_127(self):
        assert _validate_url("http://127.0.0.1/tool") is not None

    def test_rejects_no_hostname(self):
        assert _validate_url("https:///path") is not None


# ── OS matching ────────────────────────────────────────────────────────────


class TestOsMatches:
    def test_empty_os_matches_everything(self):
        assert _os_matches([]) is True

    def test_linux_matches_on_linux(self):
        # In CI/Docker this test runs on Linux; locally it may differ
        # We test the logic, not the platform
        import platform
        if platform.system().lower() == "linux":
            assert _os_matches(["linux"]) is True
        elif platform.system().lower() == "darwin":
            assert _os_matches(["darwin"]) is True

    def test_empty_normalized_matches_everything(self):
        """If os list contains no recognized platforms, skip."""
        # Empty set after filtering = no platforms specified
        assert _os_matches([]) is True


# ── Command building ───────────────────────────────────────────────────────


class TestBuildArgv:
    def test_node_package(self):
        spec = InstallSpec(kind="node", package="clawhub")
        argv, err = _build_argv(spec)
        assert err is None
        assert argv == ["npm", "install", "-g", "clawhub"]

    def test_go_module(self):
        spec = InstallSpec(kind="go", module="github.com/steipete/eightctl/cmd/eightctl@latest")
        argv, err = _build_argv(spec)
        assert err is None
        assert argv == ["go", "install", "github.com/steipete/eightctl/cmd/eightctl@latest"]

    def test_uv_package(self):
        spec = InstallSpec(kind="uv", package="yfinance")
        argv, err = _build_argv(spec)
        assert err is None
        assert argv == ["uv", "tool", "install", "yfinance"]

    def test_apt_package(self):
        spec = InstallSpec(kind="apt", package="jq")
        argv, err = _build_argv(spec)
        assert err is None
        assert argv == ["apt-get", "install", "-y", "--no-install-recommends", "jq"]

    def test_brew_returns_error(self):
        spec = InstallSpec(kind="brew", formula="himalaya")
        argv, err = _build_argv(spec)
        assert argv is None
        assert "brew" in err.lower()

    def test_node_missing_package(self):
        spec = InstallSpec(kind="node")
        argv, err = _build_argv(spec)
        assert argv is None
        assert "missing" in err.lower()

    def test_go_missing_module(self):
        spec = InstallSpec(kind="go")
        argv, err = _build_argv(spec)
        assert argv is None
        assert "missing" in err.lower()

    def test_unknown_kind(self):
        spec = InstallSpec(kind="cargo")
        argv, err = _build_argv(spec)
        assert argv is None
        assert "unsupported" in err.lower()

    def test_injection_in_node_package(self):
        spec = InstallSpec(kind="node", package="foo; rm -rf /")
        argv, err = _build_argv(spec)
        assert argv is None
        assert err is not None

    def test_injection_in_go_module(self):
        spec = InstallSpec(kind="go", module="$(whoami)@latest")
        argv, err = _build_argv(spec)
        assert argv is None
        assert err is not None
