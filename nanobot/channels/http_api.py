"""HTTP API channel for programmatic access to nanobot."""

import asyncio
import platform
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from pydantic import BaseModel, Field

from nanobot.agent.loop import AgentLoop
from nanobot.config.schema import HTTPAPIConfig

# ── Chat models ──────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """POST /api/v1/chat request body."""
    message: str
    session_key: str = "api:default"
    channel: str = "api"
    chat_id: str = "default"
    extra_system_prompt: str | None = None
    extra_env: dict[str, str] | None = None


class ChatResponse(BaseModel):
    """POST /api/v1/chat response body."""
    response: str
    session_key: str


# ── Install models ───────────────────────────────────────────────────────────

_VALID_KINDS = frozenset({"node", "go", "uv", "apt", "download", "brew"})

# Allowlist regexes for package names — uses create_subprocess_exec (no shell)
_PKG_VALIDATORS: dict[str, re.Pattern[str]] = {
    "node": re.compile(r"^(@[a-z0-9\-_.]+/)?[a-z0-9\-_.]+(@[a-z0-9\-_.^~>=<]+)?$", re.I),
    "go": re.compile(r"^[a-zA-Z0-9\-_.]+(/[a-zA-Z0-9\-_.]+)+@[a-zA-Z0-9\-_.]+$"),
    "uv": re.compile(r"^[a-zA-Z0-9\-_.]+(\[.*\])?$"),
    "apt": re.compile(r"^[a-z0-9][a-z0-9.+\-]+$"),
}

# Characters that must never appear in any package name
_SHELL_META = set(";|&$`\\(){}!#\n\r\t")

_CONTAINER_OS = platform.system().lower()

# Serialize install operations (package managers aren't concurrent-safe)
_install_lock = asyncio.Lock()


class InstallSpec(BaseModel):
    """A single ClawhHub install specification."""
    id: str | None = None
    kind: str
    label: str | None = None
    bins: list[str] = Field(default_factory=list)
    os: list[str] = Field(default_factory=list)
    formula: str | None = None
    package: str | None = None
    module: str | None = None
    url: str | None = None
    archive: str | None = None
    extract: bool | None = None
    strip_components: int | None = Field(default=None, alias="stripComponents")
    target_dir: str | None = Field(default=None, alias="targetDir")

    model_config = {"populate_by_name": True}


class InstallRequest(BaseModel):
    """POST /api/v1/install request body."""
    skill_slug: str
    install_specs: list[InstallSpec]


class InstallResult(BaseModel):
    """Result for a single install spec."""
    spec_id: str
    ok: bool
    message: str
    kind: str


class InstallResponse(BaseModel):
    """POST /api/v1/install response body."""
    ok: bool
    results: list[InstallResult]
    message: str


def _os_matches(spec_os: list[str]) -> bool:
    """Check if the current OS matches the spec's os filter."""
    if not spec_os:
        return True
    normalized = {o.lower() for o in spec_os}
    if _CONTAINER_OS == "linux" and "linux" in normalized:
        return True
    if _CONTAINER_OS == "darwin" and ("darwin" in normalized or "macos" in normalized):
        return True
    # If the spec lists platforms and none match, skip it
    return not normalized


def _validate_pkg(kind: str, value: str) -> str | None:
    """Validate a package name. Returns error string or None if ok."""
    if not value or not value.strip():
        return "empty package name"
    if any(c in value for c in _SHELL_META):
        return f"forbidden characters in package name: {value!r}"
    pattern = _PKG_VALIDATORS.get(kind)
    if pattern and not pattern.match(value):
        return f"invalid {kind} package format: {value!r}"
    return None


def _validate_url(url: str | None) -> str | None:
    """Validate a download URL. Returns error string or None if ok."""
    if not url:
        return "missing URL"
    try:
        parsed = urlparse(url)
    except Exception:
        return f"invalid URL: {url!r}"
    if parsed.scheme not in ("http", "https"):
        return f"URL scheme must be http/https, got {parsed.scheme!r}"
    if not parsed.hostname:
        return "URL missing hostname"
    if parsed.hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return f"URL points to localhost: {url!r}"
    return None


def _build_argv(spec: InstallSpec) -> tuple[list[str] | None, str | None]:
    """Build command argv list from an install spec. Returns (argv, error).

    Uses create_subprocess_exec (arg list, no shell) for safety.
    """
    kind = spec.kind.lower()

    if kind == "brew":
        return None, "brew not supported (macOS only)"

    if kind == "node":
        if not spec.package:
            return None, "missing node package name"
        err = _validate_pkg("node", spec.package)
        if err:
            return None, err
        return ["npm", "install", "-g", spec.package], None

    if kind == "go":
        if not spec.module:
            return None, "missing go module path"
        err = _validate_pkg("go", spec.module)
        if err:
            return None, err
        return ["go", "install", spec.module], None

    if kind == "uv":
        if not spec.package:
            return None, "missing uv package name"
        err = _validate_pkg("uv", spec.package)
        if err:
            return None, err
        return ["uv", "tool", "install", spec.package], None

    if kind == "apt":
        if not spec.package:
            return None, "missing apt package name"
        err = _validate_pkg("apt", spec.package)
        if err:
            return None, err
        return ["apt-get", "install", "-y", "--no-install-recommends", spec.package], None

    return None, f"unsupported install kind: {kind!r}"


async def _run_install(argv: list[str], timeout: float = 300) -> tuple[bool, str]:
    """Run an install command via create_subprocess_exec (no shell). Returns (ok, message)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode == 0:
            return True, f"Installed via {argv[0]}"
        else:
            err_text = stderr.decode("utf-8", errors="replace").strip()
            return False, f"exit {proc.returncode}: {err_text[:500]}"
    except asyncio.TimeoutError:
        return False, f"timed out after {timeout}s"
    except FileNotFoundError:
        return False, f"command not found: {argv[0]!r}"


_EXECUTABLE_SUFFIXES = frozenset({".py", ".sh", ".js", ".mjs", ""})
_SKILL_TOOLS_DIR = Path("/root/.nanobot/skill-tools")
_SKILL_BIN_DIR = Path("/usr/local/bin")


async def _handle_download(spec: InstallSpec) -> InstallResult:
    """Handle a 'download' install spec using httpx + tarfile (no shell)."""
    import tarfile
    import zipfile

    import httpx

    spec_id = spec.id or "download"
    err = _validate_url(spec.url)
    if err:
        return InstallResult(spec_id=spec_id, ok=False, message=err, kind="download")

    install_dir = _SKILL_TOOLS_DIR / spec_id
    install_dir.mkdir(parents=True, exist_ok=True)

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
            resp = await client.get(spec.url)  # type: ignore[arg-type]
            resp.raise_for_status()

        filename = Path(urlparse(spec.url or "").path).name or "download"
        filepath = install_dir / filename
        filepath.write_bytes(resp.content)

        should_extract = spec.extract if spec.extract is not None else (
            filename.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".zip"))
        )

        if should_extract:
            if filename.endswith(".zip"):
                with zipfile.ZipFile(filepath) as zf:
                    zf.extractall(install_dir)
            elif filename.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
                with tarfile.open(filepath) as tf:
                    tf.extractall(install_dir)
            filepath.unlink(missing_ok=True)
        else:
            # Make script files executable
            if filepath.suffix in _EXECUTABLE_SUFFIXES:
                filepath.chmod(0o755)

        logger.info(f"  Downloaded {filename} -> {install_dir}")
        return InstallResult(
            spec_id=spec_id, ok=True,
            message=f"Downloaded to {install_dir}", kind="download",
        )
    except Exception as e:
        logger.error(f"  Download failed for {spec_id}: {e}")
        return InstallResult(spec_id=spec_id, ok=False, message=str(e), kind="download")


def _symlink_skill_bins(skill_slug: str, all_specs: list[InstallSpec]) -> None:
    """After all downloads for a skill, symlink declared bins to /usr/local/bin."""
    # Collect all declared bins from all specs
    bins: set[str] = set()
    for spec in all_specs:
        bins.update(spec.bins)

    if not bins:
        return

    # Search for matching files in the skill-tools directory
    skill_dir = _SKILL_TOOLS_DIR
    for bin_name in bins:
        # Look for the bin in any subdirectory of skill-tools
        for candidate in skill_dir.rglob(bin_name):
            if candidate.is_file():
                link_path = _SKILL_BIN_DIR / bin_name
                link_path.unlink(missing_ok=True)
                link_path.symlink_to(candidate)
                candidate.chmod(0o755)
                logger.info(f"Symlinked {bin_name} -> {candidate}")
                break
        # Also check for bin_name.py, bin_name.sh variants
        for ext in (".py", ".sh"):
            for candidate in skill_dir.rglob(f"{bin_name}{ext}"):
                if candidate.is_file():
                    link_path = _SKILL_BIN_DIR / bin_name
                    link_path.unlink(missing_ok=True)
                    link_path.symlink_to(candidate)
                    candidate.chmod(0o755)
                    logger.info(f"Symlinked {bin_name} -> {candidate}")
                    break


class HTTPAPIChannel:
    """HTTP API channel — exposes nanobot as a REST endpoint.

    Follows the same pattern as other nanobot channels (Telegram, Discord, etc.)
    but uses HTTP request/response instead of a message bus.
    """

    def __init__(self, config: HTTPAPIConfig, agent_loop: AgentLoop) -> None:
        self.config = config
        self.agent = agent_loop
        self._app: Any = None  # lazily created

    def _create_app(self) -> Any:
        """Create the FastAPI application with routes."""
        from fastapi import Depends, FastAPI, Header, HTTPException

        app = FastAPI(title="nanobot API", version="1.0.0")

        def _verify_token(authorization: str = Header(default="")) -> None:
            """FastAPI dependency that verifies bearer token."""
            if not self.config.auth_token:
                return  # No auth required

            if not authorization:
                raise HTTPException(status_code=401, detail="Authorization header required")

            parts = authorization.split(" ", 1)
            if len(parts) != 2 or parts[0].lower() != "bearer":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authorization format (expected 'Bearer <token>')",
                )

            if parts[1] != self.config.auth_token:
                raise HTTPException(status_code=403, detail="Invalid token")

        @app.get("/api/v1/health")
        async def health() -> dict:
            return {"status": "ok"}

        @app.post("/api/v1/chat", response_model=ChatResponse)
        async def chat(req: ChatRequest, _: None = Depends(_verify_token)) -> ChatResponse:
            response = await self.agent.process_direct(
                content=req.message,
                session_key=req.session_key,
                channel=req.channel,
                chat_id=req.chat_id,
                extra_system_prompt=req.extra_system_prompt,
                extra_env=req.extra_env,
            )
            return ChatResponse(response=response, session_key=req.session_key)

        @app.post("/api/v1/install", response_model=InstallResponse)
        async def install_skill_deps(
            req: InstallRequest, _: None = Depends(_verify_token),
        ) -> InstallResponse:
            """Install dependencies for a ClawhHub skill."""
            logger.info(f"Install request for skill '{req.skill_slug}' ({len(req.install_specs)} specs)")

            async with _install_lock:
                results: list[InstallResult] = []

                for spec in req.install_specs:
                    spec_id = spec.id or f"{spec.kind}-{len(results)}"

                    # OS filter
                    if not _os_matches(spec.os):
                        results.append(InstallResult(
                            spec_id=spec_id, ok=True,
                            message=f"Skipped (OS filter: {spec.os})",
                            kind=spec.kind,
                        ))
                        continue

                    # Skip brew
                    if spec.kind.lower() == "brew":
                        results.append(InstallResult(
                            spec_id=spec_id, ok=True,
                            message="Skipped (brew not supported on Linux)",
                            kind="brew",
                        ))
                        continue

                    # Downloads handled via Python (no subprocess)
                    if spec.kind.lower() == "download":
                        results.append(await _handle_download(spec))
                        continue

                    # Build command for node/go/uv/apt
                    argv, err = _build_argv(spec)
                    if err:
                        results.append(InstallResult(
                            spec_id=spec_id, ok=False, message=err, kind=spec.kind,
                        ))
                        continue

                    # For apt, refresh package lists first
                    if spec.kind.lower() == "apt":
                        await _run_install(["apt-get", "update"], timeout=120)

                    logger.info(f"  Running: {' '.join(argv)}")
                    ok, msg = await _run_install(argv)
                    results.append(InstallResult(
                        spec_id=spec_id, ok=ok, message=msg, kind=spec.kind,
                    ))

                # Symlink declared bins after all downloads complete
                has_downloads = any(
                    r.kind == "download" and r.ok for r in results
                )
                if has_downloads:
                    try:
                        _symlink_skill_bins(req.skill_slug, req.install_specs)
                    except Exception as e:
                        logger.warning(f"Failed to symlink bins: {e}")

                all_ok = all(r.ok for r in results)
                return InstallResponse(
                    ok=all_ok,
                    results=results,
                    message="All dependencies installed" if all_ok else "Some installations failed",
                )

        return app

    async def start(self, host: str, port: int) -> None:
        """Start the HTTP API server."""
        import uvicorn

        self._app = self._create_app()
        logger.info(f"HTTP API starting on {host}:{port}")
        config = uvicorn.Config(self._app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
