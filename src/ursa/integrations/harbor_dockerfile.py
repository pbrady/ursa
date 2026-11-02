"""Translate a Dockerfile into an Apptainer definition file.

Rootless Buildah and Podman cannot run ``RUN`` steps for users without
``/etc/subuid`` entries, but ``apptainer build --fakeroot`` can, so the
Singularity environment builds such tasks from a generated definition.

The definition itself stays fixed: each stage bootstraps from the stage's
base image, copies a staging directory into the root filesystem in
``%setup``, and runs a generated ``build.sh`` from ``%post``. The script
replays the Dockerfile in order, so ``COPY`` and ``RUN`` steps stay
interleaved even though Apptainer runs ``%files`` before ``%post``. Keeping
Dockerfile text out of the definition also avoids Apptainer's ``{{ name }}``
templating and its ``%section`` markers, which have no escape syntax.
"""

from __future__ import annotations

import fnmatch
import json
import os
import platform
import posixpath
import re
import shlex
import shutil
import ssl
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from http.client import HTTPException
from pathlib import Path
from typing import Any, NoReturn

# Part of the SIF cache key for definition builds; bump it whenever the
# generated definition or build script changes.
DOCKERFILE_TRANSLATOR_VERSION = 3

_STAGING = "/.ursa-build"
_DEFAULT_SHELL = ("/bin/sh", "-c")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FLAG = re.compile(r"--([A-Za-z][A-Za-z0-9-]*)(?:=(\S*))?(?:\s+|\Z)")
_DIRECTIVE = re.compile(r"#\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(.*?)\s*")
_HEREDOC = re.compile(
    r"<<(-?)(?:\"([^\"\s]+)\"|'([^'\s]+)'|([A-Za-z_][A-Za-z0-9_.-]*))"
)
_CHMOD = re.compile(
    r"[0-7]{3,4}|[ugoa]*[-+=][rwxXst]*(,[ugoa]*[-+=][rwxXst]*)*"
)
_GLOB_CHARACTERS = frozenset("*?[")
_PROXY_ARGS = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "FTP_PROXY",
    "ftp_proxy",
    "NO_PROXY",
    "no_proxy",
    "ALL_PROXY",
    "all_proxy",
)
_IGNORED_INSTRUCTIONS = frozenset({
    "EXPOSE",
    "HEALTHCHECK",
    "LABEL",
    "MAINTAINER",
    "STOPSIGNAL",
    "VOLUME",
})
_ARCHITECTURES = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
}


class DockerfileUnsupported(ValueError):
    """A Dockerfile that the Apptainer translation cannot reproduce."""


@dataclass(frozen=True)
class Heredoc:
    name: str
    # The content, with leading tabs removed for ``<<-``.
    body: str
    # The lines as written, including the terminator.
    raw: str


@dataclass(frozen=True)
class Instruction:
    line: int
    keyword: str
    args: str
    heredocs: tuple[Heredoc, ...] = ()

    @property
    def summary(self) -> str:
        text = f"{self.keyword} {self.args}".strip()
        return text if len(text) <= 160 else f"{text[:157]}..."


@dataclass(frozen=True)
class ParsedDockerfile:
    escape: str
    instructions: tuple[Instruction, ...]


def _fail(instruction: Instruction, reason: str) -> NoReturn:
    raise DockerfileUnsupported(
        f"line {instruction.line}: {reason} ({instruction.summary})"
    )


def _heredoc_markers(text: str, escape: str) -> list[tuple[str, bool]]:
    """Find ``<<NAME`` markers outside quotes as (name, strip_tabs)."""
    markers = []
    quote = None
    index = 0
    while index < len(text):
        character = text[index]
        if quote is not None:
            if character == quote:
                quote = None
            elif character == escape and quote == '"':
                index += 1
        elif character in "'\"":
            quote = character
        elif character == escape:
            index += 1
        elif (
            text.startswith("<<", index)
            and not text.startswith("<<<", index)
            and (index == 0 or text[index - 1].isspace())
        ):
            match = _HEREDOC.match(text, index)
            if match is not None:
                name = match[2] or match[3] or match[4]
                markers.append((name, match[1] == "-"))
                index = match.end()
                continue
        index += 1
    return markers


def parse_dockerfile(text: str) -> ParsedDockerfile:
    """Split a Dockerfile into instructions as BuildKit does.

    This handles parser directives, line continuations with the escape
    character, comment and blank lines inside continuations, and heredocs
    for ``RUN``, ``COPY``, and ``ADD``.
    """
    lines = text.splitlines()
    escape = "\\"
    seen: set[str] = set()
    for line in lines:
        match = _DIRECTIVE.fullmatch(line.strip())
        if match is None:
            break
        key = match[1].lower()
        if key not in {"syntax", "escape", "check"} or key in seen:
            break
        seen.add(key)
        if key == "escape":
            if match[2] not in {"\\", "`"}:
                raise DockerfileUnsupported(
                    f"invalid escape directive {match[2]!r}"
                )
            escape = match[2]
    continuation = re.compile(re.escape(escape) + r"[ \t]*\Z")

    def trim(line: str) -> tuple[str, bool]:
        match = continuation.search(line)
        if match is None:
            return line, True
        return line[: match.start()], False

    instructions = []
    index = 0
    while index < len(lines):
        stripped = lines[index].lstrip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        start = index
        logical, complete = trim(stripped)
        while not complete and index < len(lines):
            line = lines[index]
            index += 1
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            part, complete = trim(line)
            logical += part
        match = re.match(r"(\S+)\s*(.*)", logical.strip(), re.DOTALL)
        keyword, args = match[1].upper(), match[2].strip()
        heredocs = []
        if keyword in {"RUN", "COPY", "ADD"}:
            # The shell expands unquoted heredocs when the step runs.
            for name, strip_tabs in _heredoc_markers(args, escape):
                body: list[str] = []
                raw: list[str] = []
                while True:
                    if index >= len(lines):
                        raise DockerfileUnsupported(
                            f"line {start}: heredoc {name} is not terminated"
                        )
                    line = lines[index]
                    index += 1
                    raw.append(f"{line}\n")
                    content = line.lstrip("\t") if strip_tabs else line
                    if content == name:
                        break
                    body.append(f"{content}\n")
                heredocs.append(Heredoc(name, "".join(body), "".join(raw)))
        instructions.append(Instruction(start, keyword, args, tuple(heredocs)))
    return ParsedDockerfile(escape, tuple(instructions))


@dataclass(frozen=True)
class _Ref:
    """A variable reference, with an optional ``${name<op>word}`` operator."""

    name: str
    op: str = ""
    word: tuple[str | _Ref, ...] = ()


_Value = tuple[str | _Ref, ...]


class _LexError(Exception):
    pass


class _Lexer:
    """BuildKit's shell-word processing: quotes, escapes, and variables."""

    def __init__(self, text: str, escape: str) -> None:
        self.text = text
        self.escape = escape
        self.position = 0

    def word(self) -> _Value:
        return self._parse(stop=None, split=False)[0]

    def words(self) -> list[_Value]:
        return self._parse(stop=None, split=True)

    def _parse(self, stop: str | None, split: bool) -> list[_Value]:
        words: list[_Value] = []
        current: list[str | _Ref] = []
        started = False
        text = self.text
        while self.position < len(text):
            character = text[self.position]
            if stop is not None and character == stop:
                break
            if split and character in " \t\r\n":
                if started:
                    words.append(_merge(current))
                    current, started = [], False
                self.position += 1
                continue
            started = True
            if character == self.escape:
                self.position += 1
                if self.position < len(text):
                    current.append(text[self.position])
                    self.position += 1
                else:
                    current.append(character)
            elif character == "'":
                end = text.find("'", self.position + 1)
                if end < 0:
                    raise _LexError("unterminated single quote")
                current.append(text[self.position + 1 : end])
                self.position = end + 1
            elif character == '"':
                self._double_quoted(current)
            elif character == "$":
                current.append(self._dollar())
            else:
                current.append(character)
                self.position += 1
        if started or not split:
            words.append(_merge(current))
        return words

    def _double_quoted(self, current: list[str | _Ref]) -> None:
        text = self.text
        self.position += 1
        while self.position < len(text):
            character = text[self.position]
            if character == '"':
                self.position += 1
                return
            if character == self.escape and self.position + 1 < len(text):
                following = text[self.position + 1]
                if following in {'"', "$", self.escape}:
                    current.append(following)
                    self.position += 2
                    continue
            if character == "$":
                current.append(self._dollar())
                continue
            current.append(character)
            self.position += 1
        raise _LexError("unterminated double quote")

    def _dollar(self) -> str | _Ref:
        text = self.text
        self.position += 1
        if text.startswith("{", self.position):
            self.position += 1
            match = _NAME.match(text, self.position)
            if match is None:
                raise _LexError("${...} needs a variable name")
            name = match[0]
            self.position = match.end()
            if text.startswith("}", self.position):
                self.position += 1
                return _Ref(name)
            for op in (":-", ":+", ":?", "-", "+", "?"):
                if text.startswith(op, self.position):
                    break
            else:
                raise _LexError(
                    f"variable modifier in ${{{name}"
                    f"{text[self.position : self.position + 1]}...}} "
                    "is not supported"
                )
            self.position += len(op)
            word = self._parse(stop="}", split=False)[0]
            if not text.startswith("}", self.position):
                raise _LexError(f"unterminated ${{{name}...}}")
            self.position += 1
            return _Ref(name, op, word)
        match = _NAME.match(text, self.position)
        if match is None:
            return "$"
        self.position = match.end()
        return _Ref(match[0])


def _merge(parts: Sequence[str | _Ref]) -> _Value:
    merged: list[str | _Ref] = []
    for part in parts:
        if isinstance(part, str):
            if not part:
                continue
            if merged and isinstance(merged[-1], str):
                merged[-1] += part
                continue
        merged.append(part)
    return tuple(merged)


def _raw_words(text: str, escape: str) -> list[str]:
    """Split on unquoted whitespace, keeping quotes and escapes as written."""
    words = []
    current = ""
    quote = None
    index = 0
    while index < len(text):
        character = text[index]
        if quote is None and character in " \t\r\n":
            if current:
                words.append(current)
                current = ""
            index += 1
            continue
        if character == escape and quote != "'" and index + 1 < len(text):
            current += text[index : index + 2]
            index += 2
            continue
        if quote is None and character in "'\"":
            quote = character
        elif character == quote:
            quote = None
        current += character
        index += 1
    if current:
        words.append(current)
    return words


# A known variable: a value, or ``None`` for an ``ARG`` without one.
_Scope = dict[str, _Value | None]


def _is_static(value: _Value) -> bool:
    return all(isinstance(part, str) for part in value)


def _text(value: _Value) -> str:
    return "".join(part for part in value if isinstance(part, str))


def _resolve(value: _Value, scope: _Scope, *, defer: bool) -> _Value:
    """Substitute known variables.

    Unknown variables, and known ones whose value is only computed inside
    the image, are kept as references when ``defer`` is true, so the build
    script evaluates them against the base image's environment, as Docker
    does. Otherwise they are unset, as in the scope before the first
    ``FROM``.
    """
    resolved: list[str | _Ref] = []
    for part in value:
        if isinstance(part, str):
            resolved.append(part)
            continue
        word = _resolve(part.word, scope, defer=defer)
        if part.name in scope:
            known = scope[part.name]
            if known is not None and not _is_static(known):
                resolved.append(_Ref(part.name, part.op, word))
                continue
            current = None if known is None else _text(known)
        elif defer:
            resolved.append(_Ref(part.name, part.op, word))
            continue
        else:
            current = None
        present = current is not None and (
            current != "" or not part.op.startswith(":")
        )
        if part.op == "":
            resolved.append(current or "")
        elif part.op in {"-", ":-"}:
            resolved.extend((current,) if present else word)
        elif part.op in {"+", ":+"}:
            resolved.extend(word if present else ())
        elif present:
            resolved.append(current)
        else:
            message = _text(word) or "parameter not set"
            raise _LexError(f"{part.name}: {message}")
    return _merge(resolved)


def _shell_reference(reference: _Ref) -> str:
    word = []
    for part in reference.word:
        if isinstance(part, _Ref):
            word.append(_shell_reference(part))
        elif "}" in part:
            raise _LexError(f"'}}' in the default of ${{{reference.name}}}")
        else:
            word.append(re.sub(r'(["\\$`])', r"\\\1", part))
    return "${" + reference.name + reference.op + "".join(word) + "}"


def _shell_word(value: _Value) -> str:
    """Render a value as one shell word that expands deferred variables."""
    if not value:
        return "''"
    return "".join(
        shlex.quote(part)
        if isinstance(part, str)
        else f'"{_shell_reference(part)}"'
        for part in value
    )


def _json_array(text: str) -> list[str] | None:
    if not text.lstrip().startswith("["):
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        return None
    return value


def _split_flags(args: str) -> tuple[dict[str, list[str]], str]:
    flags: dict[str, list[str]] = {}
    position = 0
    while (match := _FLAG.match(args, position)) is not None:
        flags.setdefault(match[1].lower(), []).append(match[2] or "")
        position = match.end()
    return flags, args[position:]


def _platform_args(machine: str) -> dict[str, str]:
    architecture = _ARCHITECTURES.get(machine.lower(), machine.lower())
    target = f"linux/{architecture}"
    return {
        "BUILDARCH": architecture,
        "BUILDOS": "linux",
        "BUILDPLATFORM": target,
        "BUILDVARIANT": "",
        "TARGETARCH": architecture,
        "TARGETOS": "linux",
        "TARGETPLATFORM": target,
        "TARGETVARIANT": "",
    }


def _parse_build_args(build_args: Sequence[str]) -> dict[str, str]:
    values = {}
    for build_arg in build_args:
        key, _, value = build_arg.partition("=")
        values[key] = value
    return values


_MANIFEST_TYPES = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])
_CHALLENGE = re.compile(r'(\w+)="([^"]*)"')
_image_configs: dict[tuple[str, str], Mapping[str, Any]] = {}
# Looks up an image's OCI config for a machine; see ``image_config``.
ImageConfig = Callable[[str, str], Mapping[str, Any] | None]


class _RegistryRedirect(urllib.request.HTTPRedirectHandler):
    """Drop the registry token when a blob redirects to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        host = urllib.parse.urlsplit(req.full_url).netloc
        if new is not None and urllib.parse.urlsplit(newurl).netloc != host:
            new.remove_header("Authorization")
        return new


def _image_reference(image: str) -> tuple[str, str, str]:
    """Split an image into its registry host, repository, and tag or digest."""
    name, _, digest = image.partition("@")
    tag = "latest"
    if ":" in name.rsplit("/", 1)[-1]:
        name, tag = name.rsplit(":", 1)
    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        registry, repository = first, rest
    else:
        registry, repository = "docker.io", name
    if registry in {"docker.io", "index.docker.io"}:
        registry = "registry-1.docker.io"
        if "/" not in repository:
            repository = f"library/{repository}"
    return registry, repository, digest or tag


def _registry_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    accept: str,
    auth: dict[str, str],
) -> Any:
    """GET a registry document, fetching an anonymous token on a 401."""

    def fetch() -> Any:
        request = urllib.request.Request(url, headers={"Accept": accept})
        if "token" in auth:
            request.add_header("Authorization", f"Bearer {auth['token']}")
        with opener.open(request, timeout=60) as response:
            return json.loads(response.read(1 << 22))

    try:
        return fetch()
    except urllib.error.HTTPError as exc:
        challenge = exc.headers.get("WWW-Authenticate", "")
        if exc.code != 401 or "token" in auth:
            raise
        if not challenge.lower().startswith("bearer "):
            raise
    params = dict(_CHALLENGE.findall(challenge))
    query = {key: params[key] for key in ("service", "scope") if key in params}
    token_url = f"{params['realm']}?{urllib.parse.urlencode(query)}"
    with opener.open(token_url, timeout=60) as response:
        body = json.loads(response.read(1 << 20))
    auth["token"] = body.get("token") or body["access_token"]
    return fetch()


def image_config(image: str, machine: str) -> Mapping[str, Any] | None:
    """Return the OCI config of ``image`` for ``machine``, or ``None``.

    Apptainer keeps only the environment of a ``docker`` bootstrap, so the
    base image's ``WorkingDir`` and ``User`` come from the registry. Only
    anonymous access is tried; ``None`` means the config could not be read.
    """
    key = (image, machine)
    if key in _image_configs:
        return _image_configs[key]
    registry, repository, reference = _image_reference(image)
    base = f"https://{registry}/v2/{repository}"
    architecture = _ARCHITECTURES.get(machine.lower(), machine.lower())
    context = ssl.create_default_context()
    # Python 3.13's strict mode rejects some site proxy CAs that Apptainer
    # accepts; the chain is still verified.
    context.verify_flags &= ~getattr(ssl, "VERIFY_X509_STRICT", 0)
    opener = urllib.request.build_opener(
        _RegistryRedirect, urllib.request.HTTPSHandler(context=context)
    )
    auth: dict[str, str] = {}
    try:
        manifest = _registry_json(
            opener, f"{base}/manifests/{reference}", _MANIFEST_TYPES, auth
        )
        if "manifests" in manifest:
            digests = [
                entry["digest"]
                for entry in manifest["manifests"]
                if entry.get("platform", {}).get("os") == "linux"
                and entry["platform"].get("architecture") == architecture
            ]
            if not digests:
                return None
            manifest = _registry_json(
                opener, f"{base}/manifests/{digests[0]}", _MANIFEST_TYPES, auth
            )
        blob = _registry_json(
            opener,
            f"{base}/blobs/{manifest['config']['digest']}",
            "application/json",
            auth,
        )
        config = blob.get("config") or {}
    except (
        AttributeError,
        HTTPException,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return None
    _image_configs[key] = config
    return config


class DockerIgnore:
    """``.dockerignore`` patterns, matched as Docker's patternmatcher does.

    Unlike ``.gitignore`` patterns, each one is anchored at the context
    root, so ``*.json`` leaves ``src/config.json`` in; a pattern that
    matches a directory also matches everything under it, and ``!``
    exceptions apply in order.
    """

    def __init__(self, lines: Sequence[str]) -> None:
        self.patterns: list[tuple[bool, re.Pattern[str]]] = []
        for line in lines:
            if line.startswith("#"):
                continue
            pattern = line.strip()
            exclusion = pattern.startswith("!")
            if exclusion:
                pattern = pattern[1:].strip()
            if not pattern:
                continue
            # Go's filepath.Clean, then relative to the context root.
            pattern = posixpath.normpath(pattern)
            pattern = pattern.lstrip("/") or pattern[:1]
            with suppress(re.error):
                self.patterns.append((exclusion, _ignore_regex(pattern)))
        self.has_exceptions = any(exclusion for exclusion, _ in self.patterns)

    def ignored(self, relative: str) -> bool:
        parents = relative.split("/")[:-1]
        matched = False
        for exclusion, regex in self.patterns:
            # An exception only matters once a path is ignored, and an
            # ignore pattern only until then.
            if exclusion != matched:
                continue
            if regex.fullmatch(relative) or any(
                regex.fullmatch("/".join(parents[: index + 1]))
                for index in range(len(parents))
            ):
                matched = not exclusion
        return matched


def _ignore_regex(pattern: str) -> re.Pattern[str]:
    """Compile a pattern as moby/patternmatcher does."""
    regex = []
    index = 0
    while index < len(pattern):
        character = pattern[index]
        index += 1
        if character == "*" and pattern.startswith("*", index):
            index += 1
            if pattern.startswith("/", index):
                index += 1
            regex.append(".*" if index == len(pattern) else "(.*/)?")
        elif character == "*":
            regex.append("[^/]*")
        elif character == "?":
            regex.append("[^/]")
        elif character == "\\" and index < len(pattern):
            regex.append(re.escape(pattern[index]))
            index += 1
        elif character in ".+()|{}$\\":
            regex.append(re.escape(character))
        else:
            regex.append(character)
    return re.compile("".join(regex))


def dockerignore(context_dir: Path) -> DockerIgnore | None:
    path = context_dir / ".dockerignore"
    if not path.is_file():
        return None
    return DockerIgnore(path.read_text().splitlines())


@dataclass(frozen=True)
class _Stage:
    index: int
    name: str | None
    image: str
    instruction: Instruction
    instructions: tuple[Instruction, ...]


def _split_stages(
    parsed: ParsedDockerfile,
    build_args: Mapping[str, str],
    machine: str,
    *,
    strict: bool,
) -> tuple[_Scope, list[_Stage]]:
    """Resolve the global ``ARG`` scope and each stage's ``FROM`` line."""
    scope: _Scope = {
        name: (value,) for name, value in _platform_args(machine).items()
    }
    stages: list[_Stage] = []
    pending: list[Instruction] = []
    current: tuple[Instruction, str | None, str] | None = None
    escape = parsed.escape

    def close() -> None:
        if current is not None:
            instruction, name, image = current
            stages.append(
                _Stage(len(stages), name, image, instruction, tuple(pending))
            )

    for instruction in parsed.instructions:
        if instruction.keyword != "FROM":
            if current is not None:
                pending.append(instruction)
            elif instruction.keyword == "ARG":
                _global_arg(instruction, scope, build_args, escape)
            elif strict:
                _fail(instruction, "only ARG may come before the first FROM")
            continue
        close()
        pending = []
        flags, rest = _split_flags(instruction.args)
        try:
            words = [
                _text(_resolve(word, scope, defer=False))
                for word in _Lexer(rest, escape).words()
            ]
        except _LexError as exc:
            if strict:
                _fail(instruction, str(exc))
            words = [rest]
        if len(words) == 3 and words[1].lower() == "as":
            image, name = words[0], words[2].lower()
        elif len(words) == 1:
            image, name = words[0], None
        elif strict:
            _fail(instruction, "expected FROM <image> [AS <name>]")
        else:
            image, name = (words or [""])[0], None
        if strict:
            _check_platform(instruction, flags, scope, machine, escape)
            unknown = set(flags) - {"platform"}
            if unknown:
                _fail(instruction, f"FROM --{sorted(unknown)[0]} is unknown")
        current = (instruction, name, image)
    close()
    return scope, stages


def _global_arg(
    instruction: Instruction,
    scope: _Scope,
    build_args: Mapping[str, str],
    escape: str,
) -> None:
    for word in _raw_words(instruction.args, escape):
        name, separator, default = word.partition("=")
        if name in build_args:
            scope[name] = (build_args[name],)
        elif separator:
            try:
                value = _Lexer(default, escape).word()
                scope[name] = _resolve(value, scope, defer=False)
            except _LexError as exc:
                _fail(instruction, str(exc))
        elif name not in scope:
            scope[name] = None


def _check_platform(
    instruction: Instruction,
    flags: Mapping[str, list[str]],
    scope: _Scope,
    machine: str,
    escape: str,
) -> None:
    for value in flags.get("platform", []):
        try:
            text = _text(
                _resolve(_Lexer(value, escape).word(), scope, defer=False)
            )
        except _LexError as exc:
            _fail(instruction, str(exc))
        parts = text.split("/")
        host = _platform_args(machine)["TARGETARCH"]
        architecture = (
            _ARCHITECTURES.get(parts[1], parts[1]) if len(parts) > 1 else ""
        )
        if parts[0] != "linux" or architecture != host:
            _fail(
                instruction,
                f"--platform={text} does not match this host (linux/{host})",
            )


def _base_scope(config: Mapping[str, Any]) -> _Scope:
    """Return a base image's environment, which Docker expands against."""
    scope: _Scope = {}
    for entry in config.get("Env") or []:
        name, separator, value = str(entry).partition("=")
        if separator:
            scope[name] = (value,)
    return scope


def _stage_workdir(
    stage: _Stage,
    escape: str,
    workdir: str,
    scope: _Scope,
    arguments: Mapping[str, str],
    global_scope: _Scope,
    *,
    defer: bool,
) -> tuple[str, bool]:
    """Return a stage's final ``WORKDIR``, as the translation sets it.

    The flag is true when the result depends on the starting directory,
    which is the base image's ``WorkingDir``.
    """
    env_names: set[str] = set()
    relative = absolute = False
    for instruction in stage.instructions:
        words = _raw_words(instruction.args, escape)
        if instruction.keyword == "ARG":
            for word in words:
                name, separator, default = word.partition("=")
                if name in env_names:
                    continue
                if name in arguments:
                    value: _Value | None = (arguments[name],)
                elif separator:
                    try:
                        value = _resolve(
                            _Lexer(default, escape).word(), scope, defer=defer
                        )
                    except _LexError:
                        value = (_Ref(name),)
                else:
                    value = global_scope.get(name)
                if value is not None:
                    scope[name] = value
        elif instruction.keyword == "ENV":
            for name, raw in _env_pairs(instruction, escape):
                env_names.add(name)
                try:
                    scope[name] = _resolve(
                        _Lexer(raw, escape).word(), scope, defer=defer
                    )
                except _LexError:
                    scope[name] = (_Ref(name),)
        elif instruction.keyword == "WORKDIR":
            try:
                value = _resolve(
                    _Lexer(instruction.args, escape).word(), scope, defer=defer
                )
            except _LexError:
                value = (_Ref("?"),)
            if not _is_static(value):
                raise ValueError(
                    "Singularity cannot resolve variables in Dockerfile "
                    f"WORKDIR: {instruction.args}"
                )
            path = _text(value)
            relative = relative or not (absolute or path.startswith("/"))
            absolute = absolute or path.startswith("/")
            workdir = posixpath.normpath(posixpath.join(workdir, path))
    return workdir, relative


def dockerfile_workdir(
    text: str,
    build_args: Sequence[str] = (),
    target: str | None = None,
    machine: str | None = None,
    *,
    image_config: ImageConfig | None = None,
    warnings: list[str] | None = None,
) -> str:
    """Return the final ``WORKDIR`` of a stage without building it.

    Variables defined by the Dockerfile and its build arguments are
    resolved. ``image_config`` looks up the base image's config only when
    the result depends on it: a relative first ``WORKDIR`` starts in its
    ``WorkingDir``, and variables can come from its environment. Anything
    still unresolved raises ``ValueError``; a relative ``WORKDIR`` whose
    base config cannot be read starts in ``/``, as the translation does,
    and adds to ``warnings``. Constructs that only a build would reject are
    ignored here.
    """
    parsed = parse_dockerfile(text)
    arguments = _parse_build_args(build_args)
    machine = machine or platform.machine()
    global_scope, stages = _split_stages(
        parsed, arguments, machine, strict=False
    )
    if not stages:
        return "/"
    parents: dict[int, _Stage | None] = {}
    names: dict[str, _Stage] = {}
    for stage in stages:
        parents[stage.index] = names.get(stage.image.lower())
        if stage.name is not None:
            names[stage.name] = stage
    if target is None:
        final = stages[-1]
    elif target.lower() in names:
        final = names[target.lower()]
    else:
        return "/"

    def workdir(stage: _Stage) -> str:
        parent = parents[stage.index]
        common = (stage, parsed.escape)
        inputs = (arguments, global_scope)
        if parent is not None:
            return _stage_workdir(
                *common, workdir(parent), {}, *inputs, defer=True
            )[0]
        try:
            result, relative = _stage_workdir(
                *common, "/", {}, *inputs, defer=True
            )
        except ValueError as exc:
            # Variables from the base image's environment.
            result, relative, error = "/", True, exc
        else:
            error = None
        config = (
            image_config(stage.image, machine)
            if relative and image_config is not None
            else None
        )
        if config is None:
            if error is not None:
                raise error
            if relative and image_config is not None and warnings is not None:
                warnings.append(
                    f"cannot read the config of {stage.image}; its relative "
                    "WORKDIR starts in / instead of the image's WORKDIR"
                )
            return result
        start = config.get("WorkingDir") or "/"
        return _stage_workdir(
            *common, start, _base_scope(config), *inputs, defer=False
        )[0]

    return workdir(final)


def _env_pairs(instruction: Instruction, escape: str) -> list[tuple[str, str]]:
    """Split ``ENV`` arguments into names and unprocessed values."""
    words = _raw_words(instruction.args, escape)
    if not words:
        return []
    if "=" not in words[0]:
        name = words[0]
        value = instruction.args[len(name) :].strip()
        return [(name, value)]
    pairs = []
    for word in words:
        name, separator, value = word.partition("=")
        if not separator:
            _fail(instruction, f"ENV {word} has no value")
        pairs.append((name, value))
    return pairs


@dataclass(frozen=True)
class ContextSource:
    """A file or directory from the build context."""

    path: str


@dataclass(frozen=True)
class ArchiveSource:
    """A local tar archive that ``ADD`` extracts."""

    path: str


@dataclass(frozen=True)
class UrlSource:
    """A remote file that ``ADD`` downloads."""

    url: str


@dataclass(frozen=True)
class ScriptSource:
    """A generated file, such as a heredoc script."""

    content: str
    mode: int = 0o755


@dataclass
class DefinitionStage:
    """One Apptainer build stage."""

    name: str
    image: str
    # The build script's instruction steps; ``None`` for an image that is
    # only a ``COPY --from`` source and needs no ``%post``.
    body: str | None = None
    # Files to stage, keyed by their path under the stage's staging root.
    staged: dict[
        str, ContextSource | ArchiveSource | UrlSource | ScriptSource
    ] = field(default_factory=dict)
    # ``%files from`` entries: (stage name, source path, destination).
    files_from: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def script(self) -> str | None:
        if self.body is None:
            return None
        return _SCRIPT_PRELUDE + self.body


@dataclass
class Definition:
    """A Dockerfile translated into Apptainer build stages."""

    stages: list[DefinitionStage]
    context_dir: Path
    # Constructs the translation ignores or approximates.
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Base images whose config could not be read, so the build differs
    # from one that could; the SIF is cached apart.
    unread_configs: list[str] = field(default_factory=list)

    def render(self, staging_dir: Path) -> str:
        """Return the definition file text for staging under ``staging_dir``."""
        sections = []
        for stage in self.stages:
            lines = ["Bootstrap: docker", f"From: {stage.image}"]
            if len(self.stages) > 1:
                lines.append(f"Stage: {stage.name}")
            by_stage: dict[str, list[str]] = {}
            for source_stage, source, destination in stage.files_from:
                by_stage.setdefault(source_stage, []).append(
                    f"    {source} {destination}"
                )
            for source_stage, entries in by_stage.items():
                lines.extend(["", f"%files from {source_stage}", *entries])
            if stage.body is not None:
                staged = shlex.quote(f"{staging_dir / stage.name}/.")
                lines.extend([
                    "",
                    "%setup",
                    f'    mkdir -p "$APPTAINER_ROOTFS{_STAGING}"',
                    f'    cp -a {staged} "$APPTAINER_ROOTFS{_STAGING}/"',
                    "",
                    "%post",
                    f"    /bin/sh {_STAGING}/build.sh",
                    f"    rm -rf {_STAGING}",
                ])
            sections.append("\n".join(lines) + "\n")
        text = "\n".join(sections)
        if "{{" in text:
            raise DockerfileUnsupported(
                "Apptainer would treat '{{' in a path or image as a template"
            )
        return text

    def write(
        self,
        staging_dir: Path,
        *,
        download: Callable[[str, Path], None] | None = None,
    ) -> Path:
        """Stage the build scripts and sources and write the definition."""
        download = download or _download
        ignore = dockerignore(self.context_dir)
        for stage in self.stages:
            if stage.body is None:
                continue
            root = staging_dir / stage.name
            root.mkdir(parents=True)
            script = root / "build.sh"
            script.write_text(stage.script or "")
            script.chmod(0o755)
            for relative, source in stage.staged.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(source, ContextSource):
                    _copy_context_path(
                        self.context_dir, source.path, target, ignore
                    )
                elif isinstance(source, ArchiveSource):
                    _extract_archive(self.context_dir / source.path, target)
                elif isinstance(source, UrlSource):
                    download(source.url, target)
                    target.chmod(0o600)
                else:
                    target.write_text(source.content)
                    target.chmod(source.mode)
        path = staging_dir / "build.def"
        path.write_text(self.render(staging_dir))
        return path


def _extract_archive(archive_path: Path, target: Path) -> None:
    """Extract a local archive for ``ADD``, with its members' metadata.

    The host user cannot set the members' owners, so a sibling
    ``.ursa-members`` script restores them, and their modes, in ``%post``.
    """
    # By name: extractall filters a directory again to set its mode.
    members: dict[str, tarfile.TarInfo] = {}

    def keep(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:
        # The "tar" filter's path checks, without its mode changes.
        checked = tarfile.tar_filter(member, path)
        checked = checked.replace(mode=member.mode, deep=False)
        members.setdefault(checked.name, checked)
        return checked

    target.mkdir()
    with tarfile.open(archive_path, "r:*") as archive:
        archive.extractall(target, filter=keep)
    lines = [
        f"_ursa_member {member.uid}:{member.gid} {member.mode & 0o7777:o} "
        f"{shlex.quote('./' + member.name)}\n"
        for member in members.values()
    ]
    target.with_name(f"{target.name}.ursa-members").write_text("".join(lines))


def _download(url: str, target: Path) -> None:
    with (
        urllib.request.urlopen(url, timeout=300) as response,  # noqa: S310
        target.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)


def _ignored(ignore: DockerIgnore | None, relative: str) -> bool:
    return ignore is not None and bool(relative) and ignore.ignored(relative)


def _in_context(
    context_dir: Path, relative: str, ignore: DockerIgnore | None
) -> bool:
    """Whether Docker would send a context path, ignored or not."""
    if not _ignored(ignore, relative):
        return True
    # An exception can re-include something inside an ignored directory,
    # which Docker then sends along with its parents.
    path = context_dir / relative
    if ignore is None or not ignore.has_exceptions or path.is_symlink():
        return False
    for directory, dirnames, filenames in os.walk(path):
        prefix = Path(directory).relative_to(context_dir).as_posix()
        for name in [*dirnames, *filenames]:
            if not ignore.ignored(f"{prefix}/{name}"):
                return True
    return False


def _copy_context_path(
    context_dir: Path,
    relative: str,
    target: Path,
    ignore: DockerIgnore | None,
) -> None:
    """Copy a context path, following only a top-level symlink as Docker does."""
    source = (context_dir / relative).resolve() if relative else context_dir
    if not source.is_dir():
        shutil.copy2(source, target)
        return
    target.mkdir()
    directories = [(source, target)]

    def parent(prefix: Path) -> Path:
        # Create the ignored parents of a path that an exception re-includes.
        missing = []
        while not (target / prefix).is_dir():
            missing.append(prefix)
            prefix = prefix.parent
        for path in reversed(missing):
            (target / path).mkdir()
            directories.append((source / path, target / path))
        return target / (missing[0] if missing else prefix)

    for directory, dirnames, filenames in os.walk(source):
        current = Path(directory)
        prefix = current.relative_to(source)
        for name in list(dirnames):
            path = current / name
            rel = posixpath.join(relative, (prefix / name).as_posix())
            ignored = _ignored(ignore, rel)
            if path.is_symlink():
                # os.walk lists links to directories here; keep the link.
                dirnames.remove(name)
                if not ignored:
                    os.symlink(os.readlink(path), parent(prefix) / name)
            elif not ignored:
                (parent(prefix) / name).mkdir()
                directories.append((path, target / prefix / name))
            elif ignore is None or not ignore.has_exceptions:
                dirnames.remove(name)
        for name in filenames:
            rel = posixpath.join(relative, (prefix / name).as_posix())
            if not _ignored(ignore, rel):
                shutil.copy2(
                    current / name,
                    parent(prefix) / name,
                    follow_symlinks=False,
                )
    # Copy directory modes last, so read-only directories can be filled.
    for path, copy in reversed(directories):
        shutil.copystat(path, copy)


def _go_glob(segment: str) -> str:
    """Convert a Go ``filepath.Match`` pattern to an ``fnmatch`` one."""
    converted = []
    index = 0
    while index < len(segment):
        character = segment[index]
        if character == "\\" and index + 1 < len(segment):
            converted.append(f"[{segment[index + 1]}]")
            index += 2
            continue
        if character == "[" and segment.startswith("[^", index):
            converted.append("[!")
            index += 2
            continue
        converted.append(character)
        index += 1
    return "".join(converted)


def _glob_context(
    context_dir: Path, pattern: str, ignore: DockerIgnore | None
) -> list[str]:
    def walk(base: str, segments: list[str]) -> Iterator[str]:
        directory = context_dir / base
        segment = segments[0]
        if _GLOB_CHARACTERS.isdisjoint(segment):
            names = [segment] if os.path.lexists(directory / segment) else []
        else:
            matcher = _go_glob(segment)
            try:
                names = sorted(
                    entry.name
                    for entry in os.scandir(directory)
                    if fnmatch.fnmatchcase(entry.name, matcher)
                )
            except OSError:
                names = []
        for name in names:
            relative = posixpath.join(base, name) if base else name
            path = context_dir / relative
            if not _in_context(context_dir, relative, ignore):
                continue
            if len(segments) == 1:
                yield relative
            elif path.is_dir() and not path.is_symlink():
                yield from walk(relative, segments[1:])

    return list(walk("", pattern.split("/")))


class _Translator:
    def __init__(
        self,
        parsed: ParsedDockerfile,
        context_dir: Path,
        build_args: Mapping[str, str],
        machine: str,
        environ: Mapping[str, str],
        image_config: ImageConfig | None = None,
    ) -> None:
        self.parsed = parsed
        self.context_dir = context_dir
        self.build_args = build_args
        self.machine = machine
        self.environ = environ
        self.image_config = image_config
        self.ignore = dockerignore(context_dir)
        self.global_scope, self.stages = _split_stages(
            parsed, build_args, machine, strict=True
        )
        self.plans: dict[int, DefinitionStage] = {}
        self.external: dict[str, DefinitionStage] = {}
        self.order: list[DefinitionStage] = []
        self.notes: list[str] = []
        self.warnings: list[str] = []
        self.unread_configs: list[str] = []

    def stage_by_reference(self, reference: str, before: int) -> _Stage | None:
        if reference.isdigit():
            index = int(reference)
            return self.stages[index] if index < before else None
        for stage in self.stages[:before]:
            if stage.name == reference.lower():
                return stage
        return None

    def translate(self, target: str | None) -> Definition:
        if not self.stages:
            raise DockerfileUnsupported("the Dockerfile has no FROM")
        if target is None:
            final = self.stages[-1]
        else:
            found = [s for s in self.stages if s.name == target.lower()]
            if not found:
                raise DockerfileUnsupported(
                    f"target stage {target!r} not found"
                )
            final = found[0]
        self.ensure(final, final=True)
        return Definition(
            self.order,
            self.context_dir,
            self.notes,
            self.warnings,
            self.unread_configs,
        )

    def ensure(self, stage: _Stage, *, final: bool = False) -> DefinitionStage:
        plan = self.plans.get(stage.index)
        if plan is None:
            plan = self.translate_stage(stage, final=final)
            self.plans[stage.index] = plan
            self.order.append(plan)
        return plan

    def external_stage(self, image: str, instruction: Instruction) -> str:
        plan = self.external.get(image)
        if plan is None:
            _check_image(image, instruction)
            plan = DefinitionStage(f"x{len(self.external)}", image)
            self.external[image] = plan
            self.order.append(plan)
        return plan.name

    def translate_stage(self, stage: _Stage, *, final: bool) -> DefinitionStage:
        image = stage.image
        if image.lower() == "scratch":
            _fail(stage.instruction, "FROM scratch is not supported")
        if self.stage_by_reference(image, stage.index) is not None:
            _fail(
                stage.instruction,
                "FROM an earlier build stage is not supported",
            )
        _check_image(image, stage.instruction)
        return _StageScript(self, stage, final=final).run()


def _check_image(image: str, instruction: Instruction) -> None:
    if not image or re.search(r"\s|\{\{", image) or "://" in image:
        _fail(instruction, f"cannot bootstrap from image {image!r}")


class _StageScript:
    """Translate one stage's instructions into build script steps."""

    def __init__(self, translator: _Translator, stage: _Stage, *, final: bool):
        self.translator = translator
        self.stage = stage
        self.final = final
        self.escape = translator.parsed.escape
        self.plan = DefinitionStage(f"s{stage.index}", stage.image, "")
        self.scope: _Scope = {}
        # Without the base image's environment, variables it may set are
        # left for the build script to expand.
        self.defer = True
        self.env_names: list[str] = []
        self.shell: list[str] = list(_DEFAULT_SHELL)
        self.user: _Value = ()
        self.lines: list[str] = []
        self.steps = 0

    def base_image(self) -> None:
        """Start in the base image's ``WorkingDir`` as its ``User``.

        Variables also expand against the image's environment, as in
        Docker, rather than against the build shell's, which has Apptainer's
        own variables and a ``HOME``.
        """
        lookup = self.translator.image_config
        if lookup is None:
            return
        config = lookup(self.stage.image, self.translator.machine)
        if config is None:
            self.translator.warnings.append(
                f"line {self.stage.instruction.line}: cannot read the config "
                f"of {self.stage.image}; build steps start in / as root "
                "instead of its WORKDIR and USER"
            )
            self.translator.unread_configs.append(self.stage.image)
            return
        self.scope.update(_base_scope(config))
        self.defer = False
        workdir = config.get("WorkingDir") or ""
        if workdir not in {"", "/"}:
            self.lines.append(f"_ursa_workdir {shlex.quote(workdir)}")
        user = config.get("User") or ""
        if user:
            self.user = (user,)
            self.lines.append(f"_ursa_user={shlex.quote(user)}")

    def run(self) -> DefinitionStage:
        proxies = [
            (name, self.translator.build_args[name])
            for name in _PROXY_ARGS
            if name in self.translator.build_args
        ] or [
            (name, self.translator.environ[name])
            for name in _PROXY_ARGS
            if self.translator.environ.get(name)
        ]
        # Docker passes predefined proxy arguments to RUN without an ARG.
        for name, value in proxies:
            self.lines.append(f"export {name}={shlex.quote(value)}")
        self.lines.append("_ursa_private_tmp")
        self.base_image()
        for instruction in self.stage.instructions:
            if instruction.keyword in _IGNORED_INSTRUCTIONS:
                self.translator.notes.append(
                    f"line {instruction.line}: {instruction.keyword} is ignored"
                )
                continue
            handler = getattr(self, f"do_{instruction.keyword.lower()}", None)
            if handler is None:
                _fail(
                    instruction,
                    f"unknown instruction {instruction.keyword}",
                )
            try:
                handler(instruction)
            except _LexError as exc:
                _fail(instruction, str(exc))
        if self.final:
            if self.env_names:
                self.lines.append("_ursa_save_env " + " ".join(self.env_names))
            user = _text(self.user) if _is_static(self.user) else "?"
            if self.user and user.split(":")[0] not in {"root", "0"}:
                self.translator.notes.append(
                    f"the final USER {user} is not applied at run time; "
                    "Harbor chooses the container user"
                )
        # A stage without steps, such as one that only names an image for
        # COPY --from, needs no %post, so its image needs no shell.
        if self.steps:
            self.plan.body = "\n".join(self.lines) + "\n"
        else:
            self.plan.body = None
        return self.plan

    def emit(self, instruction: Instruction, *commands: str) -> None:
        summary = instruction.summary.splitlines()[0]
        self.steps += 1
        self.lines.append(
            f"_ursa_step {instruction.line} {shlex.quote(summary)}"
        )
        self.lines.extend(commands)

    def lex(self, text: str) -> _Value:
        return _resolve(
            _Lexer(text, self.escape).word(), self.scope, defer=self.defer
        )

    def static(self, value: _Value, instruction: Instruction, what: str) -> str:
        for part in value:
            if isinstance(part, _Ref):
                _fail(
                    instruction,
                    f"{what} uses ${part.name}, which is only known inside "
                    "the image",
                )
        return _text(value)

    def check_name(self, name: str, instruction: Instruction) -> None:
        if _NAME.fullmatch(name) is None or name.startswith("_ursa_"):
            _fail(instruction, f"{name!r} is not a supported variable name")

    def do_arg(self, instruction: Instruction) -> None:
        exports = []
        for word in _raw_words(instruction.args, self.escape):
            name, separator, default = word.partition("=")
            self.check_name(name, instruction)
            if name in self.translator.build_args:
                value: _Value | None = (self.translator.build_args[name],)
            elif separator:
                value = self.lex(default)
            elif name in self.translator.global_scope:
                value = self.translator.global_scope[name]
            else:
                value = None
            # ENV wins over an ARG of the same name, as in Docker. An ARG
            # without a value leaves the name to the base image.
            if name in self.env_names or value is None:
                continue
            self.scope[name] = value
            exports.append(f"{name}={_shell_word(value)}")
        self.emit(
            instruction, *([f"export {' '.join(exports)}"] if exports else [])
        )

    def do_env(self, instruction: Instruction) -> None:
        pairs = _env_pairs(instruction, self.escape)
        if not pairs:
            _fail(instruction, "ENV needs a name and a value")
        # Every value in one ENV sees the variables from before it.
        values = [(name, self.lex(raw)) for name, raw in pairs]
        exports = []
        for name, value in values:
            self.check_name(name, instruction)
            self.scope[name] = value
            if name not in self.env_names:
                self.env_names.append(name)
            exports.append(f"{name}={_shell_word(value)}")
        self.emit(instruction, f"export {' '.join(exports)}")

    def do_workdir(self, instruction: Instruction) -> None:
        value = self.lex(instruction.args)
        if not value:
            _fail(instruction, "WORKDIR needs a path")
        self.emit(instruction, f"_ursa_workdir {_shell_word(value)}")

    def do_user(self, instruction: Instruction) -> None:
        value = self.lex(instruction.args)
        if not value:
            _fail(instruction, "USER needs a user")
        self.user = value
        self.emit(instruction, f"_ursa_user={_shell_word(value)}")

    def do_shell(self, instruction: Instruction) -> None:
        shell = _json_array(instruction.args)
        if not shell:
            _fail(instruction, "SHELL needs a JSON array")
        self.shell = shell
        self.emit(instruction)

    def do_cmd(self, instruction: Instruction) -> None:
        self.translator.notes.append(
            f"line {instruction.line}: {instruction.keyword} is ignored; "
            "Harbor runs its own commands"
        )

    do_entrypoint = do_cmd

    def do_onbuild(self, instruction: Instruction) -> None:
        self.translator.warnings.append(
            f"line {instruction.line}: ONBUILD is ignored"
        )

    def do_run(self, instruction: Instruction) -> None:
        flags, command = _split_flags(instruction.args)
        for value in flags.pop("mount", []):
            options = dict(
                option.partition("=")[::2] for option in value.split(",")
            )
            if options.get("type") != "cache":
                _fail(instruction, f"RUN --mount={value} is not supported")
            self.translator.notes.append(
                f"line {instruction.line}: RUN --mount={value} is ignored"
            )
        for value in flags.pop("network", []):
            if value not in {"default", "host"}:
                _fail(instruction, f"RUN --network={value} is not supported")
        for value in flags.pop("security", []):
            if value != "sandbox":
                _fail(instruction, f"RUN --security={value} is not supported")
        if flags:
            _fail(instruction, f"RUN --{sorted(flags)[0]} is not supported")
        command = command.strip()
        if not command:
            _fail(instruction, "RUN needs a command")
        argv: list[str]
        if instruction.heredocs:
            heredoc = instruction.heredocs[0]
            # Like BuildKit, run a RUN that is only a heredoc as a script.
            sole = (
                len(instruction.heredocs) == 1
                and len(_raw_words(command, self.escape)) == 1
            )
            if sole and heredoc.body.startswith("#!"):
                path = f"run-{instruction.line}"
                self.plan.staged[path] = ScriptSource(heredoc.body)
                argv = [f"{_STAGING}/{path}"]
            elif sole:
                argv = [*self.shell, heredoc.body]
            else:
                bodies = "".join(item.raw for item in instruction.heredocs)
                argv = [*self.shell, f"{command}\n{bodies}"]
        elif (exec_form := _json_array(command)) is not None:
            if not exec_form:
                _fail(instruction, "RUN needs a command")
            argv = exec_form
        else:
            argv = [*self.shell, command]
        self.emit(instruction, "_ursa_run " + shlex.join(argv))

    def do_copy(self, instruction: Instruction) -> None:
        keyword = instruction.keyword
        flags, rest = _split_flags(instruction.args)
        if instruction.heredocs:
            _fail(instruction, f"{keyword} with a heredoc is not supported")
        flags.pop("link", None)
        owner = flags.pop("chown", [""])[-1]
        mode = flags.pop("chmod", [""])[-1]
        source_stage = (
            flags.pop("from", [None])[-1] if keyword == "COPY" else None
        )
        if flags:
            _fail(
                instruction, f"{keyword} --{sorted(flags)[0]} is not supported"
            )
        exec_form = _json_array(rest)
        words = (
            [_Lexer(item, self.escape).word() for item in exec_form]
            if exec_form is not None
            else _Lexer(rest, self.escape).words()
        )
        words = [_resolve(word, self.scope, defer=self.defer) for word in words]
        if len(words) < 2:
            _fail(instruction, f"{keyword} needs a source and a destination")
        *sources, destination = words
        if not destination:
            _fail(instruction, f"{keyword} needs a destination")
        directory_destination = isinstance(destination[-1], str) and (
            destination[-1].endswith("/")
            or posixpath.basename(destination[-1]) in {".", ".."}
        )
        if (
            len(sources) > 1
            and isinstance(destination[-1], str)
            and not directory_destination
        ):
            _fail(
                instruction,
                f"{keyword} with several sources needs a destination ending "
                "in /",
            )
        if owner:
            owner_word = _shell_word(self.lex(owner))
        else:
            # Like BuildKit, COPY --from keeps the source's owners, which
            # %files from copies when the build runs as (fake) root.
            owner_word = "''" if source_stage is None else "-"
        mode_text = ""
        if mode:
            mode_text = self.static(self.lex(mode), instruction, "--chmod")
            if _CHMOD.fullmatch(mode_text) is None:
                _fail(instruction, f"--chmod={mode_text} is not supported")
        key = f"{keyword.lower()}-{instruction.line}"
        if source_stage is not None:
            staged = self.copy_from_stage(
                instruction,
                self.static(self.lex(source_stage), instruction, "--from"),
                [self.static(s, instruction, "the source") for s in sources],
                key,
            )
        else:
            staged = self.copy_from_context(
                instruction,
                [self.static(s, instruction, "the source") for s in sources],
                key,
            )
        if (
            len(staged) > 1
            and isinstance(destination[-1], str)
            and not directory_destination
        ):
            _fail(
                instruction,
                f"{keyword} of several files needs a destination ending in /",
            )
        self.emit(
            instruction,
            f"_ursa_copy {owner_word} {shlex.quote(mode_text)} "
            + _shell_word(destination)
            + "".join(f" {shlex.quote(path)}" for path in staged),
        )

    do_add = do_copy

    def copy_from_stage(
        self,
        instruction: Instruction,
        reference: str,
        sources: list[str],
        key: str,
    ) -> list[str]:
        stage = self.translator.stage_by_reference(reference, self.stage.index)
        if stage is not None:
            name = self.translator.ensure(stage).name
        else:
            name = self.translator.external_stage(reference, instruction)
        staged = []
        for index, source in enumerate(sources):
            if (
                not source
                or re.search(r"\s|\{\{", source)
                or not _GLOB_CHARACTERS.isdisjoint(source)
            ):
                _fail(
                    instruction,
                    f"COPY --from source {source!r} is not supported",
                )
            path = posixpath.normpath(posixpath.join("/", source))
            leaf = posixpath.basename(path) or "root"
            destination = f"{_STAGING}/{key}/{index}/{leaf}"
            self.plan.files_from.append((name, path, destination))
            staged.append(destination)
        return staged

    def copy_from_context(
        self, instruction: Instruction, sources: list[str], key: str
    ) -> list[str]:
        translator = self.translator
        staged = []
        for source in sources:
            if instruction.keyword == "ADD" and re.match(
                r"[a-z][a-z0-9+.-]*://|git@", source, re.IGNORECASE
            ):
                staged.append(self.add_url(instruction, source, key, staged))
                continue
            relative = posixpath.normpath(source.lstrip("/") or ".")
            if relative == ".." or relative.startswith("../"):
                _fail(instruction, f"source {source} is outside the context")
            if relative == ".":
                matches = [""]
            elif _GLOB_CHARACTERS.isdisjoint(relative):
                path = translator.context_dir / relative
                if not os.path.lexists(path) or not _in_context(
                    translator.context_dir, relative, translator.ignore
                ):
                    _fail(
                        instruction,
                        f"source {source} is not in the build context",
                    )
                matches = [relative]
            else:
                matches = _glob_context(
                    translator.context_dir, relative, translator.ignore
                )
                if not matches:
                    _fail(instruction, f"no source files match {source}")
            for match in matches:
                path = translator.context_dir / match
                if path.is_symlink():
                    resolved = path.resolve()
                    if not resolved.is_relative_to(
                        translator.context_dir.resolve()
                    ):
                        _fail(
                            instruction,
                            f"source {match} links outside the build context",
                        )
                    if not resolved.exists():
                        _fail(instruction, f"source {match} is a broken link")
                leaf = posixpath.basename(match) or "context"
                staged_path = f"{key}/{len(staged)}/{leaf}"
                if (
                    instruction.keyword == "ADD"
                    and path.is_file()
                    and tarfile.is_tarfile(path)
                ):
                    self.plan.staged[staged_path] = ArchiveSource(match)
                else:
                    self.plan.staged[staged_path] = ContextSource(match)
                staged.append(f"{_STAGING}/{staged_path}")
        return staged

    def add_url(
        self,
        instruction: Instruction,
        url: str,
        key: str,
        staged: list[str],
    ) -> str:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or (
            parsed.path.endswith(".git")
        ):
            _fail(instruction, f"ADD {url} is not supported")
        leaf = posixpath.basename(parsed.path)
        if not leaf or re.search(r"[^\w.+-]", leaf):
            _fail(instruction, f"ADD {url} needs a plain file name in the URL")
        staged_path = f"{key}/{len(staged)}/{leaf}"
        self.plan.staged[staged_path] = UrlSource(url)
        return f"{_STAGING}/{staged_path}"


def translate_dockerfile(
    dockerfile: Path,
    context_dir: Path,
    *,
    build_args: Sequence[str] = (),
    target: str | None = None,
    machine: str | None = None,
    environ: Mapping[str, str] | None = None,
    image_config: ImageConfig | None = None,
) -> Definition:
    """Translate a Dockerfile into Apptainer build stages.

    Raises ``DockerfileUnsupported``, naming the line, for constructs the
    translation cannot reproduce, before anything is built. ``image_config``
    looks up a base image's OCI config, as ``image_config()`` does, so build
    steps start in its ``WorkingDir`` as its ``User``; without it they start
    in ``/`` as root.
    """
    try:
        parsed = parse_dockerfile(dockerfile.read_text())
        translator = _Translator(
            parsed,
            context_dir,
            _parse_build_args(build_args),
            machine or platform.machine(),
            os.environ if environ is None else environ,
            image_config,
        )
        return translator.translate(target)
    except DockerfileUnsupported as exc:
        raise DockerfileUnsupported(f"{dockerfile}: {exc}") from None


_SCRIPT_PRELUDE = r"""#!/bin/sh
# Generated by URSA from a Dockerfile; run from Apptainer's %post.
set -e
umask 022
_ursa_wd=/
_ursa_user=

_ursa_step() {
    printf '#%s %s\n' "$1" "$2"
}

# %post binds the host's /tmp and /var/tmp; unmount them so files that a
# step leaves there stay in the image and concurrent builds do not mix.
_ursa_private_tmp() {
    for _ursa_dir in /var/tmp /tmp; do
        _ursa_n=0
        while [ "$_ursa_n" -lt 8 ] && umount "$_ursa_dir" 2>/dev/null; do
            _ursa_n=$((_ursa_n + 1))
        done
    done
    [ -r /proc/self/mountinfo ] || return 0
    while read -r _ _ _ _ _ursa_mount _; do
        case $_ursa_mount in
        /tmp | /var/tmp)
            echo "ursa: warning: $_ursa_mount is the host directory in" \
                "this build; files written there are not kept" >&2
            ;;
        esac
    done </proc/self/mountinfo
}

_ursa_run() {
    (
        cd "$_ursa_wd"
        _ursa_name=${_ursa_user%%:*}
        case $_ursa_name in
        '' | root | 0) exec "$@" ;;
        esac
        # "--" keeps older su from reading the step's own options.
        if su -s /bin/sh -c : -- "$_ursa_name" >/dev/null 2>&1; then
            exec su -s /bin/sh -c 'exec "$0" "$@"' -- "$_ursa_name" "$@"
        fi
        echo "ursa: warning: cannot switch to user $_ursa_name in this" \
            "build; running the step as root" >&2
        exec "$@"
    )
}

_ursa_workdir() {
    case $1 in
    /*) _ursa_wd=$1 ;;
    *) _ursa_wd=${_ursa_wd%/}/$1 ;;
    esac
    mkdir -p "$_ursa_wd"
}

# _ursa_member OWNER MODE PATH: restore an archive member's owner and mode,
# which ADD keeps; the mode goes last, since chown clears setuid bits.
_ursa_member() {
    if [ -z "$_ursa_given" ] && ! chown -h "$1" "$3"; then
        echo "ursa: warning: cannot chown $3 to $1" >&2
    fi
    [ -L "$3" ] || chmod "$2" "$3"
}

# _ursa_copy OWNER MODE DEST SOURCE...: copy staged sources as COPY does.
# Build context files belong to root unless OWNER is given; "-" keeps the
# owners of files from another stage. A user without a group also names
# the group, by the user's UID.
_ursa_copy() {
    _ursa_owner=$1
    _ursa_given=$1
    case $_ursa_owner in
    '') _ursa_owner=0:0 ;;
    - | *:*) ;;
    *[!0-9]*)
        if _ursa_uid=$(id -u "$_ursa_owner" 2>/dev/null); then
            _ursa_owner=$_ursa_owner:$_ursa_uid
        fi
        ;;
    *) _ursa_owner=$_ursa_owner:$_ursa_owner ;;
    esac
    _ursa_mode=$2
    _ursa_dest=$3
    shift 3
    case $_ursa_dest in
    /*) ;;
    *) _ursa_dest=${_ursa_wd%/}/$_ursa_dest ;;
    esac
    case $_ursa_dest in
    */ | */. | */..) _ursa_into=1 ;;
    *) _ursa_into= ;;
    esac
    for _ursa_src do
        if [ "$_ursa_owner" != - ] && ! chown -hR "$_ursa_owner" "$_ursa_src"
        then
            echo "ursa: warning: cannot chown to $_ursa_owner" >&2
        fi
        # An extracted archive keeps its members' owners and modes.
        if [ -f "$_ursa_src.ursa-members" ]; then
            (cd "$_ursa_src" && . "$_ursa_src.ursa-members")
        fi
        [ -z "$_ursa_mode" ] || chmod -R "$_ursa_mode" "$_ursa_src"
        if [ -d "$_ursa_src" ] && [ ! -L "$_ursa_src" ]; then
            mkdir -p "$_ursa_dest"
            for _ursa_entry in "$_ursa_src"/* "$_ursa_src"/.[!.]* \
                "$_ursa_src"/..?*; do
                if [ -e "$_ursa_entry" ] || [ -L "$_ursa_entry" ]; then
                    cp -a "$_ursa_entry" "$_ursa_dest/"
                fi
            done
        elif [ -n "$_ursa_into" ] || [ -d "$_ursa_dest" ]; then
            mkdir -p "$_ursa_dest"
            cp -a "$_ursa_src" "$_ursa_dest/"
        else
            mkdir -p "${_ursa_dest%/*}/"
            cp -a "$_ursa_src" "$_ursa_dest"
        fi
    done
}

_ursa_quote() {
    _ursa_rest=$1
    _ursa_out=
    while :; do
        case $_ursa_rest in
        *\'*)
            _ursa_out="$_ursa_out${_ursa_rest%%\'*}'\\''"
            _ursa_rest=${_ursa_rest#*\'}
            ;;
        *) break ;;
        esac
    done
    printf "'%s%s'" "$_ursa_out" "$_ursa_rest"
}

# Drop the entry that Apptainer adds to PATH only while it builds.
_ursa_build_path() {
    _ursa_rest=$1:
    _ursa_out=
    while [ -n "$_ursa_rest" ]; do
        _ursa_entry=${_ursa_rest%%:*}
        _ursa_rest=${_ursa_rest#*:}
        [ "$_ursa_entry" = /.singularity.d/libs ] && continue
        _ursa_out=${_ursa_out:+$_ursa_out:}$_ursa_entry
    done
    printf '%s' "$_ursa_out"
}

# Keep the final ENV values at run time, as Docker records them in the image.
_ursa_save_env() {
    _ursa_env=${APPTAINER_ENVIRONMENT:-/.singularity.d/env/91-environment.sh}
    for _ursa_name do
        eval "_ursa_value=\${$_ursa_name-}"
        if [ "$_ursa_name" = PATH ]; then
            _ursa_value=$(_ursa_build_path "$_ursa_value")
        fi
        printf 'export %s=%s\n' "$_ursa_name" \
            "$(_ursa_quote "$_ursa_value")" >>"$_ursa_env"
    done
}

"""
