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
import tarfile
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

from pathspec import GitIgnoreSpec

# Part of the SIF cache key for definition builds; bump it whenever the
# generated definition or build script changes.
DOCKERFILE_TRANSLATOR_VERSION = 1

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


def dockerignore_spec(context_dir: Path) -> GitIgnoreSpec | None:
    path = context_dir / ".dockerignore"
    if not path.is_file():
        return None
    return GitIgnoreSpec.from_lines(path.read_text().splitlines())


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


def dockerfile_workdir(
    text: str,
    build_args: Sequence[str] = (),
    target: str | None = None,
    machine: str | None = None,
) -> str:
    """Return the final ``WORKDIR`` of a stage without building it.

    Variables defined by the Dockerfile and its build arguments are
    resolved; anything set only by the base image raises ``ValueError``.
    Constructs that only a build would reject are ignored here.
    """
    parsed = parse_dockerfile(text)
    arguments = _parse_build_args(build_args)
    global_scope, stages = _split_stages(
        parsed, arguments, machine or platform.machine(), strict=False
    )
    workdirs: dict[int, str] = {}
    names: dict[str, int] = {}
    for stage in stages:
        inherited = names.get(stage.image.lower())
        workdir = workdirs[inherited] if inherited is not None else "/"
        scope: _Scope = {}
        for instruction in stage.instructions:
            words = _raw_words(instruction.args, parsed.escape)
            if instruction.keyword == "ARG":
                for word in words:
                    name, separator, default = word.partition("=")
                    if name in arguments:
                        scope[name] = (arguments[name],)
                    elif separator:
                        try:
                            scope[name] = _resolve(
                                _Lexer(default, parsed.escape).word(),
                                scope,
                                defer=True,
                            )
                        except _LexError:
                            scope[name] = (_Ref(name),)
                    else:
                        scope[name] = global_scope.get(name)
            elif instruction.keyword == "ENV":
                for name, raw in _env_pairs(instruction, parsed.escape):
                    try:
                        scope[name] = _resolve(
                            _Lexer(raw, parsed.escape).word(),
                            scope,
                            defer=True,
                        )
                    except _LexError:
                        scope[name] = (_Ref(name),)
            elif instruction.keyword == "WORKDIR":
                try:
                    value = _resolve(
                        _Lexer(instruction.args, parsed.escape).word(),
                        scope,
                        defer=True,
                    )
                except _LexError:
                    value = (_Ref("?"),)
                if not _is_static(value):
                    raise ValueError(
                        "Singularity cannot resolve variables in Dockerfile "
                        f"WORKDIR: {instruction.args}"
                    )
                workdir = posixpath.normpath(
                    posixpath.join(workdir, _text(value))
                )
        workdirs[stage.index] = workdir
        if stage.name is not None:
            names[stage.name] = stage.index
    if not stages:
        return "/"
    if target is None:
        return workdirs[stages[-1].index]
    index = names.get(target.lower())
    return workdirs[index] if index is not None else "/"


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
        ignore = dockerignore_spec(self.context_dir)
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
                    target.mkdir()
                    with tarfile.open(
                        self.context_dir / source.path, "r:*"
                    ) as archive:
                        archive.extractall(target, filter="tar")
                elif isinstance(source, UrlSource):
                    download(source.url, target)
                    target.chmod(0o600)
                else:
                    target.write_text(source.content)
                    target.chmod(source.mode)
        path = staging_dir / "build.def"
        path.write_text(self.render(staging_dir))
        return path


def _download(url: str, target: Path) -> None:
    with (
        urllib.request.urlopen(url, timeout=300) as response,  # noqa: S310
        target.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)


def _ignored(ignore: GitIgnoreSpec | None, relative: str, is_dir: bool) -> bool:
    if ignore is None or not relative:
        return False
    return ignore.match_file(relative) or (
        is_dir and ignore.match_file(f"{relative}/")
    )


def _copy_context_path(
    context_dir: Path,
    relative: str,
    target: Path,
    ignore: GitIgnoreSpec | None,
) -> None:
    """Copy a context path, following only a top-level symlink as Docker does."""
    source = (context_dir / relative).resolve() if relative else context_dir
    if not source.is_dir():
        shutil.copy2(source, target)
        return
    target.mkdir()
    directories = [(source, target)]
    for directory, dirnames, filenames in os.walk(source):
        current = Path(directory)
        prefix = current.relative_to(source)
        for name in list(dirnames):
            path = current / name
            rel = posixpath.join(relative, (prefix / name).as_posix())
            if _ignored(ignore, rel, True):
                dirnames.remove(name)
            elif path.is_symlink():
                # os.walk lists links to directories here; keep the link.
                dirnames.remove(name)
                os.symlink(os.readlink(path), target / prefix / name)
            else:
                (target / prefix / name).mkdir()
                directories.append((path, target / prefix / name))
        for name in filenames:
            rel = posixpath.join(relative, (prefix / name).as_posix())
            if not _ignored(ignore, rel, False):
                shutil.copy2(
                    current / name,
                    target / prefix / name,
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
    context_dir: Path, pattern: str, ignore: GitIgnoreSpec | None
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
            if _ignored(ignore, relative, path.is_dir()):
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
    ) -> None:
        self.parsed = parsed
        self.context_dir = context_dir
        self.build_args = build_args
        self.machine = machine
        self.environ = environ
        self.ignore = dockerignore_spec(context_dir)
        self.global_scope, self.stages = _split_stages(
            parsed, build_args, machine, strict=True
        )
        self.plans: dict[int, DefinitionStage] = {}
        self.external: dict[str, DefinitionStage] = {}
        self.order: list[DefinitionStage] = []
        self.notes: list[str] = []
        self.warnings: list[str] = []

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
            self.order, self.context_dir, self.notes, self.warnings
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
        self.env_names: list[str] = []
        self.shell: list[str] = list(_DEFAULT_SHELL)
        self.user: _Value = ()
        self.lines: list[str] = []

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
        self.plan.body = "\n".join(self.lines) + "\n"
        return self.plan

    def emit(self, instruction: Instruction, *commands: str) -> None:
        summary = instruction.summary.splitlines()[0]
        self.lines.append(
            f"_ursa_step {instruction.line} {shlex.quote(summary)}"
        )
        self.lines.extend(commands)

    def lex(self, text: str) -> _Value:
        return _resolve(
            _Lexer(text, self.escape).word(), self.scope, defer=True
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
            # ENV wins over an ARG of the same name, as in Docker.
            if name in self.env_names:
                continue
            self.scope[name] = value
            if value is not None:
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
        words = [_resolve(word, self.scope, defer=True) for word in words]
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
        owner_word = _shell_word(self.lex(owner)) if owner else "''"
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
                if not os.path.lexists(path) or _ignored(
                    translator.ignore, relative, path.is_dir()
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
) -> Definition:
    """Translate a Dockerfile into Apptainer build stages.

    Raises ``DockerfileUnsupported``, naming the line, for constructs the
    translation cannot reproduce, before anything is built.
    """
    try:
        parsed = parse_dockerfile(dockerfile.read_text())
        translator = _Translator(
            parsed,
            context_dir,
            _parse_build_args(build_args),
            machine or platform.machine(),
            os.environ if environ is None else environ,
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

# _ursa_copy OWNER MODE DEST SOURCE...: copy staged sources as COPY does.
_ursa_copy() {
    _ursa_owner=${1:-0:0}
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
        if ! chown -hR "$_ursa_owner" "$_ursa_src"; then
            echo "ursa: warning: cannot chown to $_ursa_owner" >&2
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
