import io
import json
import os
import subprocess
import tarfile
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from textwrap import dedent

import pytest

pytest.importorskip("pathspec")

from ursa.integrations import harbor_dockerfile  # noqa: E402
from ursa.integrations.harbor_dockerfile import (  # noqa: E402
    ArchiveSource,
    ContextSource,
    Definition,
    DockerfileUnsupported,
    DockerIgnore,
    ScriptSource,
    UrlSource,
    dockerfile_workdir,
    parse_dockerfile,
    translate_dockerfile,
)

SCICODE = """\
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \\
    build-essential curl git && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install scientific computing dependencies
RUN pip install --no-cache-dir \\
    numpy scipy matplotlib sympy h5py

ENV PYTHONPATH=/app:/tests
ENV TEST_DIR=/tests
"""


def _translate(
    tmp_path: Path,
    dockerfile: str,
    files: dict[str, str] | None = None,
    **kwargs,
) -> Definition:
    context = tmp_path / "context"
    context.mkdir(exist_ok=True)
    for name, content in (files or {}).items():
        path = context / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (context / "Dockerfile").write_text(dockerfile)
    kwargs.setdefault("machine", "x86_64")
    kwargs.setdefault("environ", {})
    return translate_dockerfile(context / "Dockerfile", context, **kwargs)


def _steps(definition: Definition, stage: int = -1) -> str:
    """Return a stage's script without its fixed setup and step headers."""
    body = definition.stages[stage].body or ""
    return "".join(
        line + "\n"
        for line in body.splitlines()
        if not line.startswith(("_ursa_step ", "_ursa_private_tmp"))
    )


def _unsupported(tmp_path: Path, dockerfile: str, **kwargs) -> str:
    with pytest.raises(DockerfileUnsupported) as raised:
        _translate(tmp_path, dockerfile, **kwargs)
    return str(raised.value)


def test_scicode_dockerfile_translates_to_fixed_definition(tmp_path: Path):
    definition = _translate(tmp_path, SCICODE)

    assert definition.render(Path("/stage")) == dedent("""\
        Bootstrap: docker
        From: python:3.11-slim

        %setup
            mkdir -p "$APPTAINER_ROOTFS/.ursa-build"
            cp -a /stage/s0/. "$APPTAINER_ROOTFS/.ursa-build/"

        %post
            /bin/sh /.ursa-build/build.sh
            rm -rf /.ursa-build
        """)
    assert definition.stages[0].body == dedent("""\
        _ursa_private_tmp
        _ursa_step 3 'RUN apt-get update && apt-get install -y     build-essential curl git &&     rm -rf /var/lib/apt/lists/*'
        _ursa_run /bin/sh -c 'apt-get update && apt-get install -y     build-essential curl git &&     rm -rf /var/lib/apt/lists/*'
        _ursa_step 7 'WORKDIR /app'
        _ursa_workdir /app
        _ursa_step 10 'RUN pip install --no-cache-dir     numpy scipy matplotlib sympy h5py'
        _ursa_run /bin/sh -c 'pip install --no-cache-dir     numpy scipy matplotlib sympy h5py'
        _ursa_step 13 'ENV PYTHONPATH=/app:/tests'
        export PYTHONPATH=/app:/tests
        _ursa_step 14 'ENV TEST_DIR=/tests'
        export TEST_DIR=/tests
        _ursa_save_env PYTHONPATH TEST_DIR
        """)  # noqa: E501
    script = definition.stages[0].script or ""
    assert script.startswith("#!/bin/sh\n")
    assert script.endswith(definition.stages[0].body or "")


def test_directives_continuations_and_comments(tmp_path: Path):
    definition = _translate(
        tmp_path,
        "# syntax=docker/dockerfile:1\n"
        "# escape=`\n"
        "FROM ubuntu:24.04\n"
        "RUN echo a `\n"
        "# a comment inside the continuation\n"
        "\n"
        "    && echo C:\\path\n",
    )

    assert _steps(definition) == (
        "_ursa_run /bin/sh -c 'echo a     && echo C:\\path'\n"
    )


def test_backslash_continuation_keeps_following_indentation():
    parsed = parse_dockerfile("FROM a\nRUN x \\  \n  y\n")

    assert parsed.instructions[1].args == "x   y"
    assert parsed.instructions[1].line == 2


def test_heredoc_only_run_is_a_script(tmp_path: Path):
    definition = _translate(
        tmp_path,
        "FROM ubuntu:24.04\n"
        "RUN <<-EOF\n"
        "\techo one\n"
        "\techo two\n"
        "\tEOF\n"
        "RUN <<EOF\n"
        "#!/usr/bin/env python3\n"
        "print('hi')\n"
        "EOF\n",
    )

    assert _steps(definition) == (
        "_ursa_run /bin/sh -c 'echo one\necho two\n'\n"
        "_ursa_run /.ursa-build/run-6\n"
    )
    assert definition.stages[0].staged == {
        "run-6": ScriptSource("#!/usr/bin/env python3\nprint('hi')\n")
    }


def test_heredoc_with_command_is_passed_to_the_shell(tmp_path: Path):
    definition = _translate(
        tmp_path,
        "FROM python:3.11-slim\n"
        "RUN python - <<'PY' && echo done\n"
        "print('$HOME')\n"
        "PY\n",
    )

    assert _steps(definition) == (
        "_ursa_run /bin/sh -c 'python - <<'\"'\"'PY'\"'\"' && echo done\n"
        "print('\"'\"'$HOME'\"'\"')\n"
        "PY\n"
        "'\n"
    )


def test_global_args_build_args_and_platform_args(tmp_path: Path):
    definition = _translate(
        tmp_path,
        "ARG BASE=ubuntu\n"
        "ARG TAG=22.04\n"
        "FROM ${BASE}:${TAG:-latest}\n"
        "ARG TAG\n"
        "ARG TARGETARCH\n"
        "ARG EMPTY\n"
        "ARG LOCAL=x-${TAG}\n"
        "RUN echo $TAG\n",
        build_args=("TAG=24.04",),
    )

    assert definition.stages[0].image == "ubuntu:24.04"
    assert _steps(definition) == (
        "export TAG=24.04\n"
        "export TARGETARCH=amd64\n"
        "export LOCAL=x-24.04\n"
        "_ursa_run /bin/sh -c 'echo $TAG'\n"
    )


def test_env_wins_over_arg_of_the_same_name(tmp_path: Path):
    definition = _translate(
        tmp_path, "FROM ubuntu\nENV MODE=env\nARG MODE=arg\nWORKDIR /$MODE\n"
    )

    assert _steps(definition) == (
        "export MODE=env\n_ursa_workdir /env\n_ursa_save_env MODE\n"
    )


def test_variables_expand_against_the_base_image_environment(
    tmp_path: Path,
):
    config = {
        "WorkingDir": "",
        "Env": ["PATH=/usr/local/bin:/usr/bin", "PYTHON_VERSION=3.11.9"],
    }
    dockerfile = (
        "FROM python:3.11-slim\n"
        "ARG PYTHON_VERSION\n"
        "ENV V=py$PYTHON_VERSION H=$HOME/bin PATH=$HOME/.cargo/bin:$PATH\n"
    )

    known = _translate(
        tmp_path, dockerfile, image_config=lambda image, machine: config
    )
    unknown = _translate(tmp_path, dockerfile)

    # As in Docker, an ARG without a value leaves the image's value, and
    # HOME, which images seldom set, is empty rather than the build's.
    assert _steps(known) == dedent("""\
        export V=py3.11.9 H=/bin PATH=/.cargo/bin:/usr/local/bin:/usr/bin
        _ursa_save_env V H PATH
        """)
    assert _steps(unknown) == dedent("""\
        export V=py"${PYTHON_VERSION}" H="${HOME}"/bin PATH="${HOME}"/.cargo/bin:"${PATH}"
        _ursa_save_env V H PATH
        """)  # noqa: E501


def test_proxy_args_come_from_build_args_or_the_host(tmp_path: Path):
    from_host = _translate(
        tmp_path,
        "FROM ubuntu\nRUN true\n",
        environ={"https_proxy": "http://proxy:3128", "HOME": "/home/me"},
    )
    from_args = _translate(
        tmp_path,
        "FROM ubuntu\nRUN true\n",
        build_args=("HTTP_PROXY=http://other:8080",),
        environ={"https_proxy": "http://proxy:3128"},
    )

    run = "_ursa_run /bin/sh -c true\n"
    assert _steps(from_host) == "export https_proxy=http://proxy:3128\n" + run
    assert _steps(from_args) == "export HTTP_PROXY=http://other:8080\n" + run


def test_env_forms_and_deferred_variables(tmp_path: Path):
    definition = _translate(
        tmp_path,
        "FROM ubuntu\n"
        'ENV PATH="/root/.cargo/bin:$PATH" A=1\n'
        "ENV LEGACY a value  with spaces\n"
        "ENV A=2 B=$A C='$A' D=${UNSET:-fallback} E=${HOME:+set}\n",
    )

    assert _steps(definition) == dedent("""\
        export PATH=/root/.cargo/bin:"${PATH}" A=1
        export LEGACY='a value  with spaces'
        export A=2 B=1 C='$A' D=""${UNSET:-fallback}"" E=""${HOME:+set}""
        _ursa_save_env PATH A LEGACY B C D E
        """).replace('""', '"')


def test_from_forms(tmp_path: Path):
    tagged = _translate(tmp_path, "FROM --platform=linux/amd64 a/b:1 AS Base\n")
    pinned = _translate(tmp_path, "FROM python@sha256:abc as x\n")
    arm = _translate(
        tmp_path,
        "FROM --platform=$BUILDPLATFORM ubuntu\n",
        machine="aarch64",
    )

    assert tagged.stages[0].image == "a/b:1"
    assert pinned.stages[0].image == "python@sha256:abc"
    assert arm.stages[0].image == "ubuntu"


def test_run_forms_and_shell(tmp_path: Path):
    definition = _translate(
        tmp_path,
        "FROM ubuntu\n"
        'RUN ["/bin/echo", "exec form", "$HOME"]\n'
        'SHELL ["/bin/bash", "-lc"]\n'
        "RUN echo $0\n"
        "RUN [not json\n"
        "RUN --mount=type=cache,target=/root/.cache --network=host make\n",
    )

    assert _steps(definition) == dedent("""\
        _ursa_run /bin/echo 'exec form' '$HOME'
        _ursa_run /bin/bash -lc 'echo $0'
        _ursa_run /bin/bash -lc '[not json'
        _ursa_run /bin/bash -lc make
        """)
    assert any("--mount" in note for note in definition.notes)


def test_workdir_user_and_ignored_instructions(tmp_path: Path):
    definition = _translate(
        tmp_path,
        "FROM ubuntu\n"
        "ENV APP=/srv\n"
        "WORKDIR $APP\n"
        "WORKDIR sub\n"
        "WORKDIR $HOME/x\n"
        "USER agent:agent\n"
        "LABEL a=b\n"
        "EXPOSE 80\n"
        "VOLUME /data\n"
        "HEALTHCHECK CMD true\n"
        "STOPSIGNAL SIGTERM\n"
        "ONBUILD RUN true\n"
        'CMD ["sleep", "1"]\n'
        "ENTRYPOINT /bin/sh\n",
    )

    assert _steps(definition) == dedent("""\
        export APP=/srv
        _ursa_workdir /srv
        _ursa_workdir sub
        _ursa_workdir "${HOME}"/x
        _ursa_user=agent:agent
        _ursa_save_env APP
        """)
    assert definition.warnings == ["line 12: ONBUILD is ignored"]
    assert any(
        "final USER agent:agent is not applied" in note
        for note in definition.notes
    )
    assert sum("is ignored" in note for note in definition.notes) == 7


def test_steps_start_in_the_base_image_workdir_and_user(tmp_path: Path):
    configs = {
        "swebench/sweb.eval:1": {"WorkingDir": "/testbed", "User": "dev"},
        "ubuntu": {"WorkingDir": "", "User": ""},
    }
    lookups = []

    def config(image: str, machine: str) -> dict[str, str] | None:
        lookups.append((image, machine))
        return configs.get(image)

    definition = _translate(
        tmp_path,
        "FROM ubuntu AS plain\n"
        "RUN pwd\n"
        "FROM private.example/base AS unknown\n"
        "RUN pwd\n"
        "FROM swebench/sweb.eval:1\n"
        "COPY --from=plain /etc/hostname ./\n"
        "COPY --from=unknown /etc/hosts ./\n"
        "RUN git add -A\n"
        "WORKDIR sub\n",
        image_config=config,
    )

    assert lookups == [
        ("swebench/sweb.eval:1", "x86_64"),
        ("ubuntu", "x86_64"),
        ("private.example/base", "x86_64"),
    ]
    assert _steps(definition, 0) == "_ursa_run /bin/sh -c pwd\n"
    assert _steps(definition, 1) == "_ursa_run /bin/sh -c pwd\n"
    assert _steps(definition).startswith(
        "_ursa_workdir /testbed\n_ursa_user=dev\n"
    )
    assert _steps(definition).endswith(
        "_ursa_run /bin/sh -c 'git add -A'\n_ursa_workdir sub\n"
    )
    assert definition.warnings == [
        "line 3: cannot read the config of private.example/base; build "
        "steps start in / as root instead of its WORKDIR and USER"
    ]
    assert any(
        "final USER dev is not applied" in note for note in definition.notes
    )


def test_stage_without_steps_has_no_post(tmp_path: Path):
    definition = _translate(
        tmp_path,
        "FROM --platform=linux/amd64 fhir@sha256:0 AS fhir\n"
        "LABEL a=b\n"
        "FROM python:3.11-slim\n"
        "COPY --from=fhir /app/main.war /app/main.war\n",
        environ={"https_proxy": "http://proxy:3128"},
    )

    assert definition.stages[0].body is None
    assert definition.render(Path("/stage")).startswith(
        "Bootstrap: docker\nFrom: fhir@sha256:0\nStage: s0\n\n"
        "Bootstrap: docker\nFrom: python:3.11-slim\nStage: s1\n"
    )
    assert _translate(tmp_path, "FROM a\nCMD x\n").stages[0].body is None


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("ubuntu", ("registry-1.docker.io", "library/ubuntu", "latest")),
        ("user/app:1", ("registry-1.docker.io", "user/app", "1")),
        ("docker.io/user/app", ("registry-1.docker.io", "user/app", "latest")),
        (
            "ghcr.io/o/r/n:t@sha256:ab",
            ("ghcr.io", "o/r/n", "sha256:ab"),
        ),
        ("localhost:5000/app:2", ("localhost:5000", "app", "2")),
        ("localhost/app", ("localhost", "app", "latest")),
    ],
)
def test_image_reference(image: str, expected: tuple[str, str, str]):
    assert harbor_dockerfile._image_reference(image) == expected


def _headers(challenge: str | None = None) -> Message:
    headers = Message()
    if challenge is not None:
        headers["WWW-Authenticate"] = challenge
    return headers


class _Registry:
    """A registry that wants an anonymous token for every request."""

    def __init__(self, documents: dict[str, object]) -> None:
        self.documents = documents
        self.requests: list[tuple[str, str | None]] = []

    def open(self, request, timeout):
        if isinstance(request, str):
            assert request == (
                "https://auth.example/token?service=reg&"
                "scope=repository%3Alibrary%2Fubuntu%3Apull"
            )
            return io.BytesIO(json.dumps({"token": "t"}).encode())
        url = request.full_url
        auth = request.get_header("Authorization")
        self.requests.append((url, auth))
        if auth != "Bearer t":
            raise urllib.error.HTTPError(
                url,
                401,
                "Unauthorized",
                _headers(
                    'Bearer realm="https://auth.example/token",service="reg",'
                    'scope="repository:library/ubuntu:pull"'
                ),
                None,
            )
        if url not in self.documents:
            raise urllib.error.HTTPError(
                url, 404, "Not Found", _headers(), None
            )
        return io.BytesIO(json.dumps(self.documents[url]).encode())


def test_image_config_reads_the_platform_config(
    monkeypatch: pytest.MonkeyPatch,
):
    base = "https://registry-1.docker.io/v2/library/ubuntu"
    registry = _Registry({
        f"{base}/manifests/24.04": {
            "manifests": [
                {
                    "digest": "sha256:arm",
                    "platform": {"os": "linux", "architecture": "arm64"},
                },
                {
                    "digest": "sha256:att",
                    "platform": {"os": "unknown", "architecture": "unknown"},
                },
                {
                    "digest": "sha256:amd",
                    "platform": {"os": "linux", "architecture": "amd64"},
                },
            ]
        },
        f"{base}/manifests/sha256:amd": {"config": {"digest": "sha256:cfg"}},
        f"{base}/blobs/sha256:cfg": {"config": {"WorkingDir": "/app"}},
    })
    monkeypatch.setattr(harbor_dockerfile, "_image_configs", {})
    monkeypatch.setattr(
        harbor_dockerfile.urllib.request, "build_opener", lambda *_: registry
    )

    assert harbor_dockerfile.image_config("ubuntu:24.04", "x86_64") == {
        "WorkingDir": "/app"
    }
    assert [url for url, auth in registry.requests if auth] == [
        f"{base}/manifests/24.04",
        f"{base}/manifests/sha256:amd",
        f"{base}/blobs/sha256:cfg",
    ]
    # Cached per process; failures are not cached.
    assert harbor_dockerfile.image_config("ubuntu:24.04", "x86_64") == {
        "WorkingDir": "/app"
    }
    assert harbor_dockerfile.image_config("ubuntu:24.04", "aarch64") is None
    assert harbor_dockerfile.image_config("ubuntu:missing", "x86_64") is None
    assert set(harbor_dockerfile._image_configs) == {("ubuntu:24.04", "x86_64")}


def test_registry_redirect_drops_the_token_for_another_host():
    handler = harbor_dockerfile._RegistryRedirect()
    request = urllib.request.Request(
        "https://ghcr.io/v2/o/r/blobs/sha256:0",
        headers={"Authorization": "Bearer t"},
    )
    same = handler.redirect_request(
        request, None, 307, "", _headers(), "https://ghcr.io/v2/other"
    )
    other = handler.redirect_request(
        request, None, 307, "", _headers(), "https://cdn.example/blob"
    )

    assert same.get_header("Authorization") == "Bearer t"
    assert other.get_header("Authorization") is None


@pytest.mark.parametrize(
    ("dockerfile", "kwargs", "expected"),
    [
        ("FROM a\nWORKDIR /app\nWORKDIR sub/../x\n", {}, "/app/x"),
        ("FROM a\nWORKDIR /a \\\n  /b\n", {}, "/a   /b"),
        ("FROM a\nENV D=/srv\nWORKDIR $D/app\n", {}, "/srv/app"),
        ("ARG D=/g\nFROM a\nARG D\nWORKDIR $D\n", {}, "/g"),
        ("FROM a\nARG D=/x\nWORKDIR $D\n", {"build_args": ("D=/y",)}, "/y"),
        (
            "FROM a AS one\nWORKDIR /one\nFROM a\nWORKDIR /two\n",
            {"target": "ONE"},
            "/one",
        ),
        (
            "FROM a AS one\nWORKDIR /one\nFROM one\nWORKDIR rel\n",
            {},
            "/one/rel",
        ),
        ("FROM a\nRUN <<EOF\nWORKDIR /no\nEOF\n", {}, "/"),
        ("FROM a\nENV D=/env\nARG D=/arg\nWORKDIR $D\n", {}, "/env"),
        ("FROM a\nWORKDIR sub\n", {}, "/sub"),
    ],
)
def test_dockerfile_workdir(dockerfile: str, kwargs: dict, expected: str):
    assert dockerfile_workdir(dockerfile, **kwargs) == expected


def test_dockerfile_workdir_starts_in_the_base_image_workdir():
    config = {"WorkingDir": "/app", "Env": ["HOME=/home/app", "V=base"]}
    lookups = []

    def lookup(image: str, machine: str) -> dict | None:
        lookups.append(image)
        return config if image == "a" else None

    def workdir(dockerfile: str, warnings: list[str] | None = None) -> str:
        return dockerfile_workdir(
            dockerfile, image_config=lookup, warnings=warnings
        )

    assert workdir("FROM a\nWORKDIR /abs\nWORKDIR rel\n") == "/abs/rel"
    assert workdir("FROM a\nRUN true\n") == "/"
    assert not lookups
    assert workdir("FROM a\nWORKDIR sub\n") == "/app/sub"
    assert workdir("FROM a\nWORKDIR $HOME/x\n") == "/home/app/x"
    assert workdir("FROM a\nARG V\nWORKDIR /$V\n") == "/base"
    warnings: list[str] = []
    assert workdir("FROM b\nWORKDIR sub\n", warnings) == "/sub"
    assert warnings == [
        "cannot read the config of b; its relative WORKDIR starts in / "
        "instead of the image's WORKDIR"
    ]
    with pytest.raises(ValueError, match="WORKDIR: \\$HOME"):
        workdir("FROM b\nWORKDIR $HOME\n")


def test_dockerfile_workdir_rejects_base_image_variables():
    with pytest.raises(ValueError, match="WORKDIR: \\$HOME/x"):
        dockerfile_workdir("FROM a\nWORKDIR $HOME/x\n")


def test_copy_sources_and_destinations(tmp_path: Path):
    definition = _translate(
        tmp_path,
        "FROM ubuntu\n"
        "WORKDIR /app\n"
        "COPY src/ ./code/\n"
        "COPY *.cfg [ab].txt /etc/myapp/\n"
        "COPY one.cfg .\n"
        'COPY ["one.cfg", "/dest with space"]\n'
        "COPY --chown=agent:agent --chmod=0755 one.cfg /usr/local/bin/tool\n"
        "COPY . /all/\n",
        files={
            "src/app.py": "",
            "one.cfg": "",
            "two.cfg": "",
            "a.txt": "",
            "c.txt": "",
            "secret/key": "",
            ".dockerignore": "secret\n",
        },
    )

    assert _steps(definition) == dedent("""\
        _ursa_workdir /app
        _ursa_copy '' '' ./code/ /.ursa-build/copy-3/0/src
        _ursa_copy '' '' /etc/myapp/ /.ursa-build/copy-4/0/one.cfg /.ursa-build/copy-4/1/two.cfg /.ursa-build/copy-4/2/a.txt
        _ursa_copy '' '' . /.ursa-build/copy-5/0/one.cfg
        _ursa_copy '' '' '/dest with space' /.ursa-build/copy-6/0/one.cfg
        _ursa_copy agent:agent 0755 /usr/local/bin/tool /.ursa-build/copy-7/0/one.cfg
        _ursa_copy '' '' /all/ /.ursa-build/copy-8/0/context
        """)  # noqa: E501
    assert definition.stages[0].staged["copy-8/0/context"] == ContextSource("")


def test_write_stages_context_files(tmp_path: Path):
    context = tmp_path / "context"
    context.mkdir()
    os.symlink("target.txt", context / "link.txt")
    definition = _translate(
        tmp_path,
        "FROM ubuntu\nCOPY . /all/\nCOPY link.txt /one\n",
        files={
            "dir/file.txt": "data",
            "target.txt": "target",
            "secret/key": "key",
            ".dockerignore": "secret\n",
        },
    )
    (context / "dir" / "file.txt").chmod(0o640)
    staging = tmp_path / "staging"
    staging.mkdir()

    path = definition.write(staging)

    assert path == staging / "build.def"
    assert "cp -a" in path.read_text()
    root = staging / "s0"
    assert os.access(root / "build.sh", os.X_OK)
    copied = root / "copy-2" / "0" / "context"
    assert (copied / "dir" / "file.txt").read_text() == "data"
    assert (copied / "dir" / "file.txt").stat().st_mode & 0o777 == 0o640
    # A symlink inside a copied directory stays a link, as in Docker ...
    assert os.readlink(copied / "link.txt") == "target.txt"
    assert not (copied / "secret").exists()
    # ... but a symlink named as the source is followed.
    single = root / "copy-3" / "0" / "link.txt"
    assert not single.is_symlink()
    assert single.read_text() == "target"


@pytest.mark.parametrize(
    ("patterns", "path", "ignored"),
    [
        ("*.json", "config.json", True),
        ("*.json", "src/config.json", False),
        ("**/*.json", "src/deep/config.json", True),
        ("/data", "data/x/y", True),
        ("data", "src/data", False),
        ("src/*", "src/a/b", True),
        ("a?c", "abc", True),
        ("a?c", "a/c", False),
        ("**", "anything/at/all", True),
        ("data\n!data/keep", "data/keep", False),
        ("data\n!data/keep", "data/drop", True),
        ("!x\nx", "x", True),
        ("# x\nx/../y", "y", True),
        ("file.(1)", "file.(1)", True),
    ],
)
def test_dockerignore_matches_as_docker_does(
    patterns: str, path: str, ignored: bool
):
    assert DockerIgnore(patterns.splitlines()).ignored(path) is ignored


def test_copy_applies_dockerignore_as_docker_does(tmp_path: Path):
    definition = _translate(
        tmp_path,
        "FROM ubuntu\nCOPY . /app/\nCOPY data /data/\n",
        files={
            "top.json": "",
            "src/config.json": "",
            "data/drop.txt": "",
            "data/sub/keep.txt": "",
            ".dockerignore": "*.json\ndata\n!data/sub/keep.txt\n",
        },
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    definition.write(staging)

    copied = staging / "s0" / "copy-2" / "0" / "context"
    files = sorted(
        path.relative_to(copied).as_posix()
        for path in copied.rglob("*")
        if path.is_file()
    )
    assert files == [
        ".dockerignore",
        "Dockerfile",
        "data/sub/keep.txt",
        "src/config.json",
    ]
    data = staging / "s0" / "copy-3" / "0" / "data"
    assert [p.name for p in data.rglob("*") if p.is_file()] == ["keep.txt"]


def test_add_extracts_local_archives_and_downloads_urls(tmp_path: Path):
    context = tmp_path / "context"
    content = tmp_path / "content"
    content.mkdir()
    (content / "inside.txt").write_text("from tar")
    context.mkdir()
    with tarfile.open(context / "input.tar.bz2", "w:bz2") as archive:
        archive.add(content / "inside.txt", "inside.txt")
    definition = _translate(
        tmp_path,
        "FROM ubuntu\n"
        "ADD input.tar.bz2 /app\n"
        "ADD https://example.com/files/tool.sh /usr/local/bin/\n",
    )
    downloads = []

    def download(url: str, target: Path) -> None:
        downloads.append(url)
        target.write_text("#!/bin/sh\n")

    assert _steps(definition) == dedent("""\
        _ursa_copy '' '' /app /.ursa-build/add-2/0/input.tar.bz2
        _ursa_copy '' '' /usr/local/bin/ /.ursa-build/add-3/0/tool.sh
        """)
    assert definition.stages[0].staged == {
        "add-2/0/input.tar.bz2": ArchiveSource("input.tar.bz2"),
        "add-3/0/tool.sh": UrlSource("https://example.com/files/tool.sh"),
    }
    staging = tmp_path / "staging"
    staging.mkdir()
    definition.write(staging, download=download)

    extracted = staging / "s0" / "add-2" / "0" / "input.tar.bz2"
    assert (extracted / "inside.txt").read_text() == "from tar"
    downloaded = staging / "s0" / "add-3" / "0" / "tool.sh"
    assert downloaded.stat().st_mode & 0o777 == 0o600
    assert downloads == ["https://example.com/files/tool.sh"]


def test_add_keeps_archive_owners_and_modes(tmp_path: Path):
    context = tmp_path / "context"
    context.mkdir()
    with tarfile.open(context / "tree.tar.gz", "w:gz") as archive:
        for name, kind, mode, owner in [
            ("shared", tarfile.DIRTYPE, 0o2775, 1000),
            ("shared/helper", tarfile.REGTYPE, 0o4755, 0),
        ]:
            info = tarfile.TarInfo(name)
            info.type, info.mode, info.uid, info.gid = kind, mode, owner, owner
            archive.addfile(info, io.BytesIO(b""))
    definition = _translate(tmp_path, "FROM ubuntu\nADD tree.tar.gz /opt/\n")
    staging = tmp_path / "staging"
    staging.mkdir()
    definition.write(staging)

    staged = staging / "s0" / "add-2" / "0" / "tree.tar.gz"
    assert (staged / "shared" / "helper").stat().st_mode & 0o7777 == 0o4755
    assert Path(f"{staged}.ursa-members").read_text() == (
        "_ursa_member 1000:1000 2775 ./shared\n"
        "_ursa_member 0:0 4755 ./shared/helper\n"
    )


def test_build_script_copy_sets_owners_as_docker_does(tmp_path: Path):
    staged = tmp_path / "staged"
    (staged / "tree").mkdir(parents=True)
    (staged / "tree" / "tool").write_text("")
    (tmp_path / "staged" / "tree.ursa-members").write_text(
        "_ursa_member 7:8 4750 ./tool\n"
    )
    (staged / "file").write_text("")
    log = tmp_path / "chown.log"
    copies = [
        ("''", "file"),
        ("1000", "file"),
        ("agent", "file"),
        ("agent:staff", "file"),
        ("-", "file"),
        ("''", "tree"),
        ("9:9", "tree"),
    ]

    result = _shell(
        f"""
        chown() {{ echo "$*" >> {log}; }}
        id() {{ [ "$2" = agent ] && echo 1234; }}
        """
        + "".join(
            f"_ursa_copy {owner} '' {tmp_path}/out/ {staged}/{source}\n"
            for owner, source in copies
        ),
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == [
        f"-hR 0:0 {staged}/file",
        # A user without a group also names the group, by the user's UID.
        f"-hR 1000:1000 {staged}/file",
        f"-hR agent:1234 {staged}/file",
        f"-hR agent:staff {staged}/file",
        # An archive's members keep their owners unless --chown is given.
        f"-hR 0:0 {staged}/tree",
        "-h 7:8 ./tool",
        f"-hR 9:9 {staged}/tree",
    ]
    assert (tmp_path / "out" / "tool").stat().st_mode & 0o7777 == 0o4750


def test_multi_stage_copies_from_stages_and_images(tmp_path: Path):
    definition = _translate(
        tmp_path,
        "FROM golang:1 AS build\n"
        "RUN go build -o /out/app\n"
        "FROM alpine AS unused\n"
        "RUN false\n"
        "FROM debian AS final\n"
        "COPY --from=build /out/app /usr/local/bin/\n"
        "COPY --from=0 /out /opt/out\n"
        "COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/\n"
        "FROM final AS later\n",
        target="final",
    )

    assert [stage.name for stage in definition.stages] == ["s0", "x0", "s2"]
    assert definition.render(Path("/stage")) == dedent("""\
        Bootstrap: docker
        From: golang:1
        Stage: s0

        %setup
            mkdir -p "$APPTAINER_ROOTFS/.ursa-build"
            cp -a /stage/s0/. "$APPTAINER_ROOTFS/.ursa-build/"

        %post
            /bin/sh /.ursa-build/build.sh
            rm -rf /.ursa-build

        Bootstrap: docker
        From: ghcr.io/astral-sh/uv:0.5
        Stage: x0

        Bootstrap: docker
        From: debian
        Stage: s2

        %files from s0
            /out/app /.ursa-build/copy-6/0/app
            /out /.ursa-build/copy-7/0/out

        %files from x0
            /uv /.ursa-build/copy-8/0/uv
            /uvx /.ursa-build/copy-8/1/uvx

        %setup
            mkdir -p "$APPTAINER_ROOTFS/.ursa-build"
            cp -a /stage/s2/. "$APPTAINER_ROOTFS/.ursa-build/"

        %post
            /bin/sh /.ursa-build/build.sh
            rm -rf /.ursa-build
        """)
    assert _steps(definition) == dedent("""\
        _ursa_copy - '' /usr/local/bin/ /.ursa-build/copy-6/0/app
        _ursa_copy - '' /opt/out /.ursa-build/copy-7/0/out
        _ursa_copy - '' /bin/ /.ursa-build/copy-8/0/uv /.ursa-build/copy-8/1/uvx
        """)  # noqa: E501
    # Only the target stage records ENV for run time.
    assert "_ursa_save_env" not in (definition.stages[0].body or "")


@pytest.mark.parametrize(
    ("dockerfile", "kwargs", "expected"),
    [
        (
            "FROM a\nRUN --mount=type=bind,target=/x make\n",
            {},
            "line 2: RUN --mount=type=bind,target=/x is not supported",
        ),
        ("FROM a\nRUN --network=none make\n", {}, "line 2: RUN --network"),
        ("FROM a\nRUN --security=insecure make\n", {}, "line 2: RUN --sec"),
        ("FROM a\nCOPY --parents a /b/\n", {}, "line 2: COPY --parents"),
        ("FROM a\nCOPY --exclude=x . /b/\n", {}, "line 2: COPY --exclude"),
        ("FROM a\nADD --checksum=sha256:0 a /b\n", {}, "line 2: ADD --check"),
        ("FROM a\nADD --from=b a /b\n", {}, "line 2: ADD --from"),
        ("FROM a\nCOPY <<EOF /x\nhi\nEOF\n", {}, "line 2: COPY with a here"),
        ("FROM scratch\n", {}, "line 1: FROM scratch"),
        ("FROM a AS b\nFROM b\n", {}, "line 2: FROM an earlier build"),
        ("FROM {{image}}\n", {}, "line 1: cannot bootstrap"),
        ("FROM --platform=linux/arm64 a\n", {}, "line 1: --platform"),
        ("FROM --platform=windows/amd64 a\n", {}, "line 1: --platform"),
        ("RUN true\nFROM a\n", {}, "line 1: only ARG may come before"),
        ("FROM a\n{PLACEHOLDER}\n", {}, "line 2: unknown instruction"),
        ("FROM a\nCOPY missing /x\n", {}, "line 2: source missing is not"),
        ("FROM a\nCOPY ../up /x\n", {}, "line 2: source ../up is outside"),
        ("FROM a\nCOPY *.none /x/\n", {}, "line 2: no source files match"),
        ("FROM a\nCOPY Dockerfile $SRC /x/\n", {}, "line 2: the source uses"),
        ("FROM a\nCOPY Dockerfile Dockerfile /x\n", {}, "ending in /"),
        ("FROM a\nCOPY --chmod=u+rwz Dockerfile /x\n", {}, "line 2: --chmod"),
        ("FROM a\nCOPY --from=b /a* /x/\n", {}, "line 2: COPY --from source"),
        ("FROM a\nENV my.var=1\n", {}, "line 2: 'my.var' is not a supported"),
        ("FROM a\nENV _ursa_wd=1\n", {}, "line 2: '_ursa_wd' is not"),
        ("FROM a\nWORKDIR ${D/a/b}\n", {}, "line 2: variable modifier"),
        ("FROM a\nWORKDIR 'x\n", {}, "line 2: unterminated single quote"),
        ("FROM a\nADD git@github.com:a/b.git /x\n", {}, "line 2: ADD git@"),
        ("FROM a\nADD https://h/r.git /x\n", {}, "line 2: ADD https://h/r.git"),
        ("FROM a\nADD https://h/ /x/\n", {}, "line 2: ADD https://h/ needs"),
        ("FROM a\nRUN <<EOF\nnever ends\n", {}, "line 2: heredoc EOF is not"),
        ("FROM a\n", {"target": "nope"}, "target stage 'nope' not found"),
        ("# just a comment\n", {}, "the Dockerfile has no FROM"),
    ],
)
def test_unsupported_constructs_name_the_line(
    tmp_path: Path, dockerfile: str, kwargs: dict, expected: str
):
    message = _unsupported(tmp_path, dockerfile, **kwargs)

    assert expected in message
    assert message.startswith(str(tmp_path / "context" / "Dockerfile"))


def test_copy_source_links_outside_context_are_rejected(tmp_path: Path):
    context = tmp_path / "context"
    context.mkdir()
    (tmp_path / "outside").write_text("x")
    os.symlink(tmp_path / "outside", context / "link")

    message = _unsupported(tmp_path, "FROM a\nCOPY link /x\n")

    assert "line 2: source link links outside the build context" in message


def _shell(script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    definition = _translate(tmp_path, "FROM a\nRUN true\n")
    prelude = definition.stages[0].script or ""
    prelude = prelude[: prelude.index("_ursa_private_tmp\n")]
    return subprocess.run(
        ["/bin/sh", "-c", prelude + script],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )


def test_build_script_copy_follows_docker_rules(tmp_path: Path):
    staged = tmp_path / "staged"
    (staged / "dir" / "sub").mkdir(parents=True)
    (staged / "dir" / "sub" / "a.txt").write_text("a")
    (staged / "dir" / ".hidden").write_text("h")
    (staged / "file.txt").write_text("f")
    root = tmp_path / "root"
    (root / "existing").mkdir(parents=True)

    result = _shell(
        f"""
        _ursa_workdir {root}/work
        _ursa_copy '' '' ./contents {staged}/dir
        _ursa_copy '' '' {root}/existing {staged}/file.txt
        _ursa_copy '' '' {root}/new/name.txt {staged}/file.txt
        _ursa_copy '' '' {root}/slash/ {staged}/file.txt
        _ursa_copy '' 600 . {staged}/file.txt
        """,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert (root / "work" / "contents" / "sub" / "a.txt").read_text() == "a"
    assert (root / "work" / "contents" / ".hidden").read_text() == "h"
    assert (root / "existing" / "file.txt").read_text() == "f"
    assert (root / "new" / "name.txt").read_text() == "f"
    assert (root / "slash" / "file.txt").read_text() == "f"
    assert (root / "work" / "file.txt").stat().st_mode & 0o777 == 0o600


def test_build_script_runs_steps_in_the_workdir(tmp_path: Path):
    result = _shell(
        f"""
        _ursa_workdir {tmp_path}/one
        _ursa_workdir two
        _ursa_run /bin/sh -c 'pwd > ../../out.txt; cd /'
        _ursa_run sh -c 'pwd >> {tmp_path}/out.txt'
        """,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    workdir = (tmp_path / "one" / "two").resolve()
    assert (tmp_path / "out.txt").read_text().split() == [
        str(workdir),
        str(workdir),
    ]


def test_build_script_saves_quoted_env_without_build_path(tmp_path: Path):
    environment = tmp_path / "91-environment.sh"
    result = _shell(
        f"""
        export APPTAINER_ENVIRONMENT={environment}
        export PATH=/opt/bin:/.singularity.d/libs:/usr/bin
        export QUOTED="it's \\$HOME and \\"x\\""
        _ursa_save_env PATH QUOTED
        """,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    loaded = subprocess.run(
        [
            "/bin/sh",
            "-c",
            f'. {environment}; printf \'%s\\n\' "$PATH" "$QUOTED"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert loaded.stdout.splitlines() == [
        "/opt/bin:/usr/bin",
        'it\'s $HOME and "x"',
    ]
