# Run URSA with Harbor

This example connects URSA's `ExecutionAgent` to
[Harbor](https://www.harborframework.com/docs/core-concepts) and includes
GPAW, Parthenon, and PyTorch installation tasks.

## Run a task

From this directory, install Python 3.12+, [uv](https://docs.astral.sh/uv/),
and Docker for Harbor's default environment. Set a provider credential, then:

```bash
export OPENAI_API_KEY=...
uv run harbor run \
  --path benchmark/tasks/install-pytorch \
  --agent ursa.integrations.harbor:UrsaHarborAgent \
  --agent-kwarg config="$PWD/ursa.yaml" \
  --model openai/openai:gpt-5.4-nano
```

A successful task reports reward `1`; inspect results with `uv run harbor view
jobs`. Use `--path benchmark/tasks` to run all included tasks.

To continue a completed trial locally, run:

```bash
uv run harbor trial handoff jobs/<job>/<trial>
```

Handoff imports the trial checkpoint into its recorded, existing URSA group and
opens the TUI; the task container's files are not copied into the workspace.

### Add a task

Start with Harbor's generator, then follow its task tutorial and reference for
the generated instruction, environment, task, and verifier files:

```bash
uv run harbor task init --tasks-dir benchmark/tasks my-org/my-task
```

The official [task tutorial](https://www.harborframework.com/docs/tasks/task-tutorial),
[format reference](https://www.harborframework.com/docs/tasks), and
[publishing guide](https://www.harborframework.com/docs/tasks/publishing) cover
authoring and sharing tasks.

### Configure URSA

Harbor models use `<inference-provider>/<model-provider>:<model-name>`, such
as `ollama/ollama:gemma4:latest`. The inference provider must exist in
`ursa.yaml`; its `model_provider` allows omitting the model-provider prefix.
`--model` replaces the configured model, and Harbor resolves the endpoint of a
custom inference provider from `<PROVIDER>_BASE_URL` (for example
`RESEARCH_BASE_URL` for `research/...`). Harbor
`[[environment.mcp_servers]]` entries are attached automatically; MCP HTTP
`headers` are not forwarded, so Harbor rejects tasks that set them. See the
[MCP task tutorial](https://www.harborframework.com/docs/tutorials/mcp-server-task).

Web tools follow Harbor's decision for each agent phase: they are on only on a
`public` network and off under `allowlist` and `no-network`. A `use_web` set in
the URSA config is kept on a `public` network, and the
`--agent-kwarg disable_web_search=true|false` kwarg replaces the network
decision, as for Harbor's own agents.

### Advanced installation

The adapter installs Python and URSA with `uv` inside each task container.
`ursa_install_spec` must be a [PEP 508 requirement](https://peps.python.org/pep-0508/)
for `ursa-ai`. It defaults to the host checkout or installed version.

| Source | `ursa_install_spec` example |
| --- | --- |
| Package version | `ursa-ai==0.17.3` |
| Git revision | `ursa-ai @ git+https://github.com/lanl/ursa.git@main` |
| Local checkout | `ursa-ai @ file:///absolute/path/to/ursa` |
| Local wheel | `ursa-ai @ file:///absolute/path/to/ursa_ai-0.17.3-py3-none-any.whl` |
| Local sdist | `ursa-ai @ file:///absolute/path/to/ursa_ai-0.17.3.tar.gz` |

Quote the whole kwarg, including the package name and `@` for URL references;
bare paths and bare Git URLs are not requirements. Use `Path(...).resolve().as_uri()`
to create a file URL with spaces or platform-specific paths encoded correctly.
Local sources are uploaded before installation; checkouts exclude Git-ignored
and common secret files. An implicit host checkout is built into a wheel first.

```bash
uv run harbor run \
  --path benchmark/tasks/install-pytorch \
  --agent ursa.integrations.harbor:UrsaHarborAgent \
  --agent-kwarg config="$PWD/ursa.yaml" \
  --model openai/openai:gpt-5.4-nano \
  --agent-kwarg 'ursa_install_spec=ursa-ai @ file:///absolute/path/to/ursa' \
  --agent-kwarg ursa_extras=image \
  --agent-kwarg 'extra_packages=["numpy", "scipy"]'
```

Requirement extras and environment markers are preserved; the `harbor` extra
is always added. `ursa_extras` adds extras, and `extra_packages` installs
additional requirements alongside URSA.

The agent's native `config` is the complete URSA configuration before Harbor's
model and MCP layer are applied.
Secret and keyring references resolve on the host; only generated environment
references reach the runner, never host config or keyring files. Trial output
includes `agent/ursa.log`, lifecycle records in `agent/ursa.jsonl` (including
chat token usage), and `agent/ursa_result.out`.

## Singularity environment

Use this environment for direct runs:

```bash
--env ursa.integrations.harbor.singularity:DockerfileSingularityEnvironment
```

It builds `environment/Dockerfile` into a SIF; no hand-written Apptainer
definition is needed. Compute nodes need `apptainer` or `singularity`, plus
Buildah, Podman, or Docker unless Apptainer builds the Dockerfile itself (see
[Dockerfile builds](#dockerfile-builds)). SIFs cache by build context in `$XDG_CACHE_HOME/ursa/harbor/sif`
or `~/.cache/ursa/harbor/sif`; concurrent builds share a lock. Set
`URSA_HARBOR_SIF_CACHE`, pass the environment constructor's
`singularity_image_cache_dir`, or use `prebuild --image-cache-dir` to choose a
shared cache.

As with Harbor's Docker backend, a task's prebuilt `[environment].docker_image`
is pulled instead of building the Dockerfile; Harbor's `--force-build` builds
the Dockerfile when one exists. A Compose `main.image` replaces that prebuilt
image, and without a prebuilt image it is pulled only when there is no
Dockerfile to build. A `docker_image` ending in `.sif` (with no `://` scheme)
is used directly as a local SIF file. Pulled images are cached by reference, so
a moving tag such as `latest` is fetched again only when refreshed:
`--environment-kwarg singularity_force_pull=true` re-pulls (and rebuilds) the
cached SIFs without turning a prebuilt-image task into a build.

### Dockerfile builds

The `singularity_builder` environment kwarg, or `URSA_HARBOR_SIF_BUILDER` when
the kwarg is unset, chooses how a Dockerfile becomes a SIF:

- `apptainer`: translate the Dockerfile into an Apptainer definition file and
  build it with `apptainer build --fakeroot` in the build context. This needs
  no other tools and no `/etc/subuid` entry, since Apptainer 1.1 and later
  fall back to a root-mapped user namespace plus the host's `fakeroot`
  command. It suits HPC hosts with an unprivileged (non-setuid) Apptainer and
  no `/etc/subuid`.
- `buildah`, `podman`, or `docker`: build an OCI image and convert it with
  `<runtime> build`. Docker needs a running daemon and socket access. Rootless
  Buildah and Podman run `RUN` steps only for users with `/etc/subuid` and
  `/etc/subgid` entries; without them every step fails, for example with
  "error setting supplemental groups list: operation not permitted".
- `auto` (default): when the runtime is Apptainer (also when installed under
  the `singularity` name), try `apptainer` first, then Buildah, Podman, and
  Docker. SingularityCE keeps only the OCI builders, since its fakeroot builds
  need subuid ranges as well. A Dockerfile the translation does not support
  goes straight to the OCI builders. A builder that fails for a host reason,
  such as unusable fakeroot or missing subuid ranges, is skipped for the rest
  of the process instead of failing again for every task.

An explicit builder is the only one tried, and it is an error when it is not
available. On a host where Buildah works, `singularity_builder=buildah` keeps
Docker's own build semantics. Definition and OCI builds are cached under
different keys, and the definition key includes the translator version, so
changing the builder rebuilds once; under `auto` a cached SIF from either
builder is reused, before any builder is looked up, so a SIF prebuilt on a
login node also serves compute nodes without a builder. When the Apptainer
build fails and an OCI builder succeeds, the Apptainer build log is kept next
to the SIF and named in a warning.

The translation replays the Dockerfile from a generated script in `%post`, so
`COPY` and `RUN` steps keep their order even though Apptainer copies `%files`
before `%post`; build context files are staged in `%setup` and removed at the
end of the build. It supports parser directives (`escape`), line continuations,
`RUN` heredocs, `ARG` (including before `FROM`, Compose build `args`, BuildKit's
platform arguments, and the predefined proxy arguments), `ENV`, `SHELL`,
`WORKDIR`, `USER` for later build steps, `COPY` and `ADD` from the context with
Docker's `.dockerignore` matching, `--chown` and `--chmod`, `ADD` of local tar
archives and `http(s)` URLs, and multi-stage builds with `COPY --from` a stage
or another image. As in Docker, variables expand against the base image's
environment and build steps start in its `WORKDIR` and as its `USER`; the
adapter reads the base image's config from its registry without credentials
(allow the registry hosts on egress-restricted hosts). When the config cannot be
read it warns, steps start in `/` as root, and the build is cached under its own
key until the registry answers. `CMD`, `ENTRYPOINT`, `LABEL`, `EXPOSE`,
`VOLUME`, `HEALTHCHECK`, `STOPSIGNAL`, and `ONBUILD` are ignored.

It rejects, naming the line, `RUN --mount` other than `type=cache` (ignored),
`RUN --network=none`, `COPY --parents` and `--exclude`, `COPY` and `ADD`
heredocs, `ADD` of Git repositories or with `--checksum`, globs in
`COPY --from`, `FROM scratch`, `FROM` an earlier stage, and variables in `COPY`
sources that only the base image defines. Under `auto` such a Dockerfile is
built by an OCI builder if one is installed. Other differences from Docker:

- The final `USER` is not applied; Harbor chooses the container user. Only root
  is mapped in the build namespace, so `RUN` after a non-root `USER`, and
  `--chown` to another user, run or copy as root while `fakeroot` reports the
  requested owner.
- The host's `fakeroot` command cannot load in images whose C library is older
  than the host's, or is musl, as in Alpine. The build is then retried with
  `--ignore-fakeroot-command`; package managers that drop privileges, such as
  `apt-get`, still fail in that mode, and the OCI builders are tried.
- Apptainer builds in `APPTAINER_TMPDIR` (default `/tmp`), which must be a local
  file system with room for the unpacked image; NFS and Lustre do not work for
  fakeroot builds.
- The contents of `ADD` URLs are not part of the cache key.

`singularity_fakeroot` defaults to `None`: fakeroot is enabled with Apptainer,
including Apptainer installed under the `singularity` name (detected from
`--version`), and disabled with SingularityCE. Passing
`--environment-kwarg singularity_fakeroot=true` or
`--environment-kwarg singularity_fakeroot=false` overrides this choice (`1`,
`0`, `yes`, and `no` are accepted too). With fakeroot enabled, each start first
checks that `--fakeroot` works and otherwise fails, before any overlay is
created, with the runtime's reason; later trials in the same process then fail
before building. Disabling fakeroot explicitly logs one warning: commands run
as the invoking host user, root-owned image files stay read-only, and tasks that
install system packages fail.

On HPC hosts with an unprivileged (non-setuid) Apptainer and no `/etc/subuid`,
Apptainer's fakeroot maps the invoking user to root and wraps each exec in the
`fakeroot(1)` command, whose `faked` daemon exits with that exec; a tmux server
or background job left behind would then hang on its next file operation. The
first exec in each instance detects this mode, and later execs share one
`faked` per instance instead. In this mode only, the start also writes
`/etc/apt/apt.conf.d/00ursa-fakeroot`, which keeps apt as root
(`APT::Sandbox::User "root"`, since apt cannot drop to `_apt`) and sets 5
immediate `Acquire` retries with a 20 s timeout, because apt 2.8 can otherwise
wait indefinitely on a delayed retry after a transient fetch failure. Where
the image has `/usr/bin/sudo`, a shim in `/usr/local/sbin` runs a root
caller's `sudo` command directly, or through `su` for another user, keeping
the fakeroot environment that sudo would strip; other callers and modes use
the image's sudo. Subuid-mapped fakeroot and SingularityCE keep the plain exec path.
Apptainer is preferred when both runtimes are installed. A failed build or task
command that reports requiring root under Singularity warns that switching to
Apptainer may help.

With Apptainer fakeroot, writable overlays are created by
`apptainer overlay create --fakeroot --sparse` so their ownership matches the
container's user namespace. Overlay size follows `[environment].storage_mb`
(default 10240 MiB; the image is sparse, so this only caps what a task can
write); Apptainer requires at least 64 MiB. Singularity and runs without
fakeroot retain the host-created ext3 overlay.

Over an Apptainer overlay the image's `/etc/localtime` is kept
(`--no-mount /etc/localtime`), so packages such as tzdata can replace it. With
fakeroot the image's `/tmp` and `/var/tmp` are kept in the overlay too
(`--no-mount tmp`) and the task's workdir is not bind-mounted, but created in
the overlay when the image lacks it, so renames between them and the rest of
the filesystem work as under Docker. Otherwise the workdir is a writable bind,
and a Harbor or Compose mount nested inside it, such as a data file under the
workdir, gets a mount point created in the bind's host directory.

Commands run with the image's `/root` as `HOME` when fakeroot has an overlay.
Without fakeroot, or without an overlay, each instance gets a private writable
home at `/harbor-home` instead, because the host's home is not mounted under
`--containall`. A `HOME` passed in a command's environment is exported in the
container shell, since both runtimes refuse to set `HOME` through
`APPTAINERENV_`/`SINGULARITYENV_`.

As in Harbor 0.25, trial log directories are bind-mounted to the host only
with `--environment-kwarg auto_mounting=true`; otherwise they are writable
directories in the trial's staging area that Harbor downloads.

A command that times out or is cancelled has its process tree killed, then
waits at most 10 seconds for the rest of its output, so a background job it
left holding stdout or stderr cannot hang the trial. After a failed start, or
once the environment has stopped, Harbor's log downloads are skipped instead of
raising a second error that would hide the startup failure.

Validate tasks against URSA's Singularity constraints, or warm the SIF cache,
without running an agent:

```bash
uv run -m ursa.integrations.harbor validate benchmark/tasks
uv run -m ursa.integrations.harbor prebuild benchmark/tasks
```

Prebuild includes separate verifier environments and every Compose service:
Dockerfile builds are built, and `image:` references are fetched as local SIFs.
Runs reuse these files from the same cache without contacting the registry.
Use `prebuild --force` to rebuild Dockerfiles and refresh fetched images; it
refreshes the SIFs a trial without `--force-build` uses, so a prebuilt
`docker_image` is re-pulled rather than built.
`prebuild --builder` sets the Dockerfile builder, like `singularity_builder`;
use the same builder for the run so that it finds the prebuilt SIFs.

### Agent install layers

Prebuild can cache the complete installation, including system packages and
installed binaries, in a read-only overlay:

```bash
export URSA_HARBOR_AGENT_CACHE="/shared/$USER/ursa-agent-layers"
uv run -m ursa.integrations.harbor prebuild benchmark/tasks \
  --agent goose --agent claude-code --agent codex
```

Each installer runs in a fresh environment based on the task's main SIF, without
trial mounts, model credentials, inference, or verification. Only a successfully
installed, stopped overlay is published. Concurrent prebuilds share a lock.
Build overlays have at least 4 GiB of sparse capacity for temporary installer
files; this does not change the trial's writable storage setting.
Layers are keyed by the base image contents, agent/version, Harbor/runtime
version, fakeroot setting, host user/home, and working directory. Use the same
per-user cache on the prebuild and compute nodes.

Select a layer when starting the environment and use URSA's installed-agent
adapter so Harbor skips installation:

```bash
harbor run --path benchmark/tasks/example \
  --agent ursa.integrations.harbor.installed_agents:Goose \
  --model provider/model \
  --env ursa.integrations.harbor.singularity:DockerfileSingularityEnvironment \
  --environment-kwarg singularity_agent=goose
```

| Prebuild / `singularity_agent` name | Harbor agent import path |
| --- | --- |
| `ursa` | `ursa.integrations.harbor:UrsaHarborAgent` |
| `goose` | `ursa.integrations.harbor.installed_agents:Goose` |
| `claude-code` | `ursa.integrations.harbor.installed_agents:ClaudeCode` |
| `codex` | `ursa.integrations.harbor.installed_agents:Codex` |

Each trial mounts a private sparse copy of the cached installation read-only
beneath its own writable overlay; its writable Harbor directories are seeded
from that combined image. Filesystems that support reflinks share the copy's
unchanged blocks. The private copy also isolates filesystem mount metadata.
Trial changes never update the cached layer. An absent or mismatched layer fails
with instructions to prebuild it. The adapters retain normal Harbor installation
when no layer is selected. The original Harbor `goose` agent unconditionally
reinstalls, so use the adapter to benefit from the cached installation.
Separate verifier environments retain their own images without the agent layer.

For URSA, prebuild with `--agent ursa`, then add
`--environment-kwarg singularity_agent=ursa` to your usual URSA Harbor run.
The existing `UrsaHarborAgent` handles reuse directly. Its default install source
is the current checkout or installed package. Each trial still prepares its own
native configuration, model settings, and secret references; these are not cached.

To customize the URSA installation, pass `--agent-kwarg` to prebuild (with one
`--agent`). Supported options are `ursa_install_spec` (a PEP 508 requirement),
`ursa_extras`, and `extra_packages`. Match those options in both the runtime
agent kwargs and the environment's `singularity_agent_kwargs` JSON object:

```bash
uv run -m ursa.integrations.harbor prebuild benchmark/tasks --agent ursa \
  --agent-kwarg 'ursa_install_spec=ursa-ai==0.17.3'

harbor run --path benchmark/tasks/example \
  --agent ursa.integrations.harbor:UrsaHarborAgent \
  --model provider/model \
  --agent-kwarg 'ursa_install_spec=ursa-ai==0.17.3' \
  --env ursa.integrations.harbor.singularity:DockerfileSingularityEnvironment \
  --environment-kwarg singularity_agent=ursa \
  --environment-kwarg 'singularity_agent_kwargs={"ursa_install_spec":"ursa-ai==0.17.3"}'
```

URSA layers also track the package requirement, extras, additional packages,
Python/uv versions, and local source contents. Source changes require a new
prebuild. Use `--force` to refresh an unpinned remote requirement.

Use `--agent NAME@VERSION` during prebuild and
`--environment-kwarg singularity_agent=NAME@VERSION` with
`--agent-kwarg version=VERSION` during the run. Version strings are forwarded to
Harbor's installer unchanged. For URSA, select versions through
`ursa_install_spec` instead. `--agent-install-timeout-sec` adjusts the
1800-second limit per installer. Prebuild honors the task's setup network policy;
run it where installation is allowed, then reuse the layer for offline setup.
Match any runtime fakeroot override with prebuild's `--fakeroot` or
`--no-fakeroot`.

The default cache is `$XDG_CACHE_HOME/ursa/harbor/agents` or
`~/.cache/ursa/harbor/agents`. Override it with `URSA_HARBOR_AGENT_CACHE`,
prebuild's `--agent-cache-dir`, or the environment's
`singularity_agent_cache_dir`. `prebuild --force` rebuilds both images and the
selected install layers. Omit `--agent` to build only images. Cached layers
persist independently of artifact retention; clear them only when no trials or
prebuilds are running.

### Compose and networking

`environment/docker-compose.yaml` starts sidecars through
[`singularity-compose`](https://singularityhub.github.io/singularity-compose/).
The `main` service may use an image or a Dockerfile build; verifier overlays
are resolved relative to their overlay file. The adapter supports images,
Dockerfile builds, list `depends_on`, commands, bind mounts, TCP ports,
replicas, environment, `env_file`, and per-service artifacts.
`deploy.resources` is accepted but not enforced, like the task's `cpus` and
`memory_mb`. Unsupported Compose features fail before launch; consult Harbor's
[multi-container guide](https://docs.harborframework.com/core-concepts/tasks/multi-container)
and the [`singularity-compose` specification](https://github.com/singularityhub/singularity-compose/blob/master/docs/spec/spec-2.0.md).
Apptainer and SingularityCE work without a system-wide `singularity` alias.

Bind mounts use `SOURCE:TARGET`, `SOURCE:TARGET:ro`, `SOURCE:TARGET:rw`, or the
long form with `type: bind` and optional `read_only: true`; the source must
exist on the host, and paths cannot contain `:` or `,`. `singularity-compose`
0.1.19 accepts only `SOURCE:TARGET` volumes, so read-only mounts, including
read-only Harbor mounts, are passed to the runtime as `--bind=SOURCE:TARGET:ro`
start options, followed by any mounts nested inside them so that the read-only
parent does not hide them.

Compose files are interpolated as Docker Compose does: `$VAR`, `${VAR}`,
`${VAR:-default}`, `${VAR-default}`, `${VAR:?message}`, `${VAR?message}`,
`${VAR:+alternative}`, `${VAR+alternative}`, nested defaults, and `$$` for a
literal `$`. As with Harbor's Docker backend, variables come from the host
environment, overlaid with the task's `[environment].env`, Harbor's persistent
environment, and then Harbor's own Compose variables: `CONTEXT_DIR`,
`MAIN_IMAGE_NAME`, `PREBUILT_IMAGE_NAME`, `CPUS`, `MEMORY`, and the legacy
`ENV_*_PATH`/`HOST_*_PATH` log mount variables. A missing required variable
fails with its message; an unset plain `$VAR` becomes an empty string with a
warning. Unquoted and double-quoted `env_file` values are interpolated too,
and single-quoted values are kept literally.

A project whose only service is `main`, without published ports or replicas,
uses host networking for public phases, as a task without Compose does, and
keeps the host's `/etc/hosts` and `/etc/resolv.conf`. Sidecars and published
ports need bridge networking, which requires root or a setuid installation
whose fakeroot CNI networking works for the invoking user (with `/etc/subuid`
and `/etc/subgid` entries). On HPC hosts with an unprivileged (non-setuid)
Apptainer and no `/etc/subuid`, such a project fails before any image is
built: the trial looks for the runtime's setuid `starter-suid` under the
`LIBEXECDIR` reported by `buildcfg` (and beside a relocated binary). Root,
`singularity_fakeroot=false`, and unrecognized layouts are never rejected by
this check. Because `singularity-compose` 0.1.19 exits 0 when an instance
fails to start, the adapter then lists the running instances and fails with
the runtime's `FATAL` output when one is missing.

The adapter supports Harbor `public`, `no-network`, and HTTP/HTTPS `allowlist`
policies. The `[environment]` policy is the setup baseline; omitted agent and
verifier phase policies inherit it:

```toml
[environment]
network_mode = "public"

[agent]
network_mode = "allowlist"
allowed_hosts = ["api.openai.com", "*.python.org"]

[verifier]
network_mode = "no-network"
```

Before each phase it restarts the same SIF with the requested policy, preserving
its writable overlay and Harbor binds. Keep the overlay enabled:
`singularity_overlay=false` disables phase changes. Restricted policies are
unavailable for Compose projects with sidecars.

### Allowlisted networking and model access

An `allowlist` policy automatically starts a host-side proxy and a loopback
bridge inside the container. This also applies when Harbor adds a known LLM
endpoint to a restricted agent policy, including `no-network`. No separate proxy
flag or process is needed. Model credentials stay as configured.
When resolving URSA profiles with a configured embedding endpoint, `resolve_run`
allows both chat and embedding hosts, including endpoints inherited from named
inference providers. Its CLI arguments preserve these allowances.

```text
agent -> container loopback HTTP proxy -> bind-mounted Unix socket
      -> host-side allowlist -> upstream HTTP/HTTPS service
```

The container runs with `--net --network none`. The integration sets uppercase
and lowercase `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY`. `NO_PROXY` keeps
localhost requests inside the container, so agents can test local services.
Clients must honor these settings. Direct external network access remains
unavailable. Phase changes stop the bridge and
close existing proxy connections before applying the next policy; model access
added for the agent does not carry into an offline verifier phase.

For model APIs on the host's loopback interface, the URSA adapter and the Goose,
Claude Code, and Codex adapters listed above automatically create dedicated
forwards and substitute their ports in the runtime model URLs. Use those
adapters even without an install cache when connecting to host-local models.
The forwards preserve API paths, TLS certificate verification, and streaming;
unrelated container-local services continue to work. Each phase gets fresh
forwards under its current allowlist.

Rules follow Harbor's host-based policy: exact DNS names, `*.example.com`
(subdomains, excluding the apex), and IPv4/IPv6 addresses are supported, on all
ports. CIDR ranges are rejected. Unlisted destinations receive HTTP 403 before
DNS lookup; redirected requests are checked separately. The proxy preserves
streaming responses and upstream errors. HTTPS uses CONNECT with client-side
certificate verification; rules control TCP destinations, not HTTP paths, TLS
SNI, or other services sharing an allowed destination.

For an unprivileged Apptainer runner, pass `--ek singularity_fakeroot=false`.
Restricted phases then use `--userns`, without sudo, setuid/setgid helpers,
subordinate UID/GID ranges, or CNI configuration. The host must permit
unprivileged user/network namespaces. The task image needs Python 3.8+ for the
stdlib bridge; a cached URSA installation's interpreter is also supported.
The proxy socket lives in a private directory and is removed on cleanup.

The trial's working directory follows the Dockerfile's `WORKDIR` lines, with
variables from its `ARG` and `ENV` lines and Compose build `args`, for the
Compose build `target` when one is set. A relative first `WORKDIR`, or one that
uses a variable only the base image sets, is resolved against the base image's
config as for definition builds; when that config cannot be read, a relative
`WORKDIR` starts in `/` with a warning. Set `[environment].workdir` when a
Dockerfile inherits a non-root workdir from its base image without a `WORKDIR`
of its own; the adapter rejects workdirs it cannot resolve before launch. A
pulled Compose `main.image`, or a prebuilt `docker_image` without a Dockerfile,
starts in the `WorkingDir` of its registry config, as under Docker, or in `/`
with a warning when that config cannot be read; `[environment].workdir`
overrides either. A prebuilt image that ships with its Dockerfile uses that
Dockerfile's `WORKDIR`. Workdirs are resolved when the trial starts, for the
image it uses: with `--force-build`, the Dockerfile's.

To verify the integrated lifecycle with an existing Python-enabled SIF, run:

```bash
setpriv --no-new-privs uv run python tests/integrations/harbor_socket_proxy_smoke.py \
  --image /path/to/agent.sif
```

The smoke test checks privilege isolation, JSON and streaming model responses,
HTTP and certificate-verified HTTPS, local services, denied hosts, phase changes,
and cleanup with stalled clients.
It uses local test servers and requires no model credentials.

## Configure other agents

For other Harbor agents, `ursa.integrations.harbor.config` exports `AgentConfig`, `ModelConfig`,
`load_agent_config`, `load_provider_config`, and `resolve_run`. It resolves one invocation;
callers control scheduling and output locations.

```yaml
# agent.yaml
harbor_agent: codex
kwargs: {}
setup_timeout_multiplier: 1.2
```

```yaml
# provider.yaml
base_url: https://inference.example/v1
api_key: {env: LAB_API_KEY}
model_provider: openai
```

```python
import os
import subprocess
from pathlib import Path

from ursa.integrations.harbor.config import (
    ModelConfig, load_agent_config, load_provider_config, resolve_run,
)

resolved = resolve_run(
    Path("benchmark/tasks/install-pytorch"),
    load_agent_config(Path("agent.yaml")),
    ModelConfig(provider="lab", name="model-name"),
    load_provider_config(Path("provider.yaml"), "lab"),
    reasoning="high",
)
# These are inspectable without resolving the credential.
print(resolved.agent.model_dump())
subprocess.run(
    ["harbor", "run", *resolved.cli_args(), "--jobs-dir", "jobs"],
    env={**os.environ, **resolved.environment()},
    check=True,
)
```

Resolution does not launch a process or read the selected credential. The
environment is materialized separately at launch. The provider uses URSA's
existing `api_key` secret/reference schema, including keyring references.
`api_key_env` in the agent config chooses the host variable handed to Harbor's
`--api-key-name` (default `URSA_HARBOR_API_KEY`). Harbor owns projection to the
harness. Provider credential values do not enter CLI arguments or generated
native configuration. Use secret references for any additional native secrets
and avoid literal credentials in `env` or arbitrary agent kwargs.

Endpoints use `--base-url`; there is no `base_url_env` setting. A `base_url=`
argument to `resolve_run` overrides the provider URL, for example for a local
timing relay. `api_format` selects Harbor's `--api-format`. OpenCode can use
`openai_chat_completions` with a custom endpoint without an URL environment
variable or custom provider overlay.

Optional agent fields:

- `provider_kwargs` maps provider aliases to shallow overrides of `kwargs`.
- `native_config` accepts an inline mapping or a YAML/JSON/TOML file, relative
  to the agent file. Harbor receives a mapping and owns native serialization.
  A harness must declare native-config support.
- `reasoning_kwarg` selects a harness-specific option such as OpenCode's
  `variant`. Otherwise Harbor's `reasoning_effort` capability is required.
  URSA receives reasoning through `llm_model.reasoning_effort`.
- `model_template` supports `{provider}`, `{model_provider}`, and `{model}`.
  The default is `{model_provider}/{model}` for native harnesses and
  `{provider}/{model}` for URSA. Use `{model}` for a bare model selector.
- `env` contains Harbor environment settings/references.

The resolver checks Harbor options and capabilities, resolves only the selected
provider into URSA's native layer, and preserves caller inputs. Model names are
not restricted to filesystem slugs. Runtime availability and API connectivity
are still checked by Harbor at execution time.

## Artifact retention

```yaml
# retention.yaml
retention:
  version: 1
  collections: [trial]
  default_action: archive
  archive:
    format: tar.xz
    path: retained-artifacts.tar.xz
    remove_originals: true
    compression_level: 6
```

```bash
uv run harbor run --path benchmark/tasks/install-pytorch --agent oracle \
  --plugin ursa.integrations.harbor.retention:RetentionPlugin \
  --plugin-kwarg policy=retention.yaml
```

The plugin runs after job finalization and all trial/verifier consumers finish,
including failed trials. Cancelled, unfinalized jobs are untouched. It selects
all files in each trial directory (the parent of `artifacts`) when
`collections: [trial]` is selected. This includes agent logs and checkpoints,
configuration, lock files, results, artifact manifests, verifier logs and rewards,
and all multi-step outputs. Job-level files and retention metadata stay outside
the archive. Select `trial` on its own.

The default scope remains `collections: [artifacts]`; `[artifacts, verifier]`
also includes verifier payloads. These narrower scopes protect collection
manifests, rewards, and verifier test stdout/stderr, and leave other trial files
untouched.

Optional rules use gitignore-style globs relative to each selected root: the
trial directory for `[trial]`, or each payload directory for the narrower scopes.
For example, `agent/**` matches agent logs in trial scope. The first
match wins; unmatched files default to `keep`. `**/*.dat` includes root and nested
files; negative/comment patterns are rejected. Archives preserve full
job-relative trial/step paths to avoid collisions.

Archives are written temporarily, verified against the inventory and content
hashes, then published without overwriting existing files. Only then are selected
originals deleted. Allow space for both sources and the archive; compression
levels range from 0–9. A lock serializes retention writers, but other payload
writers must have stopped. POSIX directory fsync protects publication and journals.
With whole-trial retention and `remove_originals: true`, restore the trial before
using Harbor commands that need its result, configuration, or logs.

`retention.json` records the plan, file identities, sizes, hashes, deletions,
link warnings, duration, and status. Reapplying a completed policy verifies it;
interrupted compression restarts and interrupted cleanup resumes. Changed inputs,
including new trials, are rejected; whole-trial cleanup can resume even after
trial results have been removed. Different policies
and unrelated archives are rejected. Each job supports one policy/archive.
Harbor logs plugin errors without failing the job or changing
scores: require `status: complete` in `retention-status.json` to confirm retention.

### Preview, resume, and restore

```bash
uv run -m ursa.integrations.harbor artifacts plan \
  --run-dir jobs/example --policy retention.yaml
uv run -m ursa.integrations.harbor artifacts apply \
  --run-dir jobs/example --policy retention.yaml
uv run -m ursa.integrations.harbor artifacts restore \
  --manifest jobs/example/retention.json --destination restored-example
```

Preview inventories files and hashes archive-selected content without writing.
Apply requires finished job/trial results, resumes interrupted cleanup, and updates
`retention-status.json`. Historical jobs are only processed when explicitly selected.

Restore verifies the archive and extracts into a fresh directory, rejecting path
traversal and escaping symlinks. Links are archived without following targets;
warnings identify missing targets. Deleted files are unrecoverable, and kept
loose files must be retrieved from the original job.

For other adapters, `ursa.artifacts.retention` exposes
`plan_retention(root, policy, roots=..., protected=...)`, `apply_retention(plan)`,
`retain_artifacts(...)`, and `restore_artifacts(manifest, destination)`.

Remove `jobs/` when results are no longer needed. Remove SIF cache entries
only when no Harbor jobs use them.
