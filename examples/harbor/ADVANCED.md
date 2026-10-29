# Advanced Harbor usage

## Customize the URSA installation

The adapter installs Python and URSA with `uv` inside each task container.
Add URSA extras or other packages with agent kwargs:

```bash
--agent-kwarg ursa_extras=image \
--agent-kwarg 'extra_packages=["numpy","scipy"]'
```

Set `ursa_install_spec` to a package requirement, Git URL, local project
directory, or local wheel/sdist archive. Local inputs are staged into the task
container; project directories exclude Git-ignored files and common secret
files:

```bash
--agent-kwarg ursa_install_spec=/path/to/checkout
```

## Choose the URSA config stack

By default the adapter merges only the file passed as `config_file`, then
applies Harbor's model and MCP settings last. Set
`--agent-kwarg config_only=false` to include URSA's system and user config
layers first. The supplied file is still merged below Harbor's settings.

Secret references are resolved on the host, including keyring references. The
adapter passes generated environment references only to the URSA runner; it
does not copy host config or keyring files into the task container.

Harbor selects the model credential from `--ae`, then the host environment, or
from the variable named by `--api-key-name`. The adapter moves that key and any
`<PROVIDER>_BASE_URL` endpoint into the URSA inference provider, again as a
generated environment reference. Harbor itself also exports every `--ae`
variable unchanged into each agent setup and run command, so a key passed with
`--ae` is visible to the install commands and the runner under its original
name. To pass the key only to the runner, as a generated reference, supply it
from the host environment or `--api-key-name`, or as an URSA `{env: ...}`
secret reference. Provider extras such as `AZURE_API_VERSION` and the AWS
variables for Bedrock and SageMaker are passed through unchanged. The text
before the first `/` in `--model` always names the URSA inference provider,
including custom providers that LiteLLM does not know. Harbor's `--base-url`
flag is not supported; set `base_url` in `ursa.yaml` or export
`<PROVIDER>_BASE_URL` instead.

Harbor MCP servers are forwarded to URSA. Servers that declare HTTP `headers`
are rejected, because the adapter does not forward MCP request headers.
URSA has no skills support, so Harbor rejects tasks that set
`environment.skills_dir` and runs that pass `--skills`.

Each trial renders human-readable agent and tool progress to `agent/ursa.log`
and writes structured chat, tool, agent, and chain lifecycle records to
`agent/ursa.jsonl`. Each JSONL record has a monotonic nanosecond timestamp;
chat end records include token usage. The formatted final result is written to
`agent/ursa_result.out`.

Harbor copies these files to the host after the agent finishes, applying any
`include_logs` and `exclude_logs` filters. To report token usage, the adapter
reads `ursa.jsonl` from a temporary copy that it deletes afterwards. Pass
`--ek auto_mounting=true` to bind the trial's log directories to the host while
the agent runs, for example to follow `ursa.log` live.

## Singularity

When a task sets `[environment].docker_image`, as every Terminal-Bench 2 task
does, the custom environment pulls that image with `<runtime> pull
docker://<image>` and needs no container builder. A reference without a tag
gets `:latest`, a reference with another scheme (such as `library://`) is
pulled as written, and a path ending in `.sif` is used directly. Otherwise it
builds `environment/Dockerfile` with Buildah, Podman, or Docker, converts it
to a cached SIF, and requires no Apptainer definition file. Building needs one
of `buildah`, `podman`, or `docker` on the host; when using Docker, its daemon
must be running and the invoking process must have socket access. Rootless
Buildah and Podman can only run Dockerfile `RUN` steps when the user has
`/etc/subuid` and `/etc/subgid` entries.

The pull-or-build choice follows Harbor: `harbor run --force-build` builds the
Dockerfile even when `docker_image` is set, and pulls again when the task has
no Dockerfile. The `singularity_force_pull` environment kwarg refreshes cached
images and rebuilds Dockerfiles, but never switches a `docker_image` task to a
build. The same rules apply to a separate verifier image and to Compose
services. A Compose `main.image` replaces a pulled `docker_image`; when the
task is built instead, it only names the built image, as in Harbor, and is
pulled only if the task has no Dockerfile. A pulled task image does not report
its `WORKDIR`; the adapter reads it from `environment/Dockerfile` when one
exists, and otherwise uses `/`, so set `[environment].workdir` for image-only
tasks that need another directory.

SIFs are cached under `$XDG_CACHE_HOME/ursa/harbor/sif`, or
`~/.cache/ursa/harbor/sif` when `XDG_CACHE_HOME` is unset. Built images are
keyed by build context and pulled images by their reference, so a tag is
pulled only once until it is refreshed and a digest-pinned reference
(`name@sha256:...`) is keyed by its digest. Concurrent trials that need the
same image share a lock and build or pull it only once. Set
`URSA_HARBOR_SIF_CACHE` or pass `singularity_image_cache_dir` as an
environment kwarg to use a different shared cache. The runtime also keeps
its own layer cache, `~/.apptainer/cache` or `~/.singularity/cache` by
default; set `APPTAINER_CACHEDIR` or `SINGULARITY_CACHEDIR` when home
quotas are small. Registry credentials come from `APPTAINER_DOCKER_USERNAME`
and `APPTAINER_DOCKER_PASSWORD` (or the `SINGULARITY_` forms), which also
avoid Docker Hub's anonymous rate limit for large task sets.

When compute nodes cannot reach a registry, prepare the cache on a login node:

```bash
uv run -m ursa.integrations.harbor prebuild path/to/tasks
```

It prepares the SIFs a trial without `--force-build` uses. Its `--force`
option rebuilds or re-pulls them, like `singularity_force_pull`, and never
switches a `docker_image` task to a Dockerfile build.

For a direct run, add:

```bash
--env ursa.integrations.harbor_singularity:DockerfileSingularityEnvironment
```

### Fakeroot

Most tasks need root inside the container at run time, for example to
`apt-get install` packages in a verifier script. The adapter therefore runs
containers with `--fakeroot`, which makes the invoking user root inside the
container and lets the adapter switch to the container user that Harbor
requests. The `singularity_fakeroot` environment kwarg controls this:

- `true` (default): before the first container command, the adapter checks
  that `<runtime> exec --fakeroot <image> true` succeeds. If it fails, for
  example because the user has no `/etc/subuid` entry, the trial fails with
  the runtime's error. The result is reused by later environments in the same
  process, such as a separate verifier, which then fail before building or
  pulling an image. A check that fails for another reason, such as an image
  pull error, reports that error instead.
- `false` (or `0`): never use fakeroot and skip the check. Commands run as the
  invoking host user even when the task requests `root` or another container
  user, so root-owned files in the image stay read-only and tasks that install
  system packages fail. The adapter logs one warning per process.

Without fakeroot, Singularity keeps the host's `HOME` path inside the
container but does not mount it, so the adapter gives each container a
private, writable home at `/harbor-home`, backed by the trial's staging
directory. Tools such as `pip install --user`, `uv`, and
`source $HOME/.local/bin/env` then work. A `HOME` passed in a command's
environment takes precedence. With fakeroot, `HOME` is `/root`, which is
writable through the overlay; with `singularity_overlay=false` the image root
filesystem, including `/root`, is read-only, so fakeroot containers get the
private `/harbor-home` as well.

To run tasks that do not need container root on a host without fakeroot:

```bash
--ek singularity_fakeroot=false
```

### Apptainer

The adapter works with SingularityCE 3.6 and Apptainer 1.1 or later, and
prefers `apptainer` when both are on `PATH`. It detects Apptainer behind a
`singularity` alias from `--version`, so task variables use the
`APPTAINERENV_` prefix and avoid Apptainer's deprecation warning on every
command. Image pulls, `--fakeroot`, and instance commands use the same syntax
on both runtimes.

An unprivileged (user-space) Apptainer installation has extra requirements:

- `squashfuse` mounts SIF images; without it, Apptainer extracts the image to
  a temporary sandbox on every start, which is slow and uses scratch space.
- The writable overlay needs `fuse2fs` and `fuse-overlayfs`. Without them,
  set `singularity_overlay=false`; the image root filesystem is then
  read-only and network-policy phases cannot change. For Apptainer the
  overlay is a bare ext3 image, and Apptainer creates `upper/` and `work/` in
  it: under root-mapped fakeroot it rejects pre-made ones, such as those from
  `apptainer overlay create`, as not writable. SingularityCE 3.6 still gets
  them pre-created, owned by the invoking user, as it requires.
- Without an `/etc/subuid` entry, Apptainer 1.1 and later implement
  `--fakeroot` with a root-mapped user namespace, helped by the host's
  `fakeroot` command when it is installed. Only root is mapped, so tasks that
  ask for a non-root container user fail to switch to it, and package
  managers can fail to drop privileges; check an `apt-get install` early.
- Bridge networking is unavailable, so Compose projects with sidecars or
  published ports cannot start. Main-only Compose projects, such as SciCode,
  and `no-network` phases still work.

### Multi-container tasks

Add Harbor's standard `environment/docker-compose.yaml` to start sidecars with
[`singularity-compose`](https://singularityhub.github.io/singularity-compose/).
The reserved `main` service defaults to `environment/Dockerfile`, but it may
use an OCI `image` or a `build` with a different context, Dockerfile, build
arguments, or target. This also works in verifier overlays such as
`tests/docker-compose.yaml`, where paths are relative to the overlay file.
Sidecars support the same `image` and `build` forms:

```yaml
services:
  main:
    build:
      context: ..
      dockerfile: tests/Dockerfile
    depends_on:
      - api

  api:
    build:
      context: ./api
    command: [python, server.py]
    expose:
      - "8000"
```

The adapter translates the subset that `singularity-compose` can represent:
images, Dockerfile builds (`context`, `dockerfile`, `args`, and `target`), list
form `depends_on`, commands, bind mounts, TCP `host:container` ports,
`deploy.replicas`, `environment`, and `env_file`. `deploy.resources` is
accepted but not enforced, like the task's `cpus` and `memory_mb`. Sidecars are
reachable by their Harbor service name. Harbor's per-service artifact
collection is also supported.

Bind mounts use `SOURCE:TARGET`, `SOURCE:TARGET:ro`, `SOURCE:TARGET:rw`, or the
long form with `type: bind` and optional `read_only: true`. The source must
exist on the host, and paths cannot contain `:` or `,`. Because
`singularity-compose` 0.1.19 accepts only `SOURCE:TARGET` volumes, the adapter
passes read-only mounts, including read-only Harbor mounts, to Singularity as
`--bind=SOURCE:TARGET:ro` start options instead, followed by any mounts nested
inside them so the read-only parent does not hide them. A mount inside the task's
workdir or another writable Harbor path gets an empty mount point in that
path's host directory, so the image does not need to provide one.

Values in Compose files are interpolated as Docker Compose does: `$VAR`,
`${VAR}`, `${VAR:-default}` and `${VAR-default}`, `${VAR:?message}` and
`${VAR?message}`, `${VAR:+alternative}` and `${VAR+alternative}`, with nested
defaults and `$$` for a literal `$`. As for Harbor's Docker backend,
variables come from the host environment, overlaid with the task's
`[environment].env` and Harbor's persistent environment, and then with
Harbor's own Compose variables, which take precedence: `CONTEXT_DIR` (the
task's `environment` directory), `MAIN_IMAGE_NAME`, `PREBUILT_IMAGE_NAME`
(the task's `docker_image`), `CPUS` and `MEMORY` (from the task's resources),
and the legacy `ENV_*_PATH` and `HOST_*_PATH` log mount variables. A missing
required variable fails with its message, and, as in Docker Compose, a plain
`$VAR` or `${VAR}` that is unset becomes an empty string with a warning.
Unquoted and double-quoted `env_file` values are interpolated with the same
variables, and single-quoted values are kept literally. A bare `KEY` in
`environment`, `env_file`, or build `args` also takes its value from them.

For example, Harbor's SciCode adapter mounts its test data with:

```yaml
services:
  main:
    volumes:
      - "${HARBOR_SCICODE_TEST_DATA_PATH:?set HARBOR_SCICODE_TEST_DATA_PATH to an absolute path to a local test_data.h5 before running harbor}:/app/test_data.h5:ro"
```

Export `HARBOR_SCICODE_TEST_DATA_PATH` in the shell that runs Harbor. SciCode
tasks have no `docker_image`, so their Dockerfile is still built and needs a
builder that can run `RUN` steps.

Unsupported Compose fields fail before launch. In particular,
`singularity-compose` cannot represent healthchecks or conditional
`depends_on`, named volumes, custom Compose networks, entrypoints, resource
limits, or privilege/capability settings. See Harbor's
[multi-container task guide](https://docs.harborframework.com/core-concepts/tasks/multi-container)
and the
[`singularity-compose` 2.0 specification](https://github.com/singularityhub/singularity-compose/blob/master/docs/spec/spec-2.0.md)
before porting a Docker Compose task.

Multi-container tasks work with either SingularityCE or Apptainer. Although
`singularity-compose` 0.1.19 invokes a command named `singularity` internally,
the adapter maps that command to the selected runtime inside its private
project directory. An Apptainer installation therefore does not need a
system-wide `singularity` alias. The `ursa-ai[harbor]` extra installs
`singularity-compose`. A project whose only service is `main`, without
published ports, uses host networking as a task without Compose does, including
the host's `/etc/hosts` and `/etc/resolv.conf` instead of the files
`singularity-compose` generates for its bridge. Projects with sidecars or ports need bridge networking, which requires the host runtime
to provide fakeroot CNI networking.

The adapter supports Harbor's `public` and `no-network` modes on
SingularityCE 3.6.2. The `[environment]` policy is the baseline used when the
container starts and while Harbor sets up the task and installs the agent:

```toml
[environment]
network_mode = "public"

[agent]
network_mode = "no-network"

[verifier]
network_mode = "no-network"
```

Immediately before `agent.run()`, Harbor switches to the `[agent]` policy. It
restores the `[environment]` baseline afterward, then applies `[verifier]`
while the verifier runs. An omitted phase policy inherits the baseline. The
adapter restarts the same SIF with the requested network namespace and reuses
its writable overlay and Harbor bind mounts, so files installed or changed
during setup remain available in later phases.

The effective Harbor policy is also projected into URSA's `use_web` setting
for each agent phase: `public` enables web tools, while `no-network` and
`allowlist` disable them, as for Harbor's built-in agents. Harbor still
enforces the actual network boundary. Keep the writable overlay enabled (the
default); setting the `singularity_overlay` environment kwarg to `false`
disables runtime policy changes, and Harbor then rejects tasks whose phases
use different policies.

`no-network` starts the task as a named Singularity instance with an isolated
`none` network. Phase switching is supported for single-container tasks and
Compose projects containing only the `main` service. A `no-network` baseline
or phase is not available to projects with sidecars because
`singularity-compose` cannot preserve communication between sidecars while
isolating the project from external networks. Network allowlists remain
unsupported because SingularityCE 3.6.2 cannot enforce them securely. See
Harbor's
[network-policy reference](https://docs.harborframework.com/tasks/network-policies#phases)
and Singularity's
[networking guide](https://docs.sylabs.io/guides/3.6/user-guide/networking.html).

Set `[environment].workdir` in `task.toml` when a Dockerfile computes
`WORKDIR` from an environment variable or inherits a non-root workdir from its
base image. The task setting takes precedence over image metadata; the
Singularity adapter rejects variable workdirs it cannot resolve before launch.

## Clean up

Remove `jobs/` when local results are no longer needed. Remove the SIF cache
only when no Harbor jobs use it.
