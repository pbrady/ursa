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
generated environment reference. Provider extras such as `AZURE_API_VERSION`
and the AWS variables for Bedrock and SageMaker are passed through unchanged.
The text before the first `/` in `--model` always names the URSA inference
provider, including custom providers that LiteLLM does not know. Harbor's
`--base-url` flag is not supported; set `base_url` in `ursa.yaml` or export
`<PROVIDER>_BASE_URL` instead.

Harbor MCP servers are forwarded to URSA. Servers that declare HTTP `headers`
are rejected, because the adapter does not forward MCP request headers.

Each trial renders human-readable agent and tool progress to `agent/ursa.log`
and writes structured chat, tool, agent, and chain lifecycle records to
`agent/ursa.jsonl`. Each JSONL record has a monotonic nanosecond timestamp;
chat end records include token usage. The formatted final result is written to
`agent/ursa_result.out`.

Harbor copies these files to the host after the agent finishes. The adapter
downloads `ursa.jsonl` itself to report token usage. Pass
`--ek auto_mounting=true` to bind the trial's log directories to the host while
the agent runs, for example to follow `ursa.log` live.

## Singularity

The custom environment builds `environment/Dockerfile` with Buildah, Podman,
or Docker, converts it to a cached SIF, and requires no Apptainer definition
file. Compute nodes need `apptainer` or `singularity`, plus one of `buildah`,
`podman`, or `docker`. When using Docker, its daemon must be running and the
invoking process must have socket access.

SIFs are cached by build context under `$XDG_CACHE_HOME/ursa/harbor/sif`, or
`~/.cache/ursa/harbor/sif` when `XDG_CACHE_HOME` is unset. Concurrent trials
with the same build context share a lock and build only once. Set
`URSA_HARBOR_SIF_CACHE` or pass `singularity_image_cache_dir` as an environment
kwarg to use a different shared cache.

For a direct run, add:

```bash
--env ursa.integrations.harbor_singularity:DockerfileSingularityEnvironment
```

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
`deploy.replicas`, `environment`, and `env_file`. Sidecars are reachable by
their Harbor service name. Harbor's per-service artifact collection is also
supported.

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
`singularity-compose`; the host runtime must provide fakeroot CNI networking.

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
