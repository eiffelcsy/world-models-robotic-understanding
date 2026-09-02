"""World models.

One directory per model. A directory is picked up as soon as it contains a
`backend.yaml`, and ignored as soon as it does not:

    backends/<model_name>/
        backend.yaml      type + params + defaults      <- usually the whole thing
        requirements.txt  optional, for a model needing its own dependencies
        backend.py        optional, only when YAML cannot express the call
        __init__.py       optional, imports backend.py so registration happens

`configs/gen.yaml` names one with `backend: <model_name>`.

Adding a world model is normally writing a YAML file, because the three shipped
backend types cover how video models are actually called:

    http_job    submit -> poll -> fetch over REST      (hosted APIs)
    command     a templated subprocess                  (local / self-hosted)
    local_dir   adopt clips generated elsewhere         (by hand, or recovery)

See `backends/veo3/backend.yaml` for a worked `http_job` and
`wmgen/backends.py` for what each type accepts.

When a model genuinely cannot be reached that way — an SDK that signs requests,
a streaming protocol — add `backend.py` and register a class in the `backend`
family, exactly as a probe adds `gt.py`:

    from wmgen import VideoBackend, register

    @register("backend", "my_model")
    class MyModel(VideoBackend):
        def submit(self, request) -> str: ...
        def wait(self, job_id, request) -> dict: ...
        def fetch(self, result, request): ...

then `type: my_model` in its `backend.yaml`. Heavy imports stay inside the
directory and are only loaded when that backend is the one selected.

Never put an API key in a `backend.yaml`. Write `${MY_API_KEY}` and export it;
an unset variable is a loud error rather than a 401.
"""
