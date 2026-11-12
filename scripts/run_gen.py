"""Generate the videos an evaluation run will score.

    python scripts/run_gen.py --config configs/gen.yaml --dry-run
    python scripts/run_gen.py --config configs/gen.yaml

Everything else is in the YAML. It writes clips to `clips_dir` and a manifest to
`manifest_out`, which `scripts/run_eval.py` then reads unchanged.

Generation costs money, so --dry-run comes first in that order deliberately: it
renders every prompt and writes the manifest that would result without calling
any backend.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmcore import registry  # noqa: E402
from wmcore.modules import import_module  # noqa: E402
from wmgen import runner  # noqa: E402
from wmgen.config import load_gen_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/gen.yaml", help="path to the run config")
    parser.add_argument("--list-components", action="store_true", help="print registered components and exit")
    parser.add_argument("--dry-run", action="store_true", help="render prompts, call no backend")
    parser.add_argument("--refresh", action="store_true", help="regenerate clips that already exist")
    parser.add_argument("--limit", type=int, default=None, help="only N frames — first N, or see --limit-by")
    parser.add_argument(
        "--limit-by",
        default=None,
        help="with --limit, draw proportionally from this frame.meta key's distribution "
             "(e.g. category) instead of taking the first N",
    )
    parser.add_argument("--arm", default=None, help="prompt arm: modules/<probe>/arms/<arm>.yaml (default: the config's, else base)")
    parser.add_argument("--repeat", type=int, default=None, help="repeat index; the backend seed is seed_base + repeat")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    config = load_gen_config(args.config, arm=args.arm, repeat=args.repeat)

    # Flags override the YAML, and only when given — so a config that already
    # says `dry_run: true` is not silently switched off by its absence here.
    if args.dry_run:
        config.dry_run = True
    if args.refresh:
        config.refresh = True
    if args.limit is not None:
        config.limit = args.limit
    if args.limit_by is not None:
        config.limit_by = args.limit_by

    if args.list_components:
        import wmgen.runner  # noqa: F401  (imports the shipped components)

        for prompt in config.prompts:
            try:
                import_module(config.modules_dir, prompt.module)
            except ImportError:
                pass
        for family in ("source", "semantics", "prompt", "backend", "check"):
            print(f"{family}:")
            for name in registry.available(family) or ["<none>"]:
                print(f"  {name}")
        return 0

    print(f"probes: {', '.join(p.name for p in config.prompts if p.enabled) or '<none>'}")
    print(f"arm: {config.arm}, repeat: {config.repeat}")
    if config.refresh and not config.dry_run:
        print("--refresh: existing clips will be generated again")
    runner.run(config)
    print(f"\nsummary:  {config.output_dir / 'gen_summary.md'}")
    print(f"manifest: {config.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
