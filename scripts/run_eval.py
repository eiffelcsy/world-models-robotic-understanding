"""Run an evaluation.

    python scripts/run_eval.py --config configs/eval.yaml

Everything else is in the YAML; --list-components prints what the discovered
modules register.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmeval import registry, runner  # noqa: E402
from wmeval.config import load_eval_config  # noqa: E402
from wmeval.modules import import_module  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/eval.yaml", help="path to the run config")
    parser.add_argument("--list-components", action="store_true", help="print registered components and exit")
    parser.add_argument("--arm", default=None, help="prompt arm to score (default: the config's, else base)")
    parser.add_argument("--repeat", type=int, default=None, help="repeat index to score")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    config = load_eval_config(args.config, arm=args.arm, repeat=args.repeat)

    if args.list_components:
        for probe in config.probes:
            import_module(config.modules_dir, probe.module)
        for family in ("readout", "gt", "extractor", "metric", "gate"):
            print(f"{family}:")
            for name in registry.available(family) or ["<none>"]:
                print(f"  {name}")
        return 0

    print(f"probes: {', '.join(p.name for p in config.probes if p.enabled) or '<none>'}")
    print(f"arm: {config.arm}, repeat: {config.repeat}, manifest: {config.manifest}")
    runner.run(config)
    print(f"\nsummary: {config.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
