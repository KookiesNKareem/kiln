"""CLI: python -m kiln_evo {run,resume,status,report,models} ..."""

from __future__ import annotations

import argparse
import json
import sys

from . import config as C


def _cfg(args) -> dict:
    over = {}
    if args.out_dir:
        over["out_dir"] = args.out_dir
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        cur = over
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = json.loads(v) if v[:1] in "[{0123456789-tfn\"" else v
    return C.load(args.campaign, over)


def _out(args) -> str:
    if args.target.endswith((".yaml", ".yml", ".json")):
        return C.load(args.target)["out_dir"]
    return args.target


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kiln_evo", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "resume"):
        s = sub.add_parser(name, help=f"{name} a campaign")
        s.add_argument("campaign")
        s.add_argument("--out-dir")
        s.add_argument("--set", action="append", help="override a config key, e.g. budget.max_evals=200")
    for name in ("status", "report"):
        s = sub.add_parser(name, help=f"{name} of a campaign (campaign file or out dir)")
        s.add_argument("target")
        if name == "report":
            s.add_argument("--top", type=int, default=10)
            s.add_argument("--json", action="store_true")
    s = sub.add_parser("models", help="list the live model catalog of the campaign's LLM provider")
    s.add_argument("campaign")
    args = p.parse_args(argv)

    if args.cmd in ("run", "resume"):
        from .campaign import Campaign, CampaignError

        try:
            cfg = _cfg(args)
            camp = Campaign(cfg, resume=args.cmd == "resume")
        except (C.ConfigError, CampaignError) as e:
            print(e, file=sys.stderr)
            return 2
        camp.run()
        from .report import report

        print(report(cfg["out_dir"], tier_b=camp.evaluator.tier_b_status())[0])
        return 0
    if args.cmd == "status":
        from .report import status

        print(status(_out(args)))
        return 0
    if args.cmd == "report":
        from .report import report

        text, data = report(_out(args), top=args.top)
        print(json.dumps(data, indent=1, default=str) if args.json else text)
        return 0
    if args.cmd == "models":
        from .llm import Spend, make_backend

        raw = C._load_text(__import__("pathlib").Path(args.campaign))
        llm = C._merge(C.DEFAULTS["llm"], raw.get("llm") or {})
        for m in make_backend(llm, Spend({}, None)).list_models():
            print(m)
        print("\nPrices are not in any catalog API: copy input/output USD per million tokens for the chosen model "
              "from the provider's pricing page into llm.price_usd_per_mtok.", file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
