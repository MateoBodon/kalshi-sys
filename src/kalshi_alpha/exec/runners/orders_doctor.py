"""Command-line helper for cleaning up outstanding DRY orders."""

from __future__ import annotations

import argparse
from pathlib import Path

from kalshi_alpha.exec.state.orders import OutstandingOrdersState


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and repair outstanding order state.")
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Mark outstanding DRY orders as cancelled and request broker cancel.",
    )
    parser.add_argument(
        "--clear-dry",
        action="store_true",
        help="Remove all outstanding DRY orders from the ledger.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display outstanding order details after any mutations.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        help="Override the default OutstandingOrdersState path (mainly for testing).",
    )
    return parser.parse_args(argv)


def _resolved_state(path: Path | None) -> OutstandingOrdersState:
    if path is None:
        return OutstandingOrdersState.load()
    return OutstandingOrdersState.load(path)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    state = _resolved_state(args.state_path)

    dry_keys = list(state.outstanding_for("dry").keys())
    reconciled: list[str] = []
    cleared: list[str] = []

    if args.reconcile and dry_keys:
        reconciled = state.mark_status("dry", dry_keys, status="cancelled")
        if reconciled:
            state.mark_cancel_all("orders_doctor_reconcile", modes=["dry"])

    if args.clear_dry and dry_keys:
        cleared = state.remove("dry", dry_keys)
        if cleared:
            state.clear_cancel_all()

    summary = state.summary()
    total = sum(summary.values())
    lines: list[str] = [f"Outstanding orders: {total}"]
    lines.append(f"- dry: {summary.get('dry', 0)}")
    lines.append(f"- live: {summary.get('live', 0)}")

    if args.reconcile:
        lines.append(f"Reconciled dry orders: {len(reconciled)}")
    if args.clear_dry:
        lines.append(f"Cleared dry orders: {len(cleared)}")

    if args.show:
        for mode in ("dry", "live"):
            bucket = state.outstanding_for(mode)
            if not bucket:
                continue
            lines.append("")
            lines.append(f"{mode.upper()} orders:")
            for key, order in sorted(bucket.items()):
                price = order.get("price")
                contracts = order.get("contracts")
                status = order.get("status", "pending")
                side = order.get("side")
                market_id = order.get("market_id")
                lines.append(
                    f"  {key}: status={status} {side} {contracts}@{price} market={market_id}"
                )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
