"""Explicit one-call API check, or offline verification of its receipt."""
import argparse
import json
from pathlib import Path
from vdc.io import read_json
from vdc.research_api import check_connection, verify_connection_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path, help='NEW directory: performs one paid request')
    group.add_argument('--check', type=Path, help='Existing status.json: no API request')
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parents[1]/'configs/research_round1.json')
    args = parser.parse_args()
    try:
        config = read_json(args.config)
        report = (verify_connection_check(args.check, config) if args.check
                  else check_connection(args.out, config))
        print(json.dumps({'operation': 'verify_receipt' if args.check else 'one_paid_api_check',
                          **report}, indent=2))
        return 0
    except Exception as exc:
        # Do not print provider bodies, environment variables or arbitrary exception text.
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__,
                          'hint': 'Check API variables/config and redacted api ledger. No automatic retry.'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
