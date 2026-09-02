"""Validate one bundled classroom domain without downloading or rewriting it."""

import argparse
import json

from ..classroom import RUNNABLE_DOMAINS, validate_domain_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=RUNNABLE_DOMAINS)
    parser.add_argument("--data_root", default="data")
    args = parser.parse_args()
    report = validate_domain_data(args.domain, args.data_root)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
