"""Compatibility wrapper for the offline CPU classroom runner."""

from nanoscigpt.classroom import RUNNABLE_DOMAINS, run_domain



def main():
    for domain in RUNNABLE_DOMAINS:
        run_domain(domain, "classroom", "data", "out/classroom")


if __name__ == "__main__":
    main()
