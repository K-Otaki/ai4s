import argparse
import random
import sys

def calc_pi(N):

    inside = 0

    for i in range(N):
        x = random.random()
        y = random.random()

        if x*x + y*y < 1.0:
            inside += 1

    return 4.0 * inside / N


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("sample count must be greater than 0")
    return value


if __name__ == '__main__':
    print(sys.version)

    parser = argparse.ArgumentParser(description="Estimate pi using Monte Carlo sampling.")
    parser.add_argument(
        "samples",
        nargs="?",
        type=positive_int,
        default=100_000_000,
        help="number of samples to generate (default: 1000000)",
    )
    args = parser.parse_args()

    N = args.samples
    random.seed(42)

    pi = calc_pi(N)

    print("Number of samples:", N)
    print("Estimated pi:", pi)
