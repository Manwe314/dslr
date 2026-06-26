#!/usr/bin/env python3

import csv
import sys
import math


STATS_ORDER = [
    "Count",
    "Mean",
    "Std",
    "Min",
    "25%",
    "50%",
    "75%",
    "Max",
]


def is_number(value: str) -> bool:
    """
    Return True if value can be converted to float.
    Empty strings are ignored.
    """
    if value is None:
        return False

    value = value.strip()

    if value == "":
        return False

    try:
        float(value)
        return True
    except ValueError:
        return False


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float], avg: float) -> float:
    """
    Sample standard deviation, like pandas describe().
    Formula: sqrt(sum((x - mean)^2) / (n - 1))
    """
    n = len(values)

    if n < 2:
        return float("nan")

    variance = sum((x - avg) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def percentile(values: list[float], q: float) -> float:
    """
    Linear interpolation percentile, close to pandas default behavior.
    q must be between 0 and 1.
    """
    if not values:
        return float("nan")

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    position = (n - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return sorted_values[int(position)]

    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]

    weight = position - lower
    return lower_value + (upper_value - lower_value) * weight


def compute_stats(values: list[float]) -> dict[str, float]:
    avg = mean(values)

    return {
        "Count": float(len(values)),
        "Mean": avg,
        "Std": std(values, avg),
        "Min": min(values),
        "25%": percentile(values, 0.25),
        "50%": percentile(values, 0.50),
        "75%": percentile(values, 0.75),
        "Max": max(values),
    }


def read_csv(filename: str) -> dict[str, list[float]]:
    """
    Read the CSV file and keep only numerical values for each column.
    """
    data: dict[str, list[float]] = {}

    with open(filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        if reader.fieldnames is None:
            raise ValueError("Empty CSV file")

        for column in reader.fieldnames:
            data[column] = []

        for row in reader:
            for column, value in row.items():
                if is_number(value):
                    data[column].append(float(value))

    return data


def keep_numeric_columns(data: dict[str, list[float]]) -> dict[str, list[float]]:
    """
    Keep columns that contain at least one numerical value.
    """
    return {
        column: values
        for column, values in data.items()
        if len(values) > 0
    }


def format_number(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"

def print_stats_table(stats_by_column: dict[str, dict[str, float]]) -> None:
    columns = list(stats_by_column.keys())

    first_col_width = 10
    col_width = 15

    # Print header
    print("".ljust(first_col_width), end="")
    for column in columns:
        print(column[:col_width - 1].rjust(col_width), end="")
    print()

    # Print each statistic row
    for stat_name in STATS_ORDER:
        print(stat_name.ljust(first_col_width), end="")

        for column in columns:
            value = stats_by_column[column][stat_name]
            print(format_number(value).rjust(col_width), end="")

        print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py dataset_train.csv")
        sys.exit(1)

    filename = sys.argv[1]

    data = read_csv(filename)
    numeric_data = keep_numeric_columns(data)

    stats_by_column = {}

    for column, values in numeric_data.items():
        stats_by_column[column] = compute_stats(values)

    print_stats_table(stats_by_column)


if __name__ == "__main__":
    main()
