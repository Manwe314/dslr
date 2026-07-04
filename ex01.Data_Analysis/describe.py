#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import sys
from typing import Optional


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

# Columns that are not real numerical features for the DSLR analysis.
# They are identifiers or categorical/text information.
NON_FEATURE_COLUMNS = {
    "Index",
    "Hogwarts House",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand",
    "Blood Status",
}


def parse_float(value: Optional[str]) -> Optional[float]:
    """
    Convert a CSV cell to a valid finite float.

    Empty cells, text values, NaN, +inf and -inf are ignored.
    This avoids counting invalid values in columns such as First Name.
    """
    if value is None:
        return None

    value = value.strip()
    if value == "":
        return None

    try:
        number = float(value)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def mean(values: list[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def std(values: list[float], avg: float) -> float:
    """
    Sample standard deviation, like pandas describe().
    Formula: sqrt(sum((x - mean)^2) / (n - 1))
    """
    n = len(values)
    if n < 2:
        return float("nan")

    total = 0.0
    for value in values:
        total += (value - avg) ** 2

    variance = total / (n - 1)
    return math.sqrt(variance)


def min_value(values: list[float]) -> float:
    result = values[0]
    for value in values[1:]:
        if value < result:
            result = value
    return result


def max_value(values: list[float]) -> float:
    result = values[0]
    for value in values[1:]:
        if value > result:
            result = value
    return result


def percentile(values: list[float], q: float) -> float:
    """
    Linear interpolation percentile, close to pandas default behavior.
    q must be between 0 and 1.
    """
    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    position = (n - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return sorted_values[lower]

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
        "Min": min_value(values),
        "25%": percentile(values, 0.25),
        "50%": percentile(values, 0.50),
        "75%": percentile(values, 0.75),
        "Max": max_value(values),
    }


def read_csv(filename: str) -> dict[str, list[float]]:
    """
    Read the CSV file and keep only valid numerical course/features columns.
    """
    data: dict[str, list[float]] = {}

    with open(filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        if reader.fieldnames is None:
            raise ValueError("Empty CSV file")

        for column in reader.fieldnames:
            if column not in NON_FEATURE_COLUMNS:
                data[column] = []

        for row in reader:
            for column in data:
                number = parse_float(row.get(column))
                if number is not None:
                    data[column].append(number)

    return data


def keep_numeric_columns(data: dict[str, list[float]]) -> dict[str, list[float]]:
    """
    Keep only columns with at least one valid numerical value.
    """
    numeric_data: dict[str, list[float]] = {}

    for column, values in data.items():
        if len(values) > 0:
            numeric_data[column] = values

    return numeric_data


def format_number(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def print_stats_table(stats_by_column: dict[str, dict[str, float]]) -> None:
    columns = list(stats_by_column.keys())

    first_col_width = max(len(name) for name in STATS_ORDER) + 2
    column_widths: dict[str, int] = {}

    for column in columns:
        max_value_width = 0
        for stat_name in STATS_ORDER:
            value_width = len(format_number(stats_by_column[column][stat_name]))
            if value_width > max_value_width:
                max_value_width = value_width

        column_widths[column] = max(len(column), max_value_width) + 2
    # Print header
    print("".ljust(first_col_width), end="")
    for column in columns:
        print(column.rjust(column_widths[column]), end="")
    print()
    # Print each statistic row
    for stat_name in STATS_ORDER:
        print(stat_name.ljust(first_col_width), end="")

        for column in columns:
            value = stats_by_column[column][stat_name]
            print(format_number(value).rjust(column_widths[column]), end="")

        print()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py dataset_train.csv")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        data = read_csv(filename)
        numeric_data = keep_numeric_columns(data)

        stats_by_column: dict[str, dict[str, float]] = {}
        for column, values in numeric_data.items():
            stats_by_column[column] = compute_stats(values)

        print_stats_table(stats_by_column)
    except FileNotFoundError:
        print(f"Error: file not found: {filename}", file=sys.stderr)
        sys.exit(1)
    except OSError as error:
        print(f"Error: cannot read file: {error}", file=sys.stderr)
        sys.exit(1)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
