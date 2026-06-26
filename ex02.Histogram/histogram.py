#!/usr/bin/env python3

import csv
import sys
import math
import matplotlib.pyplot as plt


NON_COURSE_COLUMNS = {
    "Index",
    "Hogwarts House",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand",
    "Blood Status",
}

HOUSES = [
    "Gryffindor",
    "Hufflepuff",
    "Ravenclaw",
    "Slytherin",
]


def is_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def read_dataset(filename):
    with open(filename, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    courses = [
        column for column in reader.fieldnames
        if column not in NON_COURSE_COLUMNS
    ]

    data = {
        course: {house: [] for house in HOUSES}
        for course in courses
    }

    for row in rows:
        house = row.get("Hogwarts House")

        if house not in HOUSES:
            continue

        for course in courses:
            value = row.get(course)

            if value != "" and is_float(value):
                data[course][house].append(float(value))

    return data, courses


def make_bins(values, number_of_bins=20):
    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        return number_of_bins

    step = (maximum - minimum) / number_of_bins
    return [minimum + i * step for i in range(number_of_bins + 1)]


def plot_histograms(data, courses):
    columns = 3
    rows = math.ceil(len(courses) / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for index, course in enumerate(courses):
        ax = axes[index]

        all_values = []
        for house in HOUSES:
            all_values.extend(data[course][house])

        if not all_values:
            continue

        bins = make_bins(all_values)

        for house in HOUSES:
            scores = data[course][house]

            if scores:
                ax.hist(
                    scores,
                    bins=bins,
                    alpha=0.45,
                    density=True,
                    label=house
                )

        ax.set_title(course)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")

    for index in range(len(courses), len(axes)):
        axes[index].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.savefig("histogram_result.png", dpi=300, bbox_inches="tight")
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py dataset_train.csv")
        sys.exit(1)

    filename = sys.argv[1]

    data, courses = read_dataset(filename)
    plot_histograms(data, courses)


if __name__ == "__main__":
    main()
