#!/usr/bin/env python3

import csv
import sys
import os
import math
import flowplot


HOUSES = [
    "Gryffindor",
    "Hufflepuff",
    "Ravenclaw",
    "Slytherin",
]

COURSES = [
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying",
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

    if reader.fieldnames is None:
        raise ValueError("The CSV file has no header.")

    missing_courses = [course for course in COURSES if course not in reader.fieldnames]
    if missing_courses:
        raise ValueError(f"Missing course columns: {', '.join(missing_courses)}")

    data = {
        course: {house: [] for house in HOUSES}
        for course in COURSES
    }

    for row in rows:
        house = row.get("Hogwarts House")

        if house not in HOUSES:
            continue

        for course in COURSES:
            value = row.get(course)

            if value != "" and is_float(value):
                score = float(value)
            else:
                score = float("nan")

            data[course][house].append(score)

    for course in COURSES:
        for house in HOUSES:
            scores = data[course][house]
            valid_scores = [score for score in scores if math.isfinite(score)]
            if not valid_scores:
                raise ValueError(f"No numeric values for {house} in {course}.")

            mean = sum(valid_scores) / len(valid_scores)
            data[course][house] = [
                score if math.isfinite(score) else mean
                for score in scores
            ]

    return data


def plot_histograms(data, template_path, output_path):
    plot = flowplot.plot(template_path)

    for house in HOUSES:
        for course in COURSES:
            plot.with_data(f"{house}.{course}", data[course][house])

    plot.write_png(output_path)
    print(f"Created: {output_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py dataset_train.csv")
        sys.exit(1)

    filename = sys.argv[1]
    script_directory = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_directory, "HistogramDslr.json")
    output_path = os.path.join(script_directory, "histogram_result.png")

    if not os.path.exists(template_path):
        print(f"Error: template file not found: {template_path}")
        print("Put HistogramDslr.json in the same folder as histogram-template.py.")
        sys.exit(1)

    data = read_dataset(filename)
    plot_histograms(data, template_path, output_path)


if __name__ == "__main__":
    main()
