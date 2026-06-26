#!/usr/bin/env python3

import csv
import sys
import os
import re
import json
import copy
import tempfile
import flowplot


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

HOUSE_STYLES = {
    "Gryffindor": {
        "fillColor": "#d94c4cb4",
        "strokeColor": "#8a1f1f",
    },
    "Hufflepuff": {
        "fillColor": "#f0c94cb4",
        "strokeColor": "#9a7412",
    },
    "Ravenclaw": {
        "fillColor": "#4c7fd9b4",
        "strokeColor": "#1e3f8a",
    },
    "Slytherin": {
        "fillColor": "#4ca86bb4",
        "strokeColor": "#1f6b3a",
    },
}


def is_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def safe_filename(name):
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def read_dataset(filename):
    with open(filename, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if reader.fieldnames is None:
        raise ValueError("The CSV file has no header.")

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


def dataset_name(house):
    return safe_filename(house)


def build_course_template(base_template, course):
    """
    Build a FlowPlot template for one course.

    The uploaded template has one histogram layer.
    Here we duplicate that layer 4 times:
    one layer for each Hogwarts house.
    """
    template = copy.deepcopy(base_template)

    template["figure"]["title"]["text"] = f"{course} score distribution by house"

    template["datasets"] = []

    base_panel = template["panels"][0]
    base_layer = base_panel["layers"][0]

    new_layers = []

    for house in HOUSES:
        name = dataset_name(house)

        template["datasets"].append({
            "name": name,
            "schema": {
                "response": "number"
            }
        })

        layer = copy.deepcopy(base_layer)
        layer["dataset"] = name
        layer["mapping"]["data"]["field"] = "response"

        layer["style"] = HOUSE_STYLES[house]

        new_layers.append(layer)

    base_panel["layers"] = new_layers
    base_panel["xAxis"]["title"]["text"] = "Score"
    base_panel["yAxis"]["title"]["text"] = "Count"

    return template


def write_temp_template(template):
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8"
    )

    json.dump(template, temp_file, indent=2)
    temp_file.close()

    return temp_file.name


def plot_histograms(data, courses, template_path):
    with open(template_path, "r", encoding="utf-8") as file:
        base_template = json.load(file)

    for course in courses:
        has_values = any(data[course][house] for house in HOUSES)

        if not has_values:
            continue

        course_template = build_course_template(base_template, course)
        temp_template_path = write_temp_template(course_template)

        try:
            plot = flowplot.plot(temp_template_path)

            for house in HOUSES:
                values = data[course][house]
                plot.with_data(f"{dataset_name(house)}.response", values)

            output_name = f"histogram_{safe_filename(course)}.png"
            plot.write_png(output_name)

            print(f"Created: {output_name}")

        finally:
            os.remove(temp_template_path)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 histogram-template.py dataset_train.csv")
        sys.exit(1)

    filename = sys.argv[1]
    template_path = "histogramDslr.json"

    if not os.path.exists(template_path):
        print(f"Error: template file not found: {template_path}")
        print("Put histogramDslr.json in the same folder as histogram-template.py.")
        sys.exit(1)

    data, courses = read_dataset(filename)
    plot_histograms(data, courses, template_path)


if __name__ == "__main__":
    main()
