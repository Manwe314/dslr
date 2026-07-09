import flowplot
import sys
import csv
import numpy as np
import math
import json
import os

score_cols = [
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
    "Flying"
]

house_names = [
	"Gryffindor",
	"Hufflepuff",
	"Ravenclaw",
	"Slytherin"
]

def mean(values, axis=None, ignore_nan=False):
	array = np.array(values, dtype=float)
	if ignore_nan:
		valid_values = ~np.isnan(array)
		totals = np.sum(np.where(valid_values, array, 0.0), axis=axis)
		counts = np.sum(valid_values, axis=axis)
	else:
		totals = np.sum(array, axis=axis)
		counts = array.size if axis is None else array.shape[axis]

	with np.errstate(invalid="ignore", divide="ignore"):
		return np.asarray(totals / counts, dtype=float)

def std(values, axis=None, ignore_nan=False):
	array = np.array(values, dtype=float)
	averages = mean(array, axis=axis, ignore_nan=ignore_nan)
	if axis is None:
		centered_values = array - averages
	else:
		centered_values = array - np.expand_dims(averages, axis=axis)

	squared_distances = centered_values ** 2
	if ignore_nan:
		valid_values = ~np.isnan(array)
		squared_distances = np.where(valid_values, squared_distances, 0.0)
		counts = np.sum(valid_values, axis=axis)
	else:
		counts = array.size if axis is None else array.shape[axis]

	totals = np.sum(squared_distances, axis=axis)
	with np.errstate(invalid="ignore", divide="ignore"):
		return np.asarray(np.sqrt(totals / counts), dtype=float)

def get_features_to_use(data):
	features = []
	houses = data["houses"]
	scores = data["scores"]

	for feature_index, feature_name in enumerate(score_cols):
		averages = []
		standard_deviations = []

		for house in house_names:
			house_values = scores[houses == house, feature_index]
			house_values = house_values[~np.isnan(house_values)]

			if len(house_values) == 0:
				break

			averages.append(float(mean(house_values)))
			standard_deviations.append(float(std(house_values)))

		if len(averages) != len(house_names):
			continue

		average_difference_sum = 0
		for i in range(len(averages)):
			for j in range(i + 1, len(averages)):
				average_difference_sum += abs(averages[i] - averages[j])

		internal_standard_deviation = sum(standard_deviations) / len(standard_deviations)
		if internal_standard_deviation == 0:
			continue

		features.append({
			"name": feature_name,
			"index": feature_index,
			"score": average_difference_sum / internal_standard_deviation,
		})

	features.sort(key=lambda feature: feature["score"], reverse=True)
	return features[:4]

def load_data(fileName):
	with open(fileName, "r", newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		houses = []
		scores = []
		for row in reader:
			student_scores = [row[col] if row[col] != "" else np.nan for col in score_cols]
			houses.append(row["Hogwarts House"])
			scores.append(student_scores)
	return {
		"houses": np.array(houses),
		"scores": np.array(scores, dtype=float)
	}

def build_pair_plot(data, features):
	plot_template = os.path.join(
		os.path.dirname(os.path.abspath(__file__)),
		"PairPlotDslr.json"
	)
	plot = flowplot.plot(plot_template)

	with open(plot_template, "r", encoding="utf-8") as file:
		plot_json = json.load(file)
	panel_indexes = {
		panel["id"]: index
		for index, panel in enumerate(plot_json["panels"])
	}

	for row_index, feature in enumerate(features, start=1):
		panel_index = panel_indexes[f"{row_index}-1"]
		plot.set(f"panels[{panel_index}].yAxis.title.text", feature["name"])
		plot.set(f"panels[{panel_index}].yAxis.title.visible", True)

	for column_index, feature in enumerate(features, start=1):
		panel_index = panel_indexes[f"4-{column_index}"]
		plot.set(f"panels[{panel_index}].xAxis.title.text", feature["name"])
		plot.set(f"panels[{panel_index}].xAxis.title.visible", True)

	feature_indexes = [feature["index"] for feature in features]
	valid_mask = ~np.isnan(data["scores"][:, feature_indexes]).any(axis=1)
	for feature_number, feature in enumerate(features, start=1):
		plot.with_data(f"main.feat{feature_number}", data["scores"][valid_mask, feature["index"]])
	plot.with_data("main.house", data["houses"][valid_mask].tolist())

	return plot

def main():
	if len(sys.argv) != 2:
		print('Usage: $> python pair_plot.py <dataset.csv>')
		sys.exit(1)
	
	fileName = sys.argv[1]

	data = load_data(fileName)

	features = get_features_to_use(data)

	plot = build_pair_plot(data, features)
	plot.write_png("./pair_plot_output.png")





if __name__ == "__main__":
	main()
