import flowplot
import sys
import csv
import numpy as np
import math

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

def pearson_correlation_coefficient(xValues, yValues, length):
	xMean = sum(xValues) / length
	yMean = sum(yValues) / length

	numerator = sum((xValues[i] - xMean) * (yValues[i] - yMean) for i in range(length))
	denominator_x = math.sqrt(sum((xValues[i] - xMean) ** 2 for i in range(length)))
	denominator_y = math.sqrt(sum((yValues[i] - yMean) ** 2 for i in range(length)))
	if denominator_x == 0 or denominator_y == 0:
		return None
	return numerator / (denominator_x * denominator_y)

def load_data(fileName):
	with open(fileName, "r", newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		data = []
		for row in reader:
			student_scores = [row[col] if row[col] != "" else np.nan for col in score_cols]
			data.append(student_scores)
	return np.array(data, dtype=float)

def get_columns_to_plot(data):
	columns_to_plot = []
	high_positive = None
	high_negative = None
	low_corelation = None
	for i in range(len(score_cols)):
		for j in range(i + 1, len(score_cols)):
			col1 = score_cols[i]
			col2 = score_cols[j]

			xValues = data[:, i]
			yValues = data[:, j]
			valid_mask = ~np.isnan(xValues) & ~np.isnan(yValues)
			xValues = xValues[valid_mask]
			yValues = yValues[valid_mask]

			if len(xValues) == len(yValues) and len(xValues) > 0:
				coefficient = pearson_correlation_coefficient(xValues, yValues, len(xValues))

				if coefficient is None:
					continue

				pair_info = {
					"column1" : col1,
					"column2" : col2,
					"index1" : i,
					"index2" : j,
					"coefficient" : coefficient
				}

				if high_positive is None or coefficient > high_positive["coefficient"]:
					high_positive = pair_info
				
				if high_negative is None or coefficient < high_negative["coefficient"]:
					high_negative = pair_info
				
				if low_corelation is None or abs(coefficient) < abs(low_corelation["coefficient"]):
					low_corelation = pair_info
	columns_to_plot.append(high_positive)
	columns_to_plot.append(high_negative)
	columns_to_plot.append(low_corelation)
	return columns_to_plot

def main():
	if len(sys.argv) != 2:
		print('Usage: $> python scatter_plot.py <dataset.csv>')
		sys.exit(1)
	
	fileName = sys.argv[1]

	data = load_data(fileName)

	columns_to_plot = get_columns_to_plot(data)

	for pair_info in columns_to_plot:
		plot = flowplot.plot("./ScatterDslr.json")
		plot.set("panels[0].yAxis.title.text", pair_info["column1"])
		plot.set("panels[0].xAxis.title.text", pair_info["column2"])
		xValues = data[:, pair_info["index2"]]
		yValues = data[:, pair_info["index1"]]
		valid_mask = ~np.isnan(xValues) & ~np.isnan(yValues)
		plot.with_data("main.y", yValues[valid_mask])
		plot.with_data("main.x", xValues[valid_mask])
		plot.write_png("./scatter_output_" + str(pair_info["coefficient"]) + ".png")



if __name__ == "__main__":
	main()


