import sys
import csv
import numpy as np
from datetime import date, datetime

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
	"Hufflepuff",
	"Ravenclaw",
	"Slytherin",
	"Gryffindor"
]

house_indexes = {
	house: index
	for index, house in enumerate(house_names)
}

current_date = date.today()

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

def get_age_in_years(birthday):
	birth_date = datetime.strptime(birthday, "%Y-%m-%d").date()
	age = current_date.year - birth_date.year
	if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
		age -= 1
	return age

def best_hand_to_number(best_hand):
	if best_hand == "Right":
		return 1.0
	if best_hand == "Left":
		return 0.0
	return np.nan

def get_house_vector(house):
	house_vector = {
		house_name: 0
		for house_name in house_names
	}
	house_vector[house] = 1
	return house_vector

def normalize_student_vectors(raw_student_vectors):
	student_vectors = np.array(raw_student_vectors, dtype=float)
	features = student_vectors[:, 1:]
	means = mean(features, axis=0, ignore_nan=True)
	standard_deviations = std(features, axis=0, ignore_nan=True)
	standard_deviations[standard_deviations == 0] = 1

	missing_values = np.isnan(features)
	features[missing_values] = np.take(means, np.where(missing_values)[1])
	student_vectors[:, 1:] = (features - means) / standard_deviations
	return student_vectors, means, standard_deviations

def load_training_resources(fileName, use_age_and_hand):
	raw_student_vectors = []
	y_vectors = []

	with open(fileName, "r", newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		for row in reader:
			student_vector = [1.0]
			if use_age_and_hand:
				student_vector.append(float(get_age_in_years(row["Birthday"])))
				student_vector.append(best_hand_to_number(row["Best Hand"]))
			for col in score_cols:
				student_vector.append(float(row[col]) if row[col] != "" else np.nan)

			raw_student_vectors.append(student_vector)
			y_vectors.append(get_house_vector(row["Hogwarts House"]))

	student_vectors, means, standard_deviations = normalize_student_vectors(raw_student_vectors)
	weights = np.zeros(student_vectors.shape[1])

	return {
		"weights": weights,
		"student_vectors": student_vectors,
		"y_vectors": y_vectors,
		"normalization_means": means,
		"normalization_standard_deviations": standard_deviations
	}

def train_one_vs_all(weights, student_vectors, y_vectors, iterations, learning_rate, house):
	X = np.array(student_vectors, dtype=float)
	weights = np.array(weights, dtype=float)
	y = np.array([student_y[house] for student_y in y_vectors], dtype=float)
	m = len(X)
	for _ in  range(iterations):
		zValues = X @ weights
		predictions = 1 / (1 + np.exp(-zValues))
		error = predictions - y
		gradient = (X.T @ error) / m
		weights = weights - learning_rate * gradient
			
	return weights

def save_learned_weights(fileName, normalization_means, normalization_standard_deviations, weights):
	with open(fileName, "w", encoding="utf-8") as file:
		file.write(",".join(str(value) for value in normalization_means) + "\n")
		file.write(",".join(str(value) for value in normalization_standard_deviations) + "\n")
		file.write(",".join(str(value) for value in weights) + "\n")

def main():
	if len(sys.argv) != 2 and len(sys.argv) != 3:
		print('Usage: $> python logreg_train.py <dataset.csv> [optional 0, to not include hand and age]')
		sys.exit(1)
	use_hand_and_age = True
	if len(sys.argv) == 3 and sys.argv[2] == "0":
		use_hand_and_age = False
	fileName = sys.argv[1]
	resources = load_training_resources(fileName, use_hand_and_age)

 
	# turn this into a loop over all 4 houses
	house_models = []
	for house in house_names:
		weights = train_one_vs_all(
			resources["weights"],
			resources["student_vectors"],
			resources["y_vectors"],
			10000,
			0.001,
			house
		)
		house_models.append((weights, house))

	with open("learned_weights.txt", "w", encoding="utf-8") as file:
		file.write(",".join(str(value) for value in resources["normalization_means"]) + "\n")
		file.write(",".join(str(value) for value in resources["normalization_standard_deviations"]) + "\n")
		for model in house_models:
			file.write(model[1] + "\n")
			file.write(",".join(str(value) for value in model[0]) + "\n")
		if use_hand_and_age:
			file.write("1\n")
		else:
			file.write("0\n")



if __name__ == "__main__":
	main()
