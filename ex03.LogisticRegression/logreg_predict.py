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

current_date = date.today()

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

def normalize_prediction_data(raw_data, means, standard_deviations):
	normalized_data = np.array(raw_data, dtype=float)
	features = normalized_data[:, 1:]

	missing_values = np.isnan(features)
	features[missing_values] = np.take(means, np.where(missing_values)[1])
	normalized_data[:, 1:] = (features - means) / standard_deviations
	return normalized_data


def load_prediction_resources(fileName, use_age_and_hand, means, standard_deviations):
	raw_student_vectors = []

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

	student_vectors = normalize_prediction_data(raw_student_vectors, means, standard_deviations)

	return student_vectors

def parse_float_vector(line):
	return np.array([float(value) for value in line.split(",")], dtype=float)

def load_model_resources(fileName):
	with open(fileName, "r", encoding="utf-8") as file:
		lines = [line.strip() for line in file if line.strip()]

	if len(lines) < 4:
		raise ValueError("Model file must contain means, standard deviations, models, and feature flag")

	means = parse_float_vector(lines[0])
	standard_deviations = parse_float_vector(lines[1])
	feature_flag = lines[-1]
	if feature_flag == "1":
		use_age_and_hand = True
		model_lines = lines[2:-1]
	elif feature_flag == "0":
		use_age_and_hand = False
		model_lines = lines[2:-1]
	else:
		use_age_and_hand = len(means) == len(score_cols) + 2
		model_lines = lines[2:]

	house_models = []
	if len(model_lines) % 2 != 0:
		raise ValueError("Each model must contain a house name line followed by a weights line")

	for index in range(0, len(model_lines), 2):
		house = model_lines[index]
		weights = parse_float_vector(model_lines[index + 1])
		house_models.append((weights, house))

	return means, standard_deviations, use_age_and_hand, house_models


def make_predictions(normalized_data, house_models):
	final_predictions = []
	for model in house_models:
		zValues = normalized_data @ model[0]
		predictions = 1 / (1 + np.exp(-zValues))
		final_predictions.append((predictions, model[1]))
	return final_predictions

def pick_house_for_student(final_predictions, index):
	max_prediction = -1.0
	house = None
	for prediction in final_predictions:
		if prediction[0][index] > max_prediction:
			house = prediction[1]
			max_prediction = prediction[0][index]
	return house

def write_output(final_predictions):
	with open("houses.csv", "w", encoding="utf-8") as file:
		file.write("Index,Hogwarts House\n")
		for i in range(len(final_predictions[0][0])):
			file.write(str(i) + ",")
			file.write(pick_house_for_student(final_predictions, i) + "\n")



def main():
	if len(sys.argv) != 3:
		print('Usage: $> python logreg_predict.py <dataset.csv> <learned_weights.txt>')
		sys.exit(1)
	dataFileName = sys.argv[1]
	modelFileName = sys.argv[2]

	means, standard_deviations, use_age_and_hand, house_models = load_model_resources(modelFileName)
	normalized_data = load_prediction_resources(
		dataFileName,
		use_age_and_hand,
		means,
		standard_deviations
	)
	final_predictions = make_predictions(normalized_data, house_models)
	write_output(final_predictions)



if __name__ == "__main__":
	main()
