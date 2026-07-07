import argparse
import csv
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_RUNS = 10
DEFAULT_TEST_RATIO = 0.2


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Validate logreg_train.py/logreg_predict.py with repeated random "
			"train/holdout splits from a labelled training dataset."
		)
	)
	script_dir = Path(__file__).resolve().parent
	parser.add_argument(
		"dataset",
		nargs="?",
		default=script_dir / "dataset_train.csv",
		type=Path,
		help="Labelled CSV dataset to split. Defaults to dataset_train.csv next to this script.",
	)
	parser.add_argument(
		"--runs",
		type=int,
		default=DEFAULT_RUNS,
		help=f"Number of independent random splits to run. Default: {DEFAULT_RUNS}.",
	)
	parser.add_argument(
		"--test-ratio",
		type=float,
		default=DEFAULT_TEST_RATIO,
		help=f"Fraction of rows to hold out for testing. Default: {DEFAULT_TEST_RATIO}.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Optional base random seed for reproducible splits.",
	)
	parser.add_argument(
		"--no-age-hand",
		action="store_true",
		help="Pass 0 to logreg_train.py, matching its option to exclude age and best hand.",
	)
	parser.add_argument(
		"--keep-splits",
		type=Path,
		default=None,
		help="Optional directory where generated split CSVs, models, and predictions are kept.",
	)
	return parser.parse_args()


def read_dataset(path):
	with path.open("r", newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		rows = list(reader)
		fieldnames = reader.fieldnames

	if not fieldnames:
		raise ValueError(f"{path} does not contain a CSV header")
	if "Hogwarts House" not in fieldnames:
		raise ValueError(f"{path} must contain a 'Hogwarts House' column")
	if len(rows) < 2:
		raise ValueError(f"{path} must contain at least two data rows")
	return fieldnames, rows


def write_dataset(path, fieldnames, rows):
	with path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def make_split(rows, test_ratio, rng):
	shuffled_rows = rows[:]
	rng.shuffle(shuffled_rows)
	test_count = round(len(shuffled_rows) * test_ratio)
	test_count = max(1, min(len(shuffled_rows) - 1, test_count))
	return shuffled_rows[test_count:], shuffled_rows[:test_count]


def run_command(command, cwd):
	try:
		subprocess.run(
			command,
			cwd=cwd,
			check=True,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
		)
	except subprocess.CalledProcessError as error:
		print(error.stdout, end="", file=sys.stderr)
		print(error.stderr, end="", file=sys.stderr)
		raise


def read_predictions(path):
	with path.open("r", newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		if reader.fieldnames != ["Index", "Hogwarts House"]:
			raise ValueError(f"{path} has unexpected columns: {reader.fieldnames}")
		return [row["Hogwarts House"] for row in reader]


def score_predictions(test_rows, predictions):
	if len(test_rows) != len(predictions):
		raise ValueError(
			f"prediction count mismatch: got {len(predictions)} predictions for {len(test_rows)} rows"
		)

	correct = 0
	for row, prediction in zip(test_rows, predictions):
		if row["Hogwarts House"] == prediction:
			correct += 1
	return correct, len(test_rows), correct / len(test_rows)


def prepare_run_dir(base_dir, run_number):
	if base_dir is None:
		return tempfile.TemporaryDirectory()

	run_dir = (base_dir / f"run_{run_number:02d}").resolve()
	run_dir.mkdir(parents=True, exist_ok=True)

	class PersistentRunDirectory:
		def __enter__(self):
			return str(run_dir)

		def __exit__(self, exc_type, exc_value, traceback):
			return False

	return PersistentRunDirectory()


def run_validation_once(args, script_dir, fieldnames, rows, run_number, seed):
	rng = random.Random(seed)
	train_rows, test_rows = make_split(rows, args.test_ratio, rng)

	with prepare_run_dir(args.keep_splits, run_number) as run_dir_name:
		run_dir = Path(run_dir_name)
		train_csv = run_dir / "train_split.csv"
		test_csv = run_dir / "test_split.csv"
		model_txt = run_dir / "learned_weights.txt"
		predictions_csv = run_dir / "houses.csv"

		write_dataset(train_csv, fieldnames, train_rows)
		write_dataset(test_csv, fieldnames, test_rows)

		train_command = [
			sys.executable,
			str(script_dir / "logreg_train.py"),
			str(train_csv),
		]
		if args.no_age_hand:
			train_command.append("0")
		run_command(train_command, run_dir)

		run_command(
			[
				sys.executable,
				str(script_dir / "logreg_predict.py"),
				str(test_csv),
				str(model_txt),
			],
			run_dir,
		)

		predictions = read_predictions(predictions_csv)

	correct, total, accuracy = score_predictions(test_rows, predictions)
	return {
		"run": run_number,
		"seed": seed,
		"train_size": len(train_rows),
		"test_size": len(test_rows),
		"correct": correct,
		"total": total,
		"accuracy": accuracy,
	}


def validate_args(args):
	if args.runs < 1:
		raise ValueError("--runs must be at least 1")
	if not 0 < args.test_ratio < 1:
		raise ValueError("--test-ratio must be between 0 and 1")
	if args.keep_splits is not None:
		args.keep_splits = args.keep_splits.resolve()
		args.keep_splits.mkdir(parents=True, exist_ok=True)


def print_result(result):
	print(
		f"Run {result['run']:02d} "
		f"seed={result['seed']} "
		f"train={result['train_size']} "
		f"test={result['test_size']} "
		f"accuracy={result['accuracy'] * 100:.2f}% "
		f"({result['correct']}/{result['total']})"
	)


def print_summary(results):
	accuracies = [result["accuracy"] for result in results]
	print()
	print(f"Average accuracy: {statistics.mean(accuracies) * 100:.2f}%")
	if len(accuracies) > 1:
		print(f"Std deviation:    {statistics.stdev(accuracies) * 100:.2f}%")
	print(f"Best accuracy:    {max(accuracies) * 100:.2f}%")
	print(f"Worst accuracy:   {min(accuracies) * 100:.2f}%")


def main():
	args = parse_args()
	validate_args(args)
	dataset = args.dataset.resolve()
	script_dir = Path(__file__).resolve().parent
	fieldnames, rows = read_dataset(dataset)

	seed_rng = random.Random(args.seed)
	results = []
	for run_number in range(1, args.runs + 1):
		seed = seed_rng.randrange(0, 2**32)
		result = run_validation_once(args, script_dir, fieldnames, rows, run_number, seed)
		results.append(result)
		print_result(result)

	print_summary(results)


if __name__ == "__main__":
	main()
