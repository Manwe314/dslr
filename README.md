# DSLR - Data Science x Logistic Regression

This repository contains a Python implementation of the 42 School **DSLR** project: **Data Science x Logistic Regression - Harry Potter and the Data Scientist**.

The goal of the project is to explore a Hogwarts student dataset, visualize the score distributions, and begin building a logistic-regression-based Sorting Hat capable of predicting a student's Hogwarts house.

## Project overview

The subject asks for three main parts:

1. **Data analysis**
   - Recreate a `describe`-style statistical summary without using high-level helper functions that do the work automatically.

2. **Data visualization**
   - Produce histograms, scatter plots, and a pair plot to understand the dataset and choose useful features.

3. **Logistic regression**
   - Train a one-vs-all logistic regression classifier using gradient descent.

## Repository structure

```text
DSLR/
├── en.subject.pdf
├── README.md
├── .gitignore
├── ex01.Data_Analysis/
│   ├── dataset_train.csv
│   └── describe.py
├── ex02.Histogram/
│   ├── dataset_train.csv
│   ├── histogram.py
│   ├── histogram-template.py
│   └── histogramDslr.json
├── ex02.Scatterplot/
│   ├── dataset_train.csv
│   ├── scatter_plot.py
│   └── scatterDslr.json
├── ex02.Pairplot/
│   ├── dataset_train.csv
│   ├── pair_plot.py
│   └── pairPlotDslr.json
└── ex03.LogisticRegression/
    ├── dataset_train.csv
    ├── dataset_test.csv
    ├── logreg_train.py
    └── learned_weights.txt
```

## Dataset

The training dataset contains **1600 students** and includes:

- personal information:
  - `Index`
  - `Hogwarts House`
  - `First Name`
  - `Last Name`
  - `Birthday`
  - `Best Hand`

- course scores:
  - `Arithmancy`
  - `Astronomy`
  - `Herbology`
  - `Defense Against the Dark Arts`
  - `Divination`
  - `Muggle Studies`
  - `Ancient Runes`
  - `History of Magic`
  - `Transfiguration`
  - `Potions`
  - `Care of Magical Creatures`
  - `Charms`
  - `Flying`

The houses are:

```text
Gryffindor
Hufflepuff
Ravenclaw
Slytherin
```

## Requirements

The scripts use Python 3.

Recommended setup:

```bash
sudo apt install python3-venv python3-pip

python3 -m venv venv
source venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install numpy matplotlib FlowPlotPy
```

Important note:

```bash
python3 -m pip install FlowPlotPy
```

installs a package that is imported in Python as:

```python
import flowplot
```

Do not confuse it with the different package named `flowplot`.


## FlowPlot GUI template exporter

Exercise 02 uses several `*.json` files as plot templates:

```text
histogramDslr.json
scatterDslr.json
pairPlotDslr.json
```

These JSON files are not datasets. They are **plot templates** used by `FlowPlotPy` to draw the final charts.

The templates were created with the author's graphical program-Levan Kukhaleishvili (download from here: https://github.com/Manwe314/FlowPlot):

```text
FlowPlotGUI-0.9.11-x86_64.AppImage
```

This program allows the user to design plots visually and export them as JSON template files.  
The Python scripts then load these templates and inject real data from `dataset_train.csv`.

### Role of the JSON templates

The `*.json` templates define the visual structure of each plot, for example:

- the figure title,
- the datasets expected by the plot,
- the x-axis and y-axis labels,
- the plot layers,
- the type of chart,
- the visual style,
- the fields that will receive the Python data.

The Python scripts do not create the full plot design from zero.  
Instead, they do this:

```python
plot = flowplot.plot("template.json")
plot.with_data("dataset.field", values)
plot.write_png("output.png")
```

So the workflow is:

```text
FlowPlotGUI AppImage
        ↓
export .json template
        ↓
Python script loads the template
        ↓
Python injects dataset_train.csv values
        ↓
FlowPlotPy renders the final .png image
```

### How to run the FlowPlot GUI on Linux

From the folder containing the AppImage:

```bash
chmod +x FlowPlotGUI-0.9.11-x86_64.AppImage
./FlowPlotGUI-0.9.11-x86_64.AppImage
```

If Linux refuses to run the AppImage because of FUSE, install FUSE:

```bash
sudo apt install libfuse2
```

Then run the AppImage again:

```bash
./FlowPlotGUI-0.9.11-x86_64.AppImage
```

### How the templates are used in Exercise 02

| Exercise | Template file | Python script | Purpose |
|---|---|---|---|
| Histogram | `histogramDslr.json` | `histogram-template.py` | Draw score distributions by house |
| Scatter plot | `scatterDslr.json` | `scatter_plot.py` | Draw relationships between two course scores |
| Pair plot | `pairPlotDslr.json` | `pair_plot.py` | Draw several feature relationships together |

### Important template rule

The names used in the JSON template must match the names used in the Python script.

For example, if the template contains a dataset named:

```text
main
```

with fields:

```text
x
y
```

then the Python script must inject data using:

```python
plot.with_data("main.x", x_values)
plot.with_data("main.y", y_values)
```

If the template contains:

```text
resonance.response
```

then the Python script must use:

```python
plot.with_data("resonance.response", values)
```

If the dataset or field names do not match, the plot cannot be generated correctly.

### Why this is useful

Using template files separates the project into two parts:

1. **Plot design**
   - done visually with `FlowPlotGUI-0.9.11-x86_64.AppImage`

2. **Data processing**
   - done in Python with `csv`, `numpy`, and `FlowPlotPy`

This makes the Exercise 02 scripts cleaner because the Python code focuses mainly on:

- reading the CSV file,
- selecting the right columns,
- computing useful statistics,
- injecting the values into the template,
- exporting the final PNG files.

## Exercise 01 - Data Analysis

Folder:

```bash
ex01.Data_Analysis/
```

Script:

```bash
describe.py
```

This script reads a CSV file and prints statistical information for all numerical columns.

It computes:

- `Count`
- `Mean`
- `Std`
- `Min`
- `25%`
- `50%`
- `75%`
- `Max`

The implementation manually computes the statistics using Python standard tools. It does not use `pandas.describe()`.

### Mathematical formulas used for data analysis

For each numerical column, the script first removes empty or non-numerical values.

Let the remaining values be:

```text
x_1, x_2, ..., x_n
```

where `n` is the number of valid numerical values.

### Count

`Count` is the number of valid numerical values in the column.

```text
Count = n
```

Empty values are ignored.

### Mean

`Mean` is the arithmetic average.

```text
Mean = (x_1 + x_2 + ... + x_n) / n
```

Equivalently:

```text
Mean = (1 / n) × sum(x_i)
```

### Std

`Std` is the sample standard deviation, like `pandas.describe()`.

First, compute the mean:

```text
mean = (1 / n) × sum(x_i)
```

Then compute the sample variance:

```text
variance = sum((x_i - mean)^2) / (n - 1)
```

Finally:

```text
Std = sqrt(variance)
```

So:

```text
Std = sqrt(sum((x_i - mean)^2) / (n - 1))
```

The denominator is `n - 1`, not `n`, because this is the **sample** standard deviation.

If there is only one valid value, the standard deviation is undefined, so the script returns:

```text
nan
```

### Min

`Min` is the smallest value in the column.

```text
Min = min(x_1, x_2, ..., x_n)
```

### Max

`Max` is the largest value in the column.

```text
Max = max(x_1, x_2, ..., x_n)
```

### Percentiles: 25%, 50%, 75%

The script sorts the values first:

```text
s_1 <= s_2 <= ... <= s_n
```

For a percentile `q`, with:

```text
q = 0.25 for 25%
q = 0.50 for 50%
q = 0.75 for 75%
```

the position is:

```text
position = (n - 1) × q
```

Then:

```text
lower = floor(position)
upper = ceil(position)
```

If `lower = upper`, the percentile is exactly the value at that position.

```text
percentile_q = s_position
```

If `lower != upper`, the script uses linear interpolation:

```text
weight = position - lower
```

```text
percentile_q = s_lower + (s_upper - s_lower) × weight
```

This gives:

```text
25% = percentile with q = 0.25
50% = percentile with q = 0.50
75% = percentile with q = 0.75
```

The `50%` value is also the median.

### Summary of parameters

| Parameter | Formula / meaning |
|---|---|
| `Count` | `n` |
| `Mean` | `(1 / n) × sum(x_i)` |
| `Std` | `sqrt(sum((x_i - mean)^2) / (n - 1))` |
| `Min` | smallest value |
| `25%` | percentile with `q = 0.25` |
| `50%` | percentile with `q = 0.50`, also the median |
| `75%` | percentile with `q = 0.75` |
| `Max` | largest value |


Run:

```bash
cd ex01.Data_Analysis
python3 describe.py dataset_train.csv
```

Example output shape:

```text
              Index     Arithmancy      Astronomy      Herbology ...
Count   ...
Mean    ...
Std     ...
Min     ...
25%     ...
50%     ...
75%     ...
Max     ...
```

## Exercise 02.1 - Histogram

Folder:

```bash
ex02.Histogram/
```

There are two histogram scripts.

### `histogram.py`

This version uses `matplotlib`.

It creates a single image containing histograms for every course, grouped by Hogwarts house.

Run:

```bash
cd ex02.Histogram
python3 histogram.py dataset_train.csv
```

Output:

```text
histogram_result.png
```

### `histogram-template.py`

This version uses the FlowPlotPy template:

```text
histogramDslr.json
```

The script dynamically builds one histogram chart per course and injects the house data into the FlowPlotPy template.

Run:

```bash
cd ex02.Histogram
python3 histogram-template.py dataset_train.csv
```

Expected outputs:

```text
histogram_arithmancy.png
histogram_astronomy.png
histogram_herbology.png
...
```

### Mathematical formulas used for histograms

A histogram represents how numerical scores are distributed by grouping values into intervals called **bins**.

For one course and one house, let the scores be:

```text
x_1, x_2, ..., x_n
```

The minimum and maximum values are:

```text
min = min(x_1, x_2, ..., x_n)
max = max(x_1, x_2, ..., x_n)
```

If the histogram uses `k` bins, the bin width is:

```text
h = (max - min) / k
```

The bin intervals are:

```text
[min, min + h[
[min + h, min + 2h[
[min + 2h, min + 3h[
...
[min + (k - 1)h, max]
```

For each bin, the histogram counts how many scores fall inside the interval:

```text
count_j = number of values x_i such that x_i belongs to bin j
```

If the histogram is normalized as a density, the height of bin `j` is:

```text
density_j = count_j / (n × h)
```

This normalization makes the total area of the histogram equal to `1`:

```text
sum(density_j × h) = 1
```

In the DSLR project, histograms are used to compare the score distributions of the four houses for each course. A useful course for classification is often a course where the houses have visibly different distributions.

Conclusion on the result:
The Hogwarts course with the most homogeneous score distribution between all four houses is:

Care of Magical Creatures
In the histogram, the four houses:

Gryffindor
Hufflepuff
Ravenclaw
Slytherin

have distributions that almost completely overlap for Care of Magical Creatures.

Why?

For this course:

the curves are centered around the same zone, roughly near 0;
the spreads are very similar;
no house is clearly separated from the others;
the colors overlap strongly.

So this course does not distinguish the houses well.

Conclusion
Care of Magical Creatures
is the most homogeneous course between all four houses.
For the DSLR project, this means it is probably not a very useful feature for predicting the Hogwarts House, because all houses behave almost the same in this course.

## Exercise 02.2 - Scatter plot

Folder:

```bash
ex02.Scatterplot/
```

Script:

```bash
scatter_plot.py
```

Template:

```bash
scatterDslr.json
```

This script computes the Pearson correlation coefficient for every pair of course-score columns.

It selects three pairs:

1. the pair with the highest positive correlation,
2. the pair with the strongest negative correlation,
3. the pair with the correlation closest to zero.

For each selected pair, it creates a scatter plot using FlowPlotPy.

Run:

```bash
cd ex02.Scatterplot
python3 scatter_plot.py dataset_train.csv
```

Expected outputs:

```text
scatter_output_<coefficient>.png
```

The coefficient is included in the output filename.

### Mathematical formulas used for scatter plots

A scatter plot represents the relationship between two numerical variables.

For two courses `X` and `Y`, each student gives one point:

```text
(x_i, y_i)
```

where:

```text
x_i = score of student i in course X
y_i = score of student i in course Y
```

The script compares every pair of courses using the **Pearson correlation coefficient**.

For two lists of values:

```text
X = x_1, x_2, ..., x_n
Y = y_1, y_2, ..., y_n
```

the means are:

```text
x_mean = (1 / n) × sum(x_i)
y_mean = (1 / n) × sum(y_i)
```

The Pearson correlation coefficient is:

```text
r = sum((x_i - x_mean)(y_i - y_mean))
    / sqrt(sum((x_i - x_mean)^2) × sum((y_i - y_mean)^2))
```

Interpretation:

```text
r close to 1   -> strong positive linear correlation
r close to -1  -> strong negative linear correlation
r close to 0   -> weak or no linear correlation
```

The script selects:

1. the pair with the highest positive correlation,
2. the pair with the strongest negative correlation,
3. the pair whose correlation is closest to `0`.

In the DSLR project, scatter plots help identify relationships between courses and detect whether some features carry similar information.


## Exercise 02.3 - Pair plot

Folder:

```bash
ex02.Pairplot/
```

Script:

```bash
pair_plot.py
```

Template:

```bash
pairPlotDslr.json
```

This script chooses four useful features by comparing how much each course separates the four houses.

For each course, it computes:

- the average score for each house,
- the standard deviation inside each house,
- a separation score based on differences between house averages.

It then keeps the four best-scoring features and builds a pair plot with FlowPlotPy.

Run:

```bash
cd ex02.Pairplot
python3 pair_plot.py dataset_train.csv
```

Output:

```text
pair_plot_output.png
```

### Mathematical formulas used for pair plots

A pair plot displays several scatter plots at once. It compares multiple selected features pair by pair.

If the selected features are:

```text
F_1, F_2, F_3, F_4
```

then the pair plot shows relationships such as:

```text
F_1 vs F_2
F_1 vs F_3
F_1 vs F_4
F_2 vs F_3
F_2 vs F_4
F_3 vs F_4
```

The script chooses useful features by measuring how well each course separates the Hogwarts houses.

For one course and one house `H`, the mean score is:

```text
mean_H = (1 / n_H) × sum(x_i)
```

where `n_H` is the number of students in house `H` with a valid score for that course.

The standard deviation inside one house is:

```text
std_H = sqrt((1 / n_H) × sum((x_i - mean_H)^2))
```

To compare houses, the script computes the absolute difference between their means:

```text
distance(H_a, H_b) = |mean_Ha - mean_Hb|
```

A course is considered more useful when the average scores of the houses are far apart compared with the internal variation of the scores.

A simplified separation idea is:

```text
separation = sum(|mean_Ha - mean_Hb| for all pairs of houses)
```

A stronger version also takes dispersion into account:

```text
normalized_separation =
    sum(|mean_Ha - mean_Hb| / (std_Ha + std_Hb))
```

The goal is to choose courses where different houses form clearer groups. In the DSLR project, the pair plot helps visually check whether the chosen features may be useful for logistic regression.


## Exercise 03 - Logistic Regression

Folder:

```bash
ex03.LogisticRegression/
```

Script:

```bash
logreg_train.py
```

This script starts the logistic regression part of the project.

It:

- reads the training dataset,
- converts `Birthday` into age,
- converts `Best Hand` into a numerical value,
- keeps the course scores as numerical features,
- replaces missing values with feature means,
- normalizes features,
- trains logistic regression with gradient descent,
- saves normalization values and learned weights into `learned_weights.txt`.

Run:

```bash
cd ex03.LogisticRegression
python3 logreg_train.py dataset_train.csv
```

Optional mode without age and hand:

```bash
python3 logreg_train.py dataset_train.csv 0
```

Output:

```text
learned_weights.txt
```

## Current logistic regression status

The current `logreg_train.py` implementation trains only the classifier for:

```text
Gryffindor
```

The code contains a comment indicating that this should be extended into a loop over all four houses.

The project subject also expects a prediction script named:

```text
logreg_predict.py
```

At the current state of the repository, this file is not present.

So the logistic regression part is started, but not fully complete yet.

## Generated files

The project may generate:

```text
*.png
learned_weights.txt
```

The `.gitignore` already ignores PNG outputs and common Python cache or virtual environment files.

## Useful commands

Run all current scripts manually:

```bash
# Data analysis
cd ex01.Data_Analysis
python3 describe.py dataset_train.csv
cd ..

# Histogram with matplotlib
cd ex02.Histogram
python3 histogram.py dataset_train.csv

# Histogram with FlowPlotPy template
python3 histogram-template.py dataset_train.csv
cd ..

# Scatter plot
cd ex02.Scatterplot
python3 scatter_plot.py dataset_train.csv
cd ..

# Pair plot
cd ex02.Pairplot
python3 pair_plot.py dataset_train.csv
cd ..

# Logistic regression training
cd ex03.LogisticRegression
python3 logreg_train.py dataset_train.csv
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'flowplot'`

Install FlowPlotPy inside the active virtual environment:

```bash
python3 -m pip install FlowPlotPy
```

Then test:

```bash
python3 -c "import flowplot; print(hasattr(flowplot, 'plot'))"
```

The result should be:

```text
True
```

### `ModuleNotFoundError: No module named 'numpy'`

Install NumPy:

```bash
python3 -m pip install numpy
```

### `ModuleNotFoundError: No module named 'matplotlib'`

Install Matplotlib:

```bash
python3 -m pip install matplotlib
```

### Template file not found

The FlowPlotPy scripts expect their JSON templates to be in the same folder as the script.

For example, run `scatter_plot.py` from this folder:

```bash
ex02.Scatterplot/
```

not from the repository root.

## Notes

This repository avoids using high-level functions such as `pandas.describe()` for the data analysis exercise, in line with the project constraints.

For the final mandatory project, the logistic regression section still needs:

- one-vs-all training for all four houses,
- a `logreg_predict.py` script,
- generation of `houses.csv`,
- accuracy evaluation against the expected labels.
