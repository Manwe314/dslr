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
│   └── HistogramDslr.json
├── ex02.Scatterplot/
│   ├── dataset_train.csv
│   ├── scatter_plot.py
│   └── ScatterDslr.json
├── ex02.Pairplot/
│   ├── dataset_train.csv
│   ├── pair_plot.py
│   └── PairPlotDslr.json
└── ex03.LogisticRegression/
    ├── dataset_train.csv
    ├── dataset_test.csv
    ├── logreg_train.py
    ├── logreg_predict.py
    ├── validate_logreg.py
    ├── houses.csv
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
HistogramDslr.json
ScatterDslr.json
PairPlotDslr.json
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

This script reads a CSV file and prints statistical information for the numerical **feature** columns used in the DSLR dataset.

The script manually computes:

- `Count`
- `Mean`
- `Std`
- `Min`
- `25%`
- `50%`
- `75%`
- `Max`

It uses Python standard tools such as `csv`, `math`, and loops. It does **not** use `pandas.describe()`.

### Columns used by the script

The current implementation does not analyze every column in the CSV file.

It first ignores non-feature columns:

```text
Index
Hogwarts House
First Name
Last Name
Birthday
Best Hand
Blood Status
```

This means that `Index` is not included in the output, even though it contains numbers. It is treated as an identifier, not as a real numerical feature.

The remaining columns are the course-score columns:

```text
Arithmancy
Astronomy
Herbology
Defense Against the Dark Arts
Divination
Muggle Studies
Ancient Runes
History of Magic
Transfiguration
Potions
Care of Magical Creatures
Charms
Flying
```

For each of these columns, the script keeps only valid finite numerical values.

Ignored values include:

- empty cells,
- text values,
- `NaN`,
- `+inf`,
- `-inf`.

A column is printed only if it contains at least one valid numerical value.

### Mathematical formulas used by the current implementation

For one valid numerical column, let the kept values be:

```text
x_1, x_2, ..., x_n
```

where:

```text
n = number of valid finite numerical values
```

### Count

`Count` is the number of valid values kept in the column.

```text
Count = n
```

The code stores this value as a float:

```text
Count = float(len(values))
```

That is why it is printed with six decimals, for example:

```text
1566.000000
```

### Mean

`Mean` is the arithmetic average of the valid values.

```text
Mean = (x_1 + x_2 + ... + x_n) / n
```

Equivalently:

```text
Mean = (1 / n) × sum(x_i)
```

In the code, this is done manually with a loop:

```text
total = x_1 + x_2 + ... + x_n
Mean = total / n
```

### Std

`Std` is the sample standard deviation, matching the usual `pandas.describe()` behavior.

First, the mean is computed:

```text
mean = (1 / n) × sum(x_i)
```

Then the script computes the sum of squared distances from the mean:

```text
sum_squared_distances = sum((x_i - mean)^2)
```

The sample variance is:

```text
variance = sum_squared_distances / (n - 1)
```

Finally:

```text
Std = sqrt(variance)
```

So the complete formula is:

```text
Std = sqrt(sum((x_i - mean)^2) / (n - 1))
```

The denominator is `n - 1`, not `n`, because the implementation uses the **sample** standard deviation.

If a column has fewer than two valid values, the standard deviation cannot be computed. In that case, the script returns:

```text
nan
```

### Min

`Min` is the smallest valid value in the column.

```text
Min = min(x_1, x_2, ..., x_n)
```

The code finds it manually by starting with the first value, then comparing every following value.

### Max

`Max` is the largest valid value in the column.

```text
Max = max(x_1, x_2, ..., x_n)
```

The code finds it manually by starting with the first value, then comparing every following value.

### Percentiles: 25%, 50%, 75%

The script uses linear interpolation percentiles, close to the default behavior of `pandas.describe()`.

First, it sorts the valid values:

```text
s[0] <= s[1] <= ... <= s[n - 1]
```

For a percentile `q`:

```text
q = 0.25 for 25%
q = 0.50 for 50%
q = 0.75 for 75%
```

The position is computed with zero-based indexing:

```text
position = (n - 1) × q
```

Then:

```text
lower = floor(position)
upper = ceil(position)
```

If `n = 1`, the percentile is simply the only value:

```text
percentile_q = s[0]
```

If `lower = upper`, the percentile is exactly the sorted value at that position:

```text
percentile_q = s[lower]
```

If `lower != upper`, the script interpolates between the two surrounding values:

```text
weight = position - lower
```

```text
percentile_q = s[lower] + (s[upper] - s[lower]) × weight
```

This gives:

```text
25% = percentile with q = 0.25
50% = percentile with q = 0.50
75% = percentile with q = 0.75
```

The `50%` value is also the median.

### Summary of Exercise 01 formulas

| Parameter | Formula / meaning in the code |
|---|---|
| `Count` | `float(n)` |
| `Mean` | `(1 / n) × sum(x_i)` |
| `Std` | `sqrt(sum((x_i - mean)^2) / (n - 1))` |
| `Min` | smallest valid finite value |
| `25%` | linear interpolation percentile with `q = 0.25` |
| `50%` | linear interpolation percentile with `q = 0.50`, also the median |
| `75%` | linear interpolation percentile with `q = 0.75` |
| `Max` | largest valid finite value |

### Output formatting

Every number is printed with six digits after the decimal point:

```text
value = f"{value:.6f}"
```

If the value is `nan`, the script prints:

```text
nan
```

Run:

```bash
cd ex01.Data_Analysis
python3 describe.py dataset_train.csv
```

Example output shape:

```text
            Arithmancy    Astronomy    Herbology  Defense Against the Dark Arts ...
Count      ...          ...          ...        ...
Mean       ...          ...          ...        ...
Std        ...          ...          ...        ...
Min        ...          ...          ...        ...
25%        ...          ...          ...        ...
50%        ...          ...          ...        ...
75%        ...          ...          ...        ...
Max        ...          ...          ...        ...
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
HistogramDslr.json
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
ScatterDslr.json
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
PairPlotDslr.json
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

Prediction script:

```bash
logreg_predict.py
```

Validation script:

```bash
validate_logreg.py
```

This part implements a one-vs-all logistic regression classifier for the four Hogwarts houses.

The training script:

- reads the training dataset,
- converts `Birthday` into age,
- converts `Best Hand` into a numerical value,
- keeps the course scores as numerical features,
- replaces missing values with feature means,
- normalizes features,
- trains one binary logistic regression model per house,
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

The prediction script:

- reads the test dataset,
- reads the normalization values from `learned_weights.txt`,
- reads the learned weights for all four houses,
- normalizes the test data with the same values used during training,
- computes one probability per house,
- chooses the house with the highest probability,
- writes the final predictions into `houses.csv`.

Run:

```bash
python3 logreg_predict.py dataset_test.csv learned_weights.txt
```

Output:

```text
houses.csv
```

The output format is:

```text
Index,Hogwarts House
0,Hufflepuff
1,Ravenclaw
...
```

### Validation with random train/test splits

The script:

```text
validate_logreg.py
```

automates local accuracy testing when the real labels for `dataset_test.csv` are not available.

By default, it:

- reads the labelled `dataset_train.csv`,
- creates 10 independent random splits,
- uses 80% of the rows for training,
- keeps 20% of the rows as a validation set,
- runs `logreg_train.py` on the 80% training split,
- runs `logreg_predict.py` on the 20% validation split,
- compares `houses.csv` predictions with the real `Hogwarts House` values,
- prints one accuracy score per run plus average, standard deviation, best, and worst accuracy.

Run:

```bash
python3 validate_logreg.py
```

Use a fixed seed for reproducible splits:

```bash
python3 validate_logreg.py --seed 42
```

Keep the generated split files, learned weights, and predictions:

```bash
python3 validate_logreg.py --seed 42 --keep-splits KEEP_SPLITS
```

Each kept run is written into its own folder:

```text
KEEP_SPLITS/run_01/train_split.csv
KEEP_SPLITS/run_01/test_split.csv
KEEP_SPLITS/run_01/learned_weights.txt
KEEP_SPLITS/run_01/houses.csv
```

Useful options:

```text
--runs N          number of random splits to test, default 10
--test-ratio R    validation fraction, default 0.2
--seed N          base random seed for repeatable splits
--no-age-hand     train without Birthday age and Best Hand features
--keep-splits DIR keep generated split/model/prediction files
```

Without `--keep-splits`, the script uses temporary folders so it does not overwrite the normal `learned_weights.txt` or `houses.csv` files.

### Features used by the model

By default, the model uses:

- age, computed from `Birthday`,
- handedness, computed from `Best Hand`,
- all course scores.

The optional mode removes age and handedness:

```bash
python3 logreg_train.py dataset_train.csv 0
```

In that mode, only the course scores are used as features.

The current trained model was generated without age and handedness.  
This means the model uses the 13 course-score columns plus the bias term.

### Learned weights file

The file:

```text
learned_weights.txt
```

contains:

1. the vector of feature means,
2. the vector of feature standard deviations,
3. one model for each house,
4. a final feature flag.

The format is:

```text
means
standard_deviations
house_name
weights_for_that_house
house_name
weights_for_that_house
house_name
weights_for_that_house
house_name
weights_for_that_house
0_or_1
```

The final flag means:

```text
1 -> the model was trained with age and handedness
0 -> the model was trained without age and handedness
```

This is important because the prediction script must build the same feature vector shape as the training script.

### Mathematical formulas used for logistic regression

For one student, the feature vector is:

```text
x = [1, x_1, x_2, ..., x_n]
```

The first value is always:

```text
1
```

This is the bias term.

For one house, the model learns a weight vector:

```text
w = [w_0, w_1, w_2, ..., w_n]
```

The linear score is:

```text
z = x · w
```

which means:

```text
z = w_0 + x_1w_1 + x_2w_2 + ... + x_nw_n
```

The sigmoid function converts this score into a probability:

```text
prediction = 1 / (1 + e^(-z))
```

For one-vs-all classification, the script trains four binary classifiers:

```text
Hufflepuff  vs all other houses
Ravenclaw   vs all other houses
Slytherin   vs all other houses
Gryffindor  vs all other houses
```

For each house, the expected output is:

```text
y = 1 if the student belongs to that house
y = 0 otherwise
```

### Gradient descent

For all students, the prediction vector is:

```text
predictions = sigmoid(Xw)
```

The error is:

```text
error = predictions - y
```

The gradient is:

```text
gradient = (X^T × error) / m
```

where:

```text
m = number of students
```

The weights are updated with:

```text
w = w - learning_rate × gradient
```

The current implementation uses:

```text
iterations = 10000
learning_rate = 0.001
```

### Normalization

Before training, each feature is normalized:

```text
normalized_value = (value - mean) / standard_deviation
```

Missing values are replaced with the feature mean before normalization.

The same means and standard deviations are saved into `learned_weights.txt` and reused during prediction.

This matters because the prediction data must be transformed in exactly the same way as the training data.

### Prediction rule

During prediction, the script computes four probabilities:

```text
P(Hufflepuff)
P(Ravenclaw)
P(Slytherin)
P(Gryffindor)
```

The final predicted house is the one with the highest probability:

```text
predicted_house = house with max probability
```

### Current logistic regression result

The generated `houses.csv` contains predictions for all 400 students in `dataset_test.csv`.

The test dataset does not contain the real Hogwarts house labels, so the true test accuracy cannot be computed directly from `dataset_test.csv`.

As a sanity check, the trained model was evaluated on `dataset_train.csv`, where the real labels are known.

Result:

```text
1571 / 1600 correct = 98.19%
```

The training confusion matrix was:

```text
                 Predicted
True          Huf  Rav  Sly  Gry
Hufflepuff    525   1    1    2
Ravenclaw       4 435    2    2
Slytherin       3   5  293    0
Gryffindor      4   5    0  318
```

The prediction distribution on `dataset_test.csv` was:

```text
Hufflepuff: 144 / 400 = 36.0%
Ravenclaw:  114 / 400 = 28.5%
Gryffindor:  79 / 400 = 19.8%
Slytherin:   63 / 400 = 15.8%
```

This distribution is close to the distribution of the training set, which is a useful consistency check.

## Generated files

The project may generate:

```text
*.png
learned_weights.txt
houses.csv
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

# Logistic regression training without age and handedness
python3 logreg_train.py dataset_train.csv 0

# Logistic regression prediction
python3 logreg_predict.py dataset_test.csv learned_weights.txt

# Logistic regression validation on random 80/20 splits
python3 validate_logreg.py --seed 42
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

The logistic regression part now trains all four one-vs-all classifiers and generates `houses.csv`.

The final test accuracy cannot be measured from `dataset_test.csv` alone because the house labels in that file are empty.
