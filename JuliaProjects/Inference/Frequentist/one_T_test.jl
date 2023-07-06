# Load in relevant analysis packages.
using CSV, DataFrames, HypothesisTests, Plots

# Read in data for onesample t-test
url = "https://raw.githubusercontent.com/HPCurtis/HPCPortfolio/main/Data/Mehr%20Song%20and%20Spelke%202016%20Experiment%201.csv"
df = CSV.read(download(url), DataFrame);

# Clean the dataset.
# Specify the number of rows from datset for the analysis.
# The data for the specific experiment is in the first 32 rows of the dataframe.
N = 32
red_df = df[1:N, :];

# Data unit test
if nrow(red_df) == N
    print("Dataset is the correct size for one sample t-test")
    print(" Test passed ")
else
    print("Dataset is the incorrect size. Program will not execute")
    print("Test passed")
    exit()
end

print(string(1))
# Calculate one sample t-test with p-value and confidence interval
# using OneSampleTTest. Specifying the population H0 to 0.
results = OneSampleTTest(red_df[:, :Baseline_Proportion_Gaze_to_Singer], 0)
ciP = confint(results)
ciL = ciP[1]
ciH = ciP[2]

# Print the analysis output
print("Baseline gaze proprtion scores were analysed using a one sample t-test \
with population parameter for Hypohtesis test set to 0 with baseline gaze \
     proportion scores having (M = " * string(round(results.xbar, digits = 2)) * ", SD = " * string(round(results.stderr, digits = 2)) * " t(" * string(results.df) * ") = " * string(round(results.t, digits = 2)) * ", p = .001, \
     with 95% CI [" * string(round(ciL, digits = 2)) * ", " * string(round(ciH, digits = 2)) * "]." )
