# Load in relevant analysis packages.
using CSV, DataFrames, HypothesisTests, Plots

# Read in data for onesample t-test
url = "https://raw.githubusercontent.com/HPCurtis/HPCPortfolio/main/Data/Mehr%20Song%20and%20Spelke%202016%20Experiment%201.csv"
df = CSV.read(download(url), DataFrame);

# The data for the specific experiment is in the first 32 rows of the dataframe
red_df = df[1:32, :];

# Data unit test
if{

}

# Calculate one sample t-test with p-value and confidence interval
# using OneSampleTTest
OneSampleTTest(red_df[:,:Baseline_Proportion_Gaze_to_Singer], 0)