library(lme4)
library(dplyr)
library(ggplot2)
library(data.table)

# Set the url link to the dataset stored at the specifiied repository. 
link <- "https://raw.githubusercontent.com/ebrlab/Statistical-methods-for-research-workers-bayes-for-psychologists-and-neuroscientists/master/wip/Data/Mehr%20Song%20and%20Spelke%202016%20Experiment%201.csv"

# Read in analysis data
df  <- read.csv(url(link), header = TRUE, sep = ",")

# Exploratory analysis of data
# reduce dataframe 

red_df <- df[1:32, ]


# Make id a factor class.
red_df$id <- factor(red_df$id)
red_df <- data.table(red_df)

# Genrate long format version of data.
red_dfL <- melt(red_df, id.vars = 'id', 
        measure.vars = c("Baseline_Proportion_Gaze_to_Singer", 
                        "Test_Proportion_Gaze_to_Singer"),
                        variable.name = "condition",
                        value.name = "gaze")

# Test dataset is the correct nuber of rows 
if (nrow(red_dfL) == 64) {
    print("Dataframe is the correct size")
} else {
    print("Dataframe is the incorrect size. Programme will terminate")
    stop()
}

print(red_dfL)

# Fit Mixed model to get a equivalent of a repeated measure t-test.
lmer(gaze ~ 1 + (1 | id), data = print(red_dfL)
)