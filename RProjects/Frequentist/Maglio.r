# Import data analysis and visualisation library 
library(ggplot2)
library(ggpubr)
library(dplyr)

# Import data from the github repository.
d <- read.csv('https://raw.githubusercontent.com/HPCurtis/Datasets/main/Maglio%20and%20Polman%202014.csv')

# Convert categorical data to factor so that R functions operate correctly.
d$orientation <- factor(d$orientation)
d$station <- as.factor(d$station)

# Calculate summary statistics (means) by group. 
stationMeans <- d %>% group_by(station) %>% summarise( mean(subjective_distance) )
orientationMeans <- d %>% group_by(orientation) %>% summarise( mean(subjective_distance) )

# Plot group means using Barplots.
stationMeans %>% ggplot(aes(x = station, y = `mean(subjective_distance)`)) + geom_col() +
  theme_bw()

orientationMeans %>% ggplot(aes(x = orientation, y = `mean(subjective_distance)`)) + geom_col() +
  theme_bw()

# Plot data using a bocplot split by orientation and station
ggboxplot(data = d, x = 'station', 
          y = 'subjective_distance', color = 'orientation') + xlab('Station') + ylab('Subjective distance recording')

# Modelling
# Build up models to demonstrate contrast coding approach to 
# linear models of standard ANOVA approaches
summary(lm(subjective_distance ~ orientation, data  = d))
summary(lm(subjective_distance ~ station, data  = d))
summary(lm(subjective_distance ~ orientation*station, data = d))


# ANOVA approach.
summary(aov(subjective_distance ~ orientation*station, data = d))

        