library('mgcv')
library('dplyr')
library('ggplot2')
library('gratia')


d <- read.csv('https://raw.githubusercontent.com/BayesianModelingandComputationInPython/BookCode_Edition1/main/data/bikes_hour.csv')
d <- d[order(d[,'hour']),]

# Standardise data
d[, 'count'] <- scale(d[, 'count'])

d <- d[seq(1, nrow(d), 50), ]

# Visualise the normalised Bike count data
ggplot(data = d, aes(x = d[ ,'hour'], y = d[ , 'count'])) +
  geom_point(alpha=.3) + xlab('hour') + ylab('Normalised Count') + theme_bw()

# Fit gam using MGCV package
M1 <- gam(count ~ s(hour), data = d, family = gaussian)
summary(M1)



sm <- smooth_estimates(M1, smooth = "s(hour)")
# Plot the GAM model fit
sm %>%
  add_confint() %>%
  ggplot(aes(y = est, x = hour)) +
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci),
              alpha = 0.2, fill = "forestgreen") +
  geom_line(colour = "forestgreen", linewidth = 1.5) + ylab('Partial effect') +
  geom_point(data = d, aes(x = d[ ,'hour'], y = d[ , 'count']),alpha=.3 ) + theme_bw()
