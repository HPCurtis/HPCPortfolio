library('lme4')

# url link to the dataset dowloaded from Kaggle and stored on github.
link <- "https://raw.githubusercontent.com/HPCurtis/HPCStatsPortfolio/main/Data/us_births_2016_2021.csv"

# Read dataset csv to dataframe.
df <- read.csv(url(link), header = TRUE, sep = ",")

# Remove district of columbia from dataset
df <- df[df$State != "District of Columbia", ]
State <- unique(df[, 'State'])
print(colnames(df))

# Generate dataset with just year 2021
df_2021 <- df[df$Year == "2021", ]


# Fit Random intercept model with gaussian likelihood
mod_2021N <- glmer(Number.of.Births ~ 1 + (1 | State) + (1  | Education.Level.of.Mother), data = df_2021, family = gaussian)
#odAllN <- glmer(Number.of.Births ~ 1 + (1 |State), data = df, family = gaussian)

# Fit Random intercept model with poisson likelihood 
# mod_2021P <- glmer(Number.of.Births ~ 1 + (1 | State), data = df_2021, family = poisson)
# modAllP <- glmer(Number.of.Births ~ 1 + (1 | State), data = df, family = poisson)

# # Print normal models.
# print(mod_2021N)
# print(modAllN)
# # Print poisson models.
# print(mod_2021P)
# print(modAllP)


# Compare models AIC using ML 
# Key the models are fit to the same data.
# print(anova(mod_2021N, mod_2021P))
# print(anova(modAllN, modAllP))
#
#print(coef(mod_2021N)$State)