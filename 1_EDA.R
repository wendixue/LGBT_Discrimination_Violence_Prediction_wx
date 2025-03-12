### EDA
#------------------------------------------------------------
# 0) Load libraries
#------------------------------------------------------------
library(tidyverse)  # For data manipulation and piping
library(psych)      # For statistical functions
library(corrplot)   # For correlation plots
library(factoextra) # For PCA visualization
library(ranger) # For predictive modeling
library(dplyr)      # For data manipulation
library(ggplot2)    # For visualization
library(viridis)    # For color palettes
library(caret)      # For modeling tools
library(glmnet)     # For regularized regression

#------------------------------------------------------------
# 1) Read and prepare data
#------------------------------------------------------------
# Import data from source file
data <- haven::read_dta("/Users/xuewendi/Desktop/MACS 30100 Project Wendi X/EU2023.dta")

# Remove variables only related to trans and intersex people for this analysis
df_clean <- data|>
  dplyr::select(-dplyr::starts_with("TR"), -dplyr::starts_with("IX"))

#------------------------------------------------------------
# 2) Process demographic variables
#------------------------------------------------------------
# 2.1) Gender Identity
df_clean$A2 <- as_factor(df_clean$A2)
df_clean <- df_clean |>
  filter(A2 != "Do not identify as woman, man, or non-binary, please specify how would you identify") |>
  mutate(
    gender = A2
  )
levels(df_clean$gender) <- c("Woman / Girl", "Man / Boy", "Trans woman / girl", 
                             "Trans man / boy", "Non-binary/fluid", "Other")

# 2.2) Sexual Orientation
df_clean <- df_clean |>
  mutate(
    # Ensure A6 is character type
    A6_char = as.character(A6),
    # Create sexuality variable
    sexuality = case_when(
      A6_char == "1" ~ "Lesbian",
      A6_char == "2" ~ "Gay",
      A6_char == "3" ~ "Asexual",
      A6_char == "4" ~ "Bisexual",
      A6_char == "5" ~ "Pansexual",
      A6_char == "6" ~ "Heterosexual/Straight",
      A6_char == "7" ~ "Other",
      TRUE ~ NA_character_
    )
  )

# Convert to factor with desired levels
df_clean$sexuality <- factor(df_clean$sexuality,
                             levels = c("Lesbian", "Gay", "Asexual", "Bisexual",
                                        "Pansexual", "Heterosexual/Straight", "Other"),
                             exclude = NULL)

#------------------------------------------------------------
# 3) Filter dataset to focus on LGBTQIA+ individuals
#------------------------------------------------------------
# Remove cishet individuals (non-LGBTQIA+)
df_filtered <- df_clean |>
  filter(
    !(
      (trimws(gender) == trimws("Man / Boy") &
         trimws(sexuality) == trimws("Heterosexual/Straight")) |
        (trimws(gender) == trimws("Woman / Girl") &
           trimws(sexuality) == trimws("Heterosexual/Straight"))
    )
  )

#------------------------------------------------------------
# 4) Process minority status variables
#------------------------------------------------------------
# Create binary indicators for each minority type
df_filtered <- df_filtered |>
  mutate(
    skin_minority = ifelse(A13_A == 1, 1, 0),
    ethnic_minority = ifelse(A13_B == 1, 1, 0),
    refugee = ifelse(A13_C == 1, 1, 0),
    religious_minority = ifelse(A13_D == 1, 1, 0),
    disability = ifelse(A13_E == 1, 1, 0),
    other_minority = ifelse(A13_F == 1, 1, 0),
    no_minority = ifelse(A13_G == 1, 1, 0),
    dont_know = ifelse(A13_H == 1, 1, 0)
  )

# Create refined minority classification
df_filtered <- df_filtered |>
  mutate(
    minority = case_when(
      no_minority == 1 | dont_know == 1 ~ "No minority",
      skin_minority == 1 & ethnic_minority == 0 & refugee == 0 &
        religious_minority == 0 & disability == 0 ~ "Skin color only",
      ethnic_minority == 1 & skin_minority == 0 & refugee == 0 &
        religious_minority == 0 & disability == 0 ~ "Ethnic/migrant only",
      religious_minority == 1 & skin_minority == 0 & ethnic_minority == 0 &
        refugee == 0 & disability == 0 ~ "Religious only",
      disability == 1 & skin_minority == 0 & ethnic_minority == 0 &
        refugee == 0 & religious_minority == 0 ~ "Disability only",
      refugee == 1 ~ "Refugee/Asylum seeker",
      (skin_minority + ethnic_minority + religious_minority + disability) > 1
      ~ "Multiple minorities",
      other_minority == 1 ~ "Other minority",
      TRUE ~ "Multiple minorities"
    )
  )

# Convert to factor with ordered levels
df_filtered$minority <- factor(df_filtered$minority,
                               levels = c("No minority",
                                          "Skin color only",
                                          "Ethnic/migrant only",
                                          "Religious only",
                                          "Disability only",
                                          "Refugee/Asylum seeker",
                                          "Multiple minorities",
                                          "Other minority"))

#------------------------------------------------------------
# 5) Process other demographic variables
#------------------------------------------------------------
# 5.1) Age
df_filtered$age <- df_filtered$A1

# 5.2) Educational Level
df_filtered$education <- as_factor(df_filtered$G1)

# 5.3) Employment Status (binary: employed/not employed)
df_filtered <- df_filtered |>
  mutate(employed = ifelse(G2 %in% c(
    "In full-time paid work (including on paternity or other temporary leave)",
    "In part-time or temporary paid work",
    "Self-employed"), 1, 0)) |>
  filter(!G2 %in% c("Don't know", "Prefer not to say"))

# 5.4) Economic Level
df_filtered <- df_filtered |>
  mutate(
    econ_level = as.character(G19),
    econ_level = na_if(econ_level, "Don't know"),
    econ_level = na_if(econ_level, "Prefer not to say"),
    econ_level = as_factor(econ_level)
  )

df_filtered <- df_filtered |>
  mutate(
    econ_level = case_when(
      G19 == 1 ~ "With great difficulty",
      G19 == 2 ~ "With difficulty",
      G19 == 3 ~ "With some difficulty",
      G19 == 4 ~ "Fairly easily",
      G19 == 5 ~ "Easily",
      G19 == 6 ~ "Very easily",
      TRUE ~ NA_character_
    )
  ) |>
  mutate(
    econ_level = factor(econ_level,
                        levels = c("With great difficulty",
                                   "With difficulty",
                                   "With some difficulty",
                                   "Fairly easily",
                                   "Easily",
                                   "Very easily"))
  )

# 5.5) Marital Status
df_filtered <- df_filtered |>
  mutate(
    marital_status = case_when(
      G4 == 1 ~ "Single",
      G4 == 2 ~ "Same_sex_married",
      G4 == 3 ~ "Different_sex_married",
      G4 == 4 ~ "Divorced",
      G4 == 5 ~ "Separated",
      G4 == 6 ~ "Widowed",
      TRUE ~ NA_character_
    )
  ) |>
  mutate(
    marital_status = factor(marital_status,
                            levels = c("Single",
                                       "Same_sex_married",
                                       "Different_sex_married",
                                       "Divorced",
                                       "Separated",
                                       "Widowed"))
  )

# 5.6) Life Satisfaction
df_filtered <- df_filtered |>
  mutate(
    life_satisfaction = case_when(
      H1 == -99 ~ NA_real_,
      H1 >= 0 & H1 <= 10 ~ H1,
      TRUE ~ NA_real_
    )
  )

# 5.7) Health Condition
df_filtered <- df_filtered |>
  mutate(
    health = case_when(
      G16 %in% c(-99, -96) ~ NA_character_,
      G16 == 1 ~ "Very good",
      G16 == 2 ~ "Good",
      G16 == 3 ~ "Fair",
      G16 == 4 ~ "Bad",
      G16 == 5 ~ "Very bad",
      TRUE ~ NA_character_
    )
  ) |>
  mutate(
    health = factor(health,
                    levels = c("Very good", "Good", "Fair", "Bad", "Very bad"),
                    ordered = TRUE)
  )

# 5.8) Religion
df_filtered <- df_filtered |>
  mutate(
    religion = case_when(
      G15 == -96 ~ NA_character_, # Prefer not to say
      G15 == 1 ~ "No religion",
      G15 %in% c(2,3,4,5) ~ "Christian",
      G15 == 6 ~ "Islam",
      G15 == 7 ~ "Hindu",
      G15 == 8 ~ "Jewish",
      G15 == 9 ~ "Buddhist",
      G15 == 10 ~ "Sikh",
      G15 == 11 ~ "Other religion",
      TRUE ~ NA_character_
    )
  ) |>
  mutate(
    religion = factor(religion)
  )

# 5.9) Residence Type
df_filtered <- df_filtered |>
  mutate(
    residence = case_when(
      G3 == 1 ~ "Big city",
      G3 == 2 ~ "Suburbs",
      G3 == 3 ~ "Town/Small city",
      G3 == 4 ~ "Village",
      G3 == 5 ~ "Countryside",
      G3 %in% c(-99, -96) ~ NA_character_,
      TRUE ~ NA_character_
    )
  ) |>
  mutate(
    residence = factor(residence,
                       levels = c("Big city", "Suburbs", "Town/Small city",
                                  "Village", "Countryside"),
                       ordered = TRUE)
  )

# 5.10) Country
df_filtered <- df_filtered |>
  mutate(country = case_when(
    A9_REC == 1 ~ "Austria",
    A9_REC == 2 ~ "Belgium",
    A9_REC == 3 ~ "Bulgaria",
    A9_REC == 4 ~ "Croatia",
    A9_REC == 5 ~ "Cyprus",
    A9_REC == 6 ~ "Czech Republic",
    A9_REC == 7 ~ "Denmark",
    A9_REC == 8 ~ "Estonia",
    A9_REC == 9 ~ "Finland",
    A9_REC == 10 ~ "France",
    A9_REC == 11 ~ "Germany",
    A9_REC == 12 ~ "Greece",
    A9_REC == 13 ~ "Hungary",
    A9_REC == 14 ~ "Ireland",
    A9_REC == 15 ~ "Italy",
    A9_REC == 16 ~ "Latvia",
    A9_REC == 17 ~ "Lithuania",
    A9_REC == 18 ~ "Luxembourg",
    A9_REC == 19 ~ "Malta",
    A9_REC == 20 ~ "Netherlands",
    A9_REC == 21 ~ "Poland",
    A9_REC == 22 ~ "Portugal",
    A9_REC == 23 ~ "Romania",
    A9_REC == 24 ~ "Slovakia",
    A9_REC == 25 ~ "Slovenia",
    A9_REC == 26 ~ "Spain",
    A9_REC == 27 ~ "Sweden",
    A9_REC == 29 ~ "North Macedonia",
    A9_REC == 30 ~ "Serbia",
    A9_REC == 32 ~ "Albania",
    TRUE ~ NA_character_
  ))

# Convert to factor with appropriate levels
df_filtered$country <- factor(df_filtered$country,
                              levels = c("Austria", "Belgium", "Bulgaria", "Croatia",
                                         "Cyprus", "Czech Republic", "Denmark", "Estonia",
                                         "Finland", "France", "Germany", "Greece",
                                         "Hungary", "Ireland", "Italy", "Latvia",
                                         "Lithuania", "Luxembourg", "Malta", "Netherlands",
                                         "Poland", "Portugal", "Romania", "Slovakia",
                                         "Slovenia", "Spain", "Sweden", "North Macedonia",
                                         "Serbia", "Albania"),
                              exclude = NULL)

#------------------------------------------------------------
# 6) Process openness-related variables
#------------------------------------------------------------
# 6.1) Coming Out Age
df_filtered <- df_filtered |>
  mutate(
    coming_out_age = case_when(
      B2 == -99 ~ NA_real_, # Don't know -> NA
      B2 == 0 ~ 0, # Not out
      B2 == 1 ~ 5, # 5 or younger
      B2 == 2 ~ 7.5, # 6-9 years old (midpoint)
      B2 == 3 ~ 12, # 10-14 years old (midpoint)
      B2 == 4 ~ 16, # 15-17 years old (midpoint)
      B2 == 5 ~ 21, # 18-24 years old (midpoint)
      B2 == 6 ~ 29.5, # 25-34 years old (midpoint)
      B2 == 7 ~ 44.5, # 35-54 years old (midpoint)
      B2 == 8 ~ 55, # 55 or older
      TRUE ~ NA_real_
    )
  )

# 6.2) Public Display of Affection and Location Comfort
df_filtered <- df_filtered |>
  mutate(
    # B3 reverse coded - higher score means more open
    pda_comfort = case_when(
      B3 == 1 ~ 4, # Never avoids -> 4
      B3 == 2 ~ 3, # Rarely avoids -> 3
      B3 == 3 ~ 2, # Often avoids -> 2
      B3 == 4 ~ 1, # Always avoids -> 1
      B3 == 9 ~ NA_real_, # No same-sex partner
      B3 < 0 ~ NA_real_, # Don't know
      TRUE ~ NA_real_
    ),
    # B4 reverse coded - higher score means more open
    location_comfort = case_when(
      B4 == 1 ~ 4, # Never avoids -> 4
      B4 == 2 ~ 3, # Rarely avoids -> 3
      B4 == 3 ~ 2, # Often avoids -> 2
      B4 == 4 ~ 1, # Always avoids -> 1
      B4 < 0 ~ NA_real_, # Don't know
      TRUE ~ NA_real_
    )
  )

# 6.3) Disclosure locations
# Create binary variables for disclosure settings (1=does not avoid, 0=avoids)
b5_vars <- paste0("B5_", LETTERS[1:11]) # A to K, excluding L(Don't know)
df_filtered <- df_filtered |>
  mutate(across(all_of(b5_vars), ~ifelse(. == 2, 1, 0))) # 2(Not Selected) means does not avoid

# Calculate total non-avoidance score (higher = more open)
df_filtered$disclosure_locations <- rowSums(df_filtered[, b5_vars], na.rm = TRUE)

# 6.4) Outness to different groups
b6_vars <- paste0("B6_", LETTERS[1:8])
df_filtered <- df_filtered |>
  mutate(across(all_of(b6_vars),
                ~case_when(
                  . == 4 ~ 3, # All -> 3
                  . == 3 ~ 2, # Most -> 2
                  . == 2 ~ 1, # A few -> 1
                  . == 1 ~ 0, # None -> 0
                  . == -98 ~ NA_real_, # Does not apply
                  TRUE ~ NA_real_
                )))

# Calculate mean outness score
df_filtered$outness_mean <- rowMeans(df_filtered[, b6_vars], na.rm = TRUE)

#------------------------------------------------------------
# 7) Create interaction variables
#------------------------------------------------------------
# 7.1) Gender & Economic Level Interaction
df_filtered <- df_filtered |>
  mutate(genderXecon = paste(gender, econ_level, sep = "_"))
df_filtered$genderXecon <- as.factor(df_filtered$genderXecon)

# 7.2) Gender & Minority Status Interaction
df_filtered <- df_filtered |>
  mutate(genderXmin = paste(gender, minority, sep = "_"))
df_filtered$genderXmin <- as.factor(df_filtered$genderXmin)

#------------------------------------------------------------
# 8) Process discrimination variables - Feature engineering
#------------------------------------------------------------
# 8.1) Rename discrimination variables for clarity
df_filtered <- df_filtered |>
  rename(
    disc_job = C1_A,
    disc_workplace = C1_B,
    disc_housing = C1_C,
    disc_healthcare = C1_D,
    disc_education = C1_E,
    disc_restaurant = C1_F,
    disc_shopping = C1_G,
    disc_public_service = C1_H,
    disc_id_doc = C1_I
  )

# 8.2) Recode as binary (1=experienced discrimination, 0=did not)
discrimination_vars <- c("disc_job", "disc_workplace", "disc_housing",
                         "disc_healthcare", "disc_education", "disc_restaurant",
                         "disc_shopping", "disc_public_service", "disc_id_doc")
for(var in discrimination_vars) {
  df_filtered[[var]] <- ifelse(df_filtered[[var]] == 1, 1, 0)
}

# 8.3) Create total discrimination score
df_filtered$disc_total <- rowSums(df_filtered[, discrimination_vars], na.rm = TRUE)

#------------------------------------------------------------
# 9) Process violence variables - Feature engineering
#------------------------------------------------------------
# Create 3 violence severity groups (never, mild, severe) based on E1
df_filtered <- df_filtered |>
  mutate(
    violence_group = case_when(
      E1 == 0 ~ 0,                           # Never
      E1 >= 1 & E1 <= 2 ~ 1,                 # Mild (1-2)                     
      E1 >= 3 & E1 <= 6 ~ 2,                 # Severe (3+)
      TRUE ~ NA_real_                        # NA
    )
  )

# Convert to factor with descriptive labels
df_filtered$violence_group <- factor(df_filtered$violence_group, 
                                     levels = c(0, 1, 2),
                                     labels = c("Never", "Mild", "Severe"))

# Check distribution
print("Distribution of violence severity groups:")
print(table(df_filtered$violence_group, useNA = "always"))

#------------------------------------------------------------
# 10) Factor Analysis for Discrimination
#------------------------------------------------------------
# 1. Prepare the data matrix - Select only discrimination variables
discrimination_vars <- c("disc_job", "disc_workplace", "disc_housing",
                         "disc_healthcare", "disc_education", "disc_restaurant",
                         "disc_shopping", "disc_public_service", "disc_id_doc")
disc_data <- df_filtered |>
  select(all_of(discrimination_vars))

# 2. Compute the correlation matrix
disc_cor <- cor(disc_data)
corrplot(disc_cor, method = "color", type = "upper",
         order = "hclust", tl.col = "black", tl.srt = 45,
         title = "Correlation Between Discrimination Types")

# 3. KMO measure to confirm suitability for factor analysis
kmo_result <- KMO(disc_cor)
print("KMO test result:")
print(kmo_result)

# 4. Determine the appropriate number of factors
# Parallel analysis
parallel_result <- fa.parallel(disc_data, fa = "fa", fm = "ml",
                               main = "Parallel Analysis Scree Plot")

# Based on factor analysis results, create total discrimination score
df_filtered$disc_total <- rowSums(df_filtered[, discrimination_vars], na.rm = TRUE)
print("Distribution of total discrimination scores:")
print(table(df_filtered$disc_total, useNA = "always"))