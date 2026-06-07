# Executive Summary

There are several public sources for general population mortality data and exposure data. Each of the sources has valuable information, however the user should understand the limitations for their intended purpose.

The purpose of this note is to consolidate information on working with each of these sources. It should be used as a resource for members of ILEC.

---

## Center for Disease Control (CDC)

The Center for Disease control is the national public health agency of the United States. It is part of the U.S. Department of Health and Human Services (HHS) and is government-funded.

The agency provides multiple datasets for mortality, including underlying cause and multiple cause of death, through the WONDER query tool:

- **2018 to Current:** Multiple cause of death (provisional)
  - https://wonder.cdc.gov/mcd-icd10-provisional.html
- **1999 to 2020:** Underlying cause of death
  - https://wonder.cdc.gov/ucd-icd10.html
- **1979 to 1998:** Compressed mortality
  - https://wonder.cdc.gov/cmf-icd9.html
- **1968 to 1978:** Compressed mortality
  - https://wonder.cdc.gov/cmf-icd8.html

### Sources

- Mortality Data is provided by NCHS
- Based on 50 states and District of Columbia. Nonresidents (e.g. nonresident aliens, nationals living abroad, residents of Puerto Rico, Guam, the Virgin Islands, and other territories of the U.S.) and fetal deaths are excluded.
- The population estimates are U.S. Census Bureau estimates of U.S. national, state, and county resident populations.
- CDC Wonder simply carries the latest available exposure data forward in time. So in 2025, they are using data for 2024 (or maybe even for 2023). The 65+ population is, in fact, growing across time, so the CDC overestimates mortality rates.

### Caveats

- There is a lag in when deaths are considered final and it will take a few weeks for deaths to be recorded. It can take 6 months for deaths to be recorded with the correct underlying cause.
- There is a lag in population data and can take 2 years for it to be populated.

### Tool Limitations

- Manual queries are limited to 75,000 rows and results are suppressed when there are 30 deaths or less to protect privacy. It also doesn't show exposure information for ages 86+. This means that users cannot query for mortality rates across each county individually (or other similar granular splits). It also means that other sources of exposures need to be used.

---

## Human Mortality Database (HMD)

The Human Mortality Database is a collaborative project between University of California, Berkeley and Max Planck Institute for Demographic Research (Germany) to collect mortality data for different countries. There are currently over 40 countries' data available.

- **1933 – 2023:** Deaths and Population
  - https://www.mortality.org/Country/Country?cntr=USA

### Sources

- Mortality Data is provided by NCHS
- For the years 1933–1969, deaths in the HMD cover both residents and nonresidents (i.e., the de facto population), and for the period starting in 1970, they only cover residents.
- Based on 50 states and District of Columbia. Nonresidents (e.g. nonresident aliens, nationals living abroad, residents of Puerto Rico, Guam, the Virgin Islands, and other territories of the U.S.) and fetal deaths are excluded.
- The population estimates are U.S. Census Bureau estimates of U.S. national, state, and county resident populations.

---

## Census Bureau (Census)

The census bureau is a government agency that provides population estimates in July of each year. It should be noted that data is frequently revised for prior years as new census is surveyed and prior estimates are adjusted.

- **Population Estimates Dataset:**
  - https://www2.census.gov/programs-surveys/popest/datasets/

---

## Congressional Budget Office (CBO)

The Congressional Budget Office has projections for demographics every year as well as a report. The demographics include population, immigrants, emigrants, mortality, and fertility.

- **2025 to 2055:** The Demographic Outlook
  - https://www.cbo.gov/about/products/major-recurring-reports#1 (Demographic Outlook section)
- **2025 to 2055:** Data
  - https://www.cbo.gov/data/budget-economic-data (Demographic Projections section)

---

## Social Security Administration (SSA)

The Social Security Administration has projections for the social security program.

- **2025:** Annual Report
  - https://www.ssa.gov/oact/tr/2025/
- **2025:** Population
  - https://www.ssa.gov/oact/tr/2025/V_A_demo.html#271410
- **2025:** Downloadables
  - https://www.ssa.gov/OACT/Downloadables/CY/index.html

### Caveats

- The SSA data captures more than the other data sources. In addition to the 50 states and DC, the SSA data captures the following population segments: civilian residents of Puerto Rico, the Virgin Islands, Guam, American Samoa, and the Northern Mariana Islands; Federal civilian employees and persons in the U.S. Armed Forces abroad and their dependents; non-citizens living abroad who are insured for Social Security benefits; and all other U.S. citizens abroad. For instance, the Census had 335M population in 2023 while the SSA had 342M population.
