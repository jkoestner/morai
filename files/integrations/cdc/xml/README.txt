# General

API documentation: https://wonder.cdc.gov/wonder/help/wonder-api.html
Helpful support API info: https://github.com/alipphardt/cdc-wonder-api

Group results by:
- Injury Intent (D76.V22) for "B_1";
- and by Injury Mechanism (D76.V23) for "B_2";

where data are limited to:
- single-year age groups (O_age value is D76.V52) for ages < 18;
- cause of death is set to Injury Intent (O_ucd values is D76.V22), and D76.V22 is limited to the 5 injury categories only;

and show:
- number of deaths (D76.M1);
- population estimates (D76.M2);
- crude death rates (D76.M3).

## Common changes to template

- update O_show_totals from `True` to `False`
- Remove the M2 (population estimates), M3 (crude death rate), and M34 (Crude 95%) columns

## List of Queries

Each query gets to a granularity of death. If querying too granular the deaths will be surpressed. Weekly, cause of death, and ages are granular fields.
Weekly data has to be queried separately from monthly because of the timing differences.

- mcd18_cod:               year, age_group, cod_sub_chapter
- mcd18_mi:                year, age_group, gender
- mcd18_monthly:           year, month
- mcd18_weekly:            year, week
- mcd18_weekly_influenza:  year, week - filtered by influenza cod_sub_chapter

## Limitations

The API will not be able "limit or group results by any location field, such as Region, Division, State or County, or Urbanization". In this case the queries will need to be done from the wonder database itself and saved as text.

### Saved Queries

- 1999-2020
  - All: https://wonder.cdc.gov/controller/saved/D76/D351F671
  - Q1:  https://wonder.cdc.gov/controller/saved/D76/D400F814
  - Q2:  https://wonder.cdc.gov/controller/saved/D76/D400F815
  - Q3:  https://wonder.cdc.gov/controller/saved/D76/D400F821
  - Q4:  https://wonder.cdc.gov/controller/saved/D76/D400F820
  - Q5:  https://wonder.cdc.gov/controller/saved/D76/D400F822
- 1979-1998
  - Q1:  https://wonder.cdc.gov/controller/saved/D16/D351F716
  - Q2:  https://wonder.cdc.gov/controller/saved/D16/D351F717
  - Q3:  https://wonder.cdc.gov/controller/saved/D16/D351F718
  - Q4:  https://wonder.cdc.gov/controller/saved/D16/D351F720
  - Q5:  https://wonder.cdc.gov/controller/saved/D16/D351F721