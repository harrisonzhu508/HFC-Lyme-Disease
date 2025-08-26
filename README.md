# SPHERE-PPL Health Forecasting Contest - UK Lyme Disease

## Introduction[^readme-1]

[^readme-1]: Adapted from UKHSA Website (<https://www.gov.uk/government/publications/lyme-borreliosis-epidemiology/lyme-borreliosis-epidemiology-and-surveillance>)

Lyme disease is the most common vector-borne human infection in England and Wales. As elsewhere in northern Europe, the spirochaetes (Borrelia burgdorferi) are transmitted by the hard bodied tick, Ixodes ricinus, commonly known as deer or sheep ticks.

Habitats suitable for acquiring infection occur in temperate regions of the northern hemisphere, usually in forested woodland or heathland areas which support the life-cycles of ticks and the small mammals and birds that can be reservoir hosts for B. burgdorferi.

Several pathogenic genospecies of B. burgdorferi have been identified in Europe and there is evidence for some variation in the types of clinical presentation caused by these different genospecies.

This contest is in collaboration with the [VBD Hub](https://vbdhub.org/).

## Forecasting Outputs

For this contest, we are looking to forecast Lyme Disease Cases at a range of geographical scales for 2023 & 2024. Incidence rates should be given as cases per 100,000 population.

Forecasts and reports should be saved into the submission folder, matching the template found within.

There are 3 columns to be filled for the forecast:

1.  Incidence (Mean)

2.  Incidence (Lower 95th Confidence Interval)

3.  Incidence (Upper 95th Confidence Interval)

***Note: Data is missing for "Bournemouth", "City of London", and "Isles of Scilly". These councils are therefore optional.***

## Joining the contest & Getting Started

In order to join the contest, you will need to fork or download the repo.

To fork the repo, simply press the "fork" button, which can be found at the top of this github page. A step-by-step guide can be found [here](https://scribehow.com/shared/Forking_a_SPHERE-PPL_Forecasting_Contest_Repository_on_GitHub__o_bLCyQlTsO0o5YCmGsk8Q).

To download the data without a github account, click the code box dropdown and download a zip of the data directly to your computer.

![Fork or Download](https://github.com/SPHERE-PPL/forecasting-contest-template/blob/main/contest_media/fork_button.png)

Here are some covariates that might be an interesting starting point:

-   Tick distributions in the UK & other VBD related data: <https://github.com/fwimp/ohvbd>

-   Geographical data for the UK: <https://github.com/rspatial/geodata>

-   National Park GIS data: <https://naturalengland-defra.opendata.arcgis.com/datasets/national-parks-england/explore?location=53.302276%2C-1.251342%2C7.67>

## Rules

-   Any coding languages are allowed but all analyses must be reproducible by the panel.
-   All entries must be loaded into a public Github repo.
-   All entries must follow the submission formats outlined below.
-   All entries must include a max 1000 word report to accompany the forecast analyses. This can be as a separate PDF/hmtl or incorporated into a quarto/jupyter notebook.
-   All submissions must be complete by Friday 27th June 2025.

## How to Win!

Awards will be given across two categories:

1. The team with the closest forecast, as measured by Root Mean Squared Error.  

2. The team with the most interesting report.

The winners will be selected by the SPHERE-PPL Team and will be invited to present their forecasts at the next Annual Meeting, with travel covered by the project.

## How to Submit

If you forked the repo, congratulations, you have already entered the contest. We will run the [Forecast AggregatoR](https://github.com/SPHERE-PPL/Forecast-AggregatoR) the day following the close of the contest and your repo will be collated with the entries.

If you did not fork the repo, please send an email to [contest\@sphere-ppl.org](mailto:contest@sphere-ppl.org) with a link to your public github repo where your forecast and report are stored. These will then be collated with the other entries.

## Connect with the Community

You can join our Zulip [here](https://sphereppl.zulipchat.com/join/olwtpi7g3wbyh5mxv4uwipaw/) and check out our events page to see the next online catch-up.

## License

![CC-BYNCSA-4](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)

Unless otherwise noted, the content in this repository is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

For the data sets in the *data/* folder, please see [*data/README.md*](data/README.md) for the applicable copyrights and licenses.
