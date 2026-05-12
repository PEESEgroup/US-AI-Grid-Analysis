# US-AI-Grid-Analysis
This study explores hourly grid impact of incorporating generative AI campus into 2035 U.S. grid systems. Its contents are listed as below:
1. Codes: include major codes to conduct the estimation process.
2. Data: include all data used during the analysis.

## Requirements
To run the codes in this repository, the following Python version and core packages are recommended:
- Python 3.9.13
- numpy 1.21.5
- pandas 2.2.3
- scipy 1.31.1
- h5py 3.12.1
- openpyxl 3.1.5
- ReEDs 2.0 (specific settings,installations and computing resource requirements can be found in https://github.com/NREL/ReEDS-2.0)
The scripts can be run on a normal desktop or laptop computer. Runtime depends on the size of the hourly ReEDS input files and the number of generated scenarios, but the example workflows do not require specialized computing resources.

## Codes
The `Codes` folder contains four example scripts:
- **generate_pure_ai_load.py**: Generates ReEDS hourly load files by adding AI computing electricity demand to the baseline grid load, without behind-the-meter generation or load shaving.
- **generate_ai_campus_load.py**: Generates ReEDS hourly load files for AI campus cases with behind-the-meter wind, solar, and nuclear generation. Excess on-site renewable generation is curtailed rather than exported to the bulk grid.
- **generate_pure_ai_shaving_load.py**: Generates ReEDS hourly load files for pure-AI demand cases with training-load shaving. The script adjusts flexible training demand to reduce net-demand ramping impacts.
- **generate_ai_campus_shaving_load.py**: Generates ReEDS hourly load files for AI campus cases that combine behind-the-meter generation and training-load shaving.

## Data
- **AI_capacity_projection.xlsx**: projected regional AI computing capacity under different AI growth scenarios.
- **AI_Spatial.txt**: spatial allocation weights used to distribute AI capacity to ReEDS balancing areas.
- **BA_area.csv**: mapping between balancing areas, states, regional identifiers, and area-related information used to allocate AI demand across ReEDS load regions.
- **EER_IRAlow_load_hourly.h5**: baseline ReEDS hourly load file used as the template for generating AI-modified load files.
- **RCP4.5_2035_Base_PUE.csv**: regional power usage effectiveness profiles for air-cooled AI data center electricity-demand calculation.
- **RCP4.5_2035_Base_PUE_i.csv**: regional power usage effectiveness profiles for immersion cooling AI data center electricity-demand calculation.
- **RCP4.5_2035_Base_WUE.csv**: baseline regional water usage effectiveness profiles. This file is included for related environmental-impact analysis but is not directly required by the four load-generation scripts.
- **RCP4.5_2035_Base_WUE_i.csv**: improved regional water usage effectiveness profiles. This file is included for related environmental-impact analysis but is not directly required by the four load-generation scripts.
- **reeds_7_weather_year_utilization_profiles.xlsx**: representative AI computing utilization profiles used to construct hourly AI electricity demand.
- **reeds_7_weather_year_training_utilization_profiles.xlsx**: training-specific utilization profiles used by the load-shaving scripts.
- **cf_upv_reference_ba_adjusted.h5**: balancing-area-level solar photovoltaic capacity-factor profiles used to estimate behind-the-meter solar generation for AI campus scenarios.
- **cf_wind-ons_open_ba_adjusted.h5**: balancing-area-level onshore wind capacity-factor profiles used to estimate behind-the-meter wind generation for AI campus scenarios.
- **_Baseline_renewable_generation_EIA.xlsx**: baseline regional grid-level renewable-generation profiles used by the shaving scripts to evaluate net-demand ramping conditions.
- **Cases_AI_Grid.csv**: the case definition file needed for conducting ReEDS simulations

## Running the code
The code files can be used by simply replace the "default_data_dir" used in the code file with the install path of our data folder to run the simulation.
After generating the edited demand files with the code, users can refer to the ReEDS 2.0 documentation at https://github.com/NREL/ReEDS-2.0 to learn how to further use Cases_AI_Grid.csv and the demand files for grid expansion simulations.

## Citation
Please use the following citation when using the data, methods or results of this work:

Xiao, T., You, F., Shaping Power Systems with AI Data Centers and On-Site Renewables: Grid-Scale Impacts Across U.S. Regions. Submitted to Nature Communications.

## License
This project is covered under the **Apache 2.0 License**.
