# US-AI-Grid-Analysis
This study explore hourly grid impact of incorporating generative AI campus into 2035 U.S. grid systems. Its contents are listed as below:
1. Codes: include major codes to conduct the estimation process.
2. Data: include all data used during the analysis.

## Requirements
To run the codes in this repository, the following Python and core packages must be installed (version is given for refenrence):
- Python 3.9.13
- numpy 1.21.5
- csv 1.0
- Pandas 2.2.3
- SciPy 1.13.1

The above packages can be conveniently downloaded through open-source library within an hour on a normal computer. There is no specific computing resource requirements to run the provided codes, which can be runned on normal computer within a few seconds.

## Codes
- **Total Demand.py**: This file contains an example case on generating the grid total demand profiles by incoporating AI campus dynamics.
- **Net Demand.py**: This file contains an example case on generating the grid ned demand profiles by incoporating AI campus dynamics.
- **Battery Storage.py**: This file contains an example case on generating the battery storage requriements to balancing ther ramping power increase driven by AI campus dynamics.

## Data
- **2035_CF_COL_NG.xlsx**: the 2035 carbon factors of coal and natrual gas across U.S. regions.
- **2035_net_demand_MW.xlsx**: the 2035 pre-AI net demand profiles of U.S. regions.
- **2035_renewables_MW.xlsx**: the 2035 pre-AI grid-level renewable generation profiles of U.S. regions.
- **2035_total_demand_MW.xlsx**: the 2035 pre-AI total demand profiles of U.S. regions.
- **Regional AI Capacity.xlsx**: the projected AI capacity installation of each U.S. region under differnt scenarios.
- **Regional_unitProfiles_wind_solar.xlsx**: the regional wind and solar generation profiles
- **all_40_weekday_weekend.xlsx**: the generated 40 AI computing profiles.
- **representative_weekday_weekend.xlsx**: the 9 xtracted representative AI computing profiles. 

## Running the code
The code files can be used by simply replace the "BASE PATH" used in the code file with the install path of our data folder to run the simulation.

## Citation
Please use the following citation when using the data, methods or results of this work:

Xiao, T., You, F., Shaping Power Systems with AI Data Centers and On-Site Renewables: Grid-Scale Impacts Across U.S. Regions. Submitted to Nature Energy.

## License
This project is covered under the **Apache 2.0 License**.
