# BatteryForecastAI: An Open-Source Tool for prognosis of battery advanced state of health

## Introduction

De-risking energy storage investments necessary to meet CO2 reduction targets requires a deep understanding of the connections between battery health, design, and use. The historical definition of the battery state of health (SOH) as the percentage of current versus initial capacity is inadequate for this purpose, motivating an expanded SOH consisting of an interrelated set of descriptors including capacity, energy, ionic and electronic impedances, opencircuit voltages, and microstructure metrics. In this work, we introduce deep transformer networks for the simultaneous prognosis of 28 battery SOH descriptors using two cycling datasets representing six lithium-ion cathode chemistries, multiple electrolyte/anode compositions, and different charge-discharge scenarios. The accuracy of these predictions for battery life (with an unprecedented mean absolute error of 19 cycles in predicting end of life for a lithium-iron-phosphate fast-charging dataset) illustrates the promise of deep learning toward providing enhanced understanding and control of battery health.


## FrameWork

WIP

<img src="/BatteryForecastAI/batteryforecast/image/FrameWork.png" width="800">

## Battery ML features

- **ECM based features:** WIP
- **Physics based features:** WIP
- **Statistical features:** WIP

## Dataset

| Data Source | Electrode Chemistry | Nominal Capacity | Voltage Range (V) | RUL dist. | SOC dist. (%) | SOH dist. (%) | Cell Count |  
|---|---|---|---|---|---|---|---|  
| CAMP | NA | NA | NA | NA | NA | NA | NA |  
| SEVERSON | LFP/graphite | 1.1 | 2.0-3.6 | 823±368 | 93±7 | 36±36 | 180 |

## Benchmark result of ML predictions

## Benchmark result of ASOH Forecasting


## Quick Start

### Install

WIP

### Download Raw Data and Run Preprocessing Scripts
<!-- Download the raw data and execute the preprocessing scripts as per the provided [instruction](./dataprepare.md). You can also use the code below to download public datasets and convert them to BatteryML's uniform data format. -->
Download raw files of public datasets and preprocess them into `BatteryData` of BatteryML is now as simple as two commands:

```bash
python batteryforecast/data/download.py CAMP /path/to/save/raw/data
batteryforecast preprocess CAMP /path/to/save/raw/data /path/to/save/processed/data
```
