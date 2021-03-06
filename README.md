# Data Assimilation Project

# Usage

## Initialization
1. install Python
2. install Rust
3. install Julia
4. `$julia julia-init.jl`
5. `$python3 python/nature.py`
6. `$python3 python/init.py`
7. `$bash xtinit.sh`
8. `$cargo run --release`
9. choose one of `x_a_init_<tur>.txt` as `x_a_init.txt`

## NMC production
1. `$bash y_o_full.sh`
2. `$bash da.sh oi-nmc`

## Experiments
1. use `$bash sh/y_o_<exp>.sh` to set observation
2. run assimilation with `$bash sh/runda.sh`
3. plot result with `python3 python/plot.py`

# Abstract

## Model
Lorenz 96 model

## DA schemes
- OI
- 3DVar
- 4DVar

## Experiment

### Basic Setup
- 20 observation, average distributed

### Synchronous Observation
- all observation sync between constant time interval

### Asynchronous Observation
- different observation station with different time
- compare with sync observation

### Abnormal Observation
- randomly missing for a long time
- randomly producing fake data for a long time

### Add-one Observation
- contains missing 20 observation of basic setup, but only one data
