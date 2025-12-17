# AE 370: Numerical Methods. Group Project 2.

## Authors
Nikita Kovalov, Daniel Song, Oliver Walsh, Brian Wu

## Topic
Euler-Bernoulli beam IBVP simulation and analyis.

## Objective
Accurately model bending and vibrational characteristics of an aircraft wing modeled as an Euler-Bernoulli beam using backward Euler method applied on the Euler-Bernoulli IBVP.

## Purpose
Identification and analysis of resonance and transient responses to loads and control inputs.

## Usage
```
git clone https://github.com/dssong2/ae370-project2.git && cd ae370-project2
```
```
pip install -r requirements.txt
```
Navigate to src folder to find code contents. 

beam.py contains the class Beam that defines the IBVP and the backward Euler implementation.

To produce error convergence and simulation results, navigate to notebooks folder and run each respective file. Will take a while to run because of IBVP complexity.