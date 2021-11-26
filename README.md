# SEIRV-COVID-19-Dashboard
### SEIRV Model from Alhamami, H.(2019)

![first eq](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7D%20S%27%28t%29%20%3D%20%5Clambda%20-%20%5Cfrac%7B%5Cbeta%20S%28t%29%20I%28t%29%7D%7BN%28t%29%7D%20-%20%5Calpha%20S%28t%29%20-%20%5Cmu%20S%28t%29%20%5Cend%7Bequation%7D)<br />
![second eq](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7D%20E%27%28t%29%20%3D%20%5Cfrac%7B%5Cbeta%20S%28t%29%20I%28t%29%7D%7BN%28t%29%7D%20&plus;%20%5Cfrac%7B%281-p%29%20%5Cbeta%20V%28t%29I%28t%29%7D%7BN%28t%29%7D%20-%20%5Csigma%20E%28t%29%20-%5Cmu%20E%28t%29%20%5Cend%7Bequation%7D)<br />
![third eq](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7D%20I%27%28t%29%20%3D%20%5Csigma%20E%28t%29%20-%20%5Ceta%20I%28t%29%20-%20%5Cdelta%20I%28t%29%20-%20%5Cmu%20I%28t%29%20%5Cend%7Bequation%7D)<br />
![forth eq](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7D%20R%27%28t%29%20%3D%20p%20V%28t%29%20&plus;%20%5Ceta%20I%28t%29%20-%20%5Cmu%20R%28t%29%20%5Cend%7Bequation%7D)<br />
![fifth eq](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7DV%27%28t%29%20%3D%20%5Calpha%20S%28t%29%20-%20%5Cfrac%7B%281-p%29%20%5Cbeta%20V%28t%29I%28t%29%7D%7BN%28t%29%7D%20-%20p%20V%28t%29%20-%20%5Cmu%20V%28t%29%20%5Cend%7Bequation%7D)<br />

| Parameters | Description |  
|-----------|:-----------:| 
| ![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7D%5Clambda%20%5Cend%7Bequation%7D) | Recruitment rate of susceptible |  
| ![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7D%5Cmu%20%5Cend%7Bequation%7D) | Natural mortality rate |  
| ![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7D%5Cdelta%20%5Cend%7Bequation%7D) | Mortality rate due to COVID-19 | 
| ![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7D%5Cbeta%20%5Cend%7Bequation%7D) | COVID-19 infection rate | 
| ![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7D%5Csigma%20%5Cend%7Bequation%7D) | Progression rate from exposed to infected | 
| ![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7D%5Ceta%20%5Cend%7Bequation%7D) | Recovery rate from COVID-19 | 
| ![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7Dp%20%5Cend%7Bequation%7D) | Vaccination success rate | 
| ![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Bequation%7D%20%5Ccolor%7BBlack%7D%5Calpha%20%5Cend%7Bequation%7D) | Vaccination rate | 
