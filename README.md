# Flow rate Forecasting

Challenge 6 at BlueArk Challenge 2019

Luana M, Emilie N, Benjamin G, Camilo P, Julien R, Jimmy V

 

## Dataset

Data is provided for several weather stations and contains flow rates (m^3/s), rainfall (mm), temperature (°C) and sunshine (W/mm^2). Measures are taken each 15 minutes and the winter months (November to April) is not accounted for in flow rates, because it's consistent.

| station    | Température | Soleil   | Pluie    | groupe |
| ---------- | ----------- | -------- | -------- | ------ |
| Tsijore    | Arolla      | -        | -        | z      |
| Bertol Inf | Arolla      | -        | -        | y      |
| Ferpecle   | Arolla      | Bricola  | Bricola  | A      |
| Edelweiss  | Zmutt       | Findelen | Findelen | B      |
| Gornera    | Zmutt       | Findelen | Findelen | B      |
| Stafel     | Zmutt       | Findelen | Findelen | B      |
| Arolla     | x           | -        | x        |        |
| Bricola    | -           | x        | x        |        |
| Findelen   | -           | x        | x        |        |
| Zmutt      | x           | -        | -        |        |



### Cleaning

We received 14 excel files containing flow rates (debit), rainfall (pluie), temperatures (temperature) and sunshine (rayonnement) for the last four years. We converted them to csv (unicode utf8) and removed the useless second title line. 

### Initial Plotting

