# Class Motif - Documentation

The Motif class is a convenience class to compute and plot a position-weight matrix (PWM). The only functionality is the plot function. The PWM and corresponding entropy values can be accessed using the self.pwm and self.entropies members, if so desired. Motifs using the following characters can be plotted : "ACGTU().HIMS".

## Methods - Overview

| name | description |
|:-|:-|
| \_\_init\_\_ | Initiliaze a motif by providing sequences or a PWM. |
| plot | Plot the motif. |
## \_\_init\_\_

``` python
def __init__(self, alphabet, sequences = None, pwm = None)
```
Initiliaze a motif by providing sequences or a PWM. 

 Either a list of sequences or a PWM with shape (sequence length, alphabet length) must be provided. 



| parameter | type | description |
|:-|:-|:-|
| alphabet | str | The alphabet of the sequences. |
| sequences | [str] | A list of strings. All strings must have the same length. |
| pwm | numpy.ndarray | A matrix of shape (sequence length, alphabet length) containing probabilities. |
## plot

``` python
def plot(self, scale = 1)
```
Plot the motif. 

 The default height of the plot is 754 pixel. The width depends on the length of the motif. Using, for instance, a scale parameter of 0.5 halves both height and width. 



| parameter | type | description |
|:-|:-|:-|
| scale | float | Adjust the size of the plot (should be > 0). |

| returns | type | description |
|:-|:-|:-|
| image | PIL.image.image | A Pillow image object. |
