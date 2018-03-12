# Class Motif - Documentation

The Motif class is a convenience class to compute and plot a position-weight matrix (PWM). The only functionality is the plot function. The PWM and corresponding entropy values can be accessed using the self.pwm and self.entropies members, if so desired. All uppercase alphanumeric characters and the following additional characters can be part of the alphabet: "()[]{}<\>,.|*".

## Methods - Overview

| name | description |
|:-|:-|
| \_\_init\_\_ | Initialize a motif by providing sequences or a PWM. |
| plot | Plot the motif. |
## \_\_init\_\_

``` python
def __init__(self, alphabet, sequences = None, pwm = None)
```
Initialize a motif by providing sequences or a PWM. 

 Either a list of sequences or a PWM with shape (sequence length, alphabet length) must be provided. 



| parameter | type | description |
|:-|:-|:-|
| alphabet | str | The alphabet of the sequences. |
| sequences | [str] | A list of strings. All strings must have the same length. |
| pwm | numpy.ndarray | A matrix of shape (sequence length, alphabet length) containing probabilities. |
## plot

``` python
def plot(self, colors={}, scale=1)
```
Plot the motif. 

 The color of individual letters can be defined via the colors dict using RGB values, e.g. {'A': '#FF0000', 'C': '#0000FF'} will result in red A's and blue C's. Non-defined characters will be plotted black. 

 The alphabets 'ACGT', 'ACGU', and 'HIMS' have predefined colors (that can be overwritten): 

  "ACGT" -\> {'A': '#00CC00', 'C': '#0000CC', 'G': '#FFB300', 'T': '#CC0000'}  
  "ACGU" -\> {'A': '#00CC00', 'C': '#0000CC', 'G': '#FFB300', 'U': '#CC0000'}  
  "HIMS" -\> {'H': '#CC0000', 'I': '#FFB300', 'M': '#00CC00', 'S': '#CC00FF'}  
 

 Using, for instance, a scale parameter of 0.5 halves both height and width of the plot. 



| parameter | type | description |
|:-|:-|:-|
| colors | dict of char->str | A dict with individual alphabet characters as keys and hexadecimal RGB specifiers as values. |
| scale | float | Adjust the size of the plot (should be > 0). |

| returns | type | description |
|:-|:-|:-|
| image | PIL.image.image | A Pillow image object. |
