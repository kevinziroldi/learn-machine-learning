# linear regression consiste nel trovare la linea che descriva meglio i dati di train
# l'equazione di una retta è y = mx + q
# m = (mean(x)*mean(y) - mean(x*y)) / ((mean(x)^2) - mean(x^2))
# q = mean(y) - m * mean(x)

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

def best_fit_slope(xs, ys):
    m = ((mean(xs)*mean(ys)) - mean(xs*ys)) / ((mean(xs)**2) - mean(xs**2))
    return m

def best_fit_ (xs, ys):
    q = mean(ys) - (m * mean(xs))
    return q

m = best_fit_slope(xs, ys)
print(m)

q = best_fit_(xs, ys)
print(q)

predict_x = 8
predict_y = m * (predict_x) + q
print(predict_y)

regression_line = [(m*x)+q for x in xs]
plt.scatter(xs, ys, color='b')
plt.plot(xs, regression_line)
plt.scatter(predict_x, predict_y, color='r')
plt.show()



# per misurare quanto è buono un modello possiamo usare r2
# r2 = 1 - (SEy^ / SEy-)

# prima calcoliamo squared error
def squaredError(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squaredError(ys_orig, ys_line)
    squared_error_y_mean = squaredError(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)



def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys =[]
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg': 
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

xs, ys = create_dataset(40, 40, 2, correlation='pos')
regression_line = [(m*x)+q for x in xs]
plt.scatter(xs, ys, color='b')
plt.plot(xs, regression_line)
plt.scatter(predict_x, predict_y, color='r')
plt.show()