import numpy as np

# WOA
# from pyMetaheuristic.algorithm import whale_optimization_algorithm
from pyMetaheuristic.utils import graphs
from whale_optimization_algorithm import whale_optimization_algorithm


# Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.
# For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

# Target Function: Easom Function
def easom(variables_values=[0, 0]):
    x1, x2 = variables_values
    func_value = -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)
    return func_value


# Target Function - Values
plot_parameters = {
    'min_values': (-5, -5),
    'max_values': (5, 5),
    'step': (0.1, 0.1),
    'solution': [],
    'proj_view': '3D',
    'view': 'notebook'
}
graphs.plot_single_function(target_function=easom, **plot_parameters)

# WOA - Parameters
parameters = {
    'hunting_party': 150,
    'min_values': (-5, -5),
    'max_values': (5, 5),
    'iterations': 500,
    'spiral_param': 0.5,
    'verbose': True
}
woa = whale_optimization_algorithm(target_function=easom, **parameters)

# WOA - Solution
variables = woa[0][:-1]
minimum = woa[0][-1]
print('Variables: ', np.around(variables, 4), ' Minimum Value Found: ', round(minimum, 4))
# WOA - Plot Solution
plot_parameters = {
    'min_values': (-5, -5),
    'max_values': (5, 5),
    'step': (0.1, 0.1),
    'solution': [variables],
    'proj_view': '3D',
    'view': 'notebook'
}
graphs.plot_single_function(target_function=easom, **plot_parameters)
