# PEC - Probabilistic Ensembles for Control

This module contains functions and classes to apply probabilistic machine learning techniques to control problems.

It contains standard neural networks and deep ensemble models as well as a constrained trajectory optimization method (constrained iLQR) for solving nonlinear optimal control problems.

In future versions it will be contain reinforcement learning techniques.


The project is structured as follows:

```
├──examples                             - example scripts
│   └──toy_examples                     - toy examples with deep ensemble models
│   
├──probecon                         - package
│   ├──control                      - control algorithms (trajectory optimization, reinforcement learning)
│   ├──data                         - data sets for machine learning tasks
│   ├──helpers                      - helper functions and wrappers for different packages
│   ├──neural_networks              - neural network models (MLP, Gaussian MLP, deep ensemble) 
│   ├──system_models                - system models ('gym'-environments)
│   │   └──symbtools_model_files    - files containing a 'symbtools.modeltools.SymbolicModel' 
│   │ 
│   └──tests                            - some unit tests
│ 
├──setup.py
│ 
└──README.md
```

It is possible to create `gym`-environments of your own mechanical system. The Lagrangian of the system is created using `sympy` and the equations of motion are derived with `symbtools.modeltools` and stored in a `symbtools.modeltools.SymbolicModel` object that is pickled to a file.

All models of dynamic systems inherit the `gym.Env`-class. This way it should be easy to apply algorithms from other packages the created environments.

```
gym.Env
    └──StateSpaceEnv                    - general state-space model environment (file: system_models/core.py)
            └──SymbtoolsEnv             - `symbtools` environment (file: system_models/core.py)
                ├──Acrobot              - acrobot system (file: system_models/core.py)
                ├──CartPole             - cart-pole system (file: system_models/cart_pole.py)
                ├──CartDoublePole       - cart-double-pole system (file: system_models/cart_double_pole.py)
                ├──CartTripePole        - cart-triple-pole system (file: system_models/cart_triple_pole.py)
                ├──CartQuadPole         - cart-quadruple-pole system (file: system_models/cart_quad_pole.py)
                ├──Pendulum             - inverted pendulum system (file: system_models/pendulum.py)
                └──PlanarManipulator    - planar two-link manipulator (file: system_models/planar_manipulator.py)
```



