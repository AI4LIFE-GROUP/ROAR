# ROAR-Internal

## Requirements
To run this code first install the packages in requirements.txt. This code also requires custom extensions of the open sourced actionable-recourse and LIME libraries contained in `actionable-recourse/` and `lime/` respectively. Follow the requirements instructions and run `python setup.py` in each folder. 

## Organization
### data.py
This file contains the data loading and pre-processing modules.

### model.py
This file contains the LR, DNN, and SVM implementations.

### recourse_methods.py
This file contains the implementations of all the recourse methods. Our method, ROAR, is called `RobustRecourse`, CFE is called `counterfactual_recourse`, AR is called `actionable_recourse`, and CR is called `CausalRecourse`.

### recourse_utils.py
This file contains helper functions for the recourse methods and evaluation.

### run_rw.py
This file generates recourses for our real world data experiments.

### run_sim.py
This file generates recourses for our synthetic experiments.

### run_delta.py
This file generates recourses for our experiments varying $\delta_max$ (see Appendix). 