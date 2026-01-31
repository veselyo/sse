# Solution Space Engineering (SSE)

## Motivation
- In early engineering design phases, there is high uncertainty in final specifications, materials, manufacturing methods, and regulatory requirements. The earlier in the design phase, the larger are the possible changes to design parameters.
- Traditional optimization approaches produce designs that are extremely sensitive to parameter changes. Inevitable later-stage adjustments can trigger extensive redesign and high costs.
- Traditional optimization methods may be ill-suited or inefficient for complex real-world problems.

## Description
### Managing Uncertainty through Solution Spaces
- Instead of identifying a single optimized design point, SSE identifies a multi-dimensional solution box.
- For each design parameter, a permissible range interval is defined independently from other
parameters. A solution box is a product of these intervals in the entire design space.
- This independence significantly reduces sensitivity, as changes in one parameter rarely require
simultaneous redesign of other parameters.
- Uncertainties in early design phases and inevitable later-stage adjustments can be accommodated
without extensive redesign.
### Application to Complex Problems
- SSE can be used for arbitrary non-linear high dimensional problems.
- SSE quickly converges to solution, making it computationally practical for problems where traditional
optimization would require an infeasible number of evaluations.

## Example Problem 1: Front Crash
- Assume a vehicle front structure designed for a front crash against a rigid barrier with deformation possible at the
deformation forces F1 and F2.
− The system design goals for the crash performance are: 1) Keep the deceleration a of the passenger cell below a critical
threshold level, and 2) Ensure an ordered deformation starting from the front of the vehicle.


![Solution](front_crash/outputs/run_1_box.png)
![Good Designs Fraction](front_crash/outputs/run_1_good_designs_fraction.png)
![Normalized Volume](front_crash/outputs/run_1_normalized_vol.png)

## AERIS Demo
![Good Designs Fraction](aeris/outputs/run_1_good_designs_fraction.png)
![Normalized Volume](aeris/outputs/run_1_normalized_vol.png)
![Length](aeris/outputs/run_1_length.png)
![Overhang](aeris/outputs/run_1_overhang.png)
![Width](aeris/outputs/run_1_width.png)
![Greenhouse](aeris/outputs/run_1_greenhouse.png)
