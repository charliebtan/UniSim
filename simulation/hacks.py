import numpy as np
from scipy.optimize import minimize
from openmm import unit

def openmm_energy_and_gradient(x, simulation, eval_counter):
    # Reshape x to match number of atoms
    x_nm = x.reshape((-1, 3)) * unit.nanometer

    # Update positions in OpenMM
    simulation.context.setPositions(x_nm)

    # Compute energy and forces
    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    grad = -forces.reshape(-1)  # SciPy expects gradient (negative force)

    eval_counter[0] += 1  # Count evaluations
    return energy, grad

def minimize_with_scipy(simulation, maxiter=100000):
    # Get initial positions
    state = simulation.context.getState(getPositions=True)
    x0 = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer).reshape(-1)

    eval_counter = [0]

    result = minimize(
        fun=lambda x: openmm_energy_and_gradient(x, simulation, eval_counter),
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        options={"gtol": 10.0, "maxiter": maxiter}  # OpenMM-like defaults
    )

    # Update OpenMM context to final positions
    x_final = result.x.reshape((-1, 3)) * unit.nanometer
    simulation.context.setPositions(x_final)

    return eval_counter[0]  # Return number of evaluations
