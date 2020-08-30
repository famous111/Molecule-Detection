# Molecule Detection (EIT)

The python code with [QuTiP](https://qutip.org/) package for the simulation of the interaction between cold molecule and two resonator modes in the paper *Resonator-assisted single-molecule quantum state detection* ([PhysRevA.102.023716](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.023716)), which proposes a protocol to detect molecular quantum state with electromagnetically-induced transparency (EIT).

## Usage
```python
util.molSystem(duration, spacing, cooperativity, kappa_i, kappa_e, gamma=12, delta_al=0,  delta_cl=0, f_fc=0.37, P_in=1, N_p=3, N_s=3, N_m=1, save=True, save_path="../data/" + strftime("%Y%m%d_%H%M%S", localtime()) + "/", ))
```
A class to describe the interaction between molecule and resonator modes

### Parameters
`duration`: simulation time (unit: &mu;s)  
`spacing`: spacing of the sample points during simulation  
`cooperativity`: dimensionless cooperativity parameter in cavity QED  
`kappa_i`: intrinsic resonator loss rate (unit: MHz)  
`kappa_e`: coupling rate between waveguide and resonator (unit: MHz)  
`gamma`: excited molecule spontaneous decay rate (unit: MHz)  
`delta_al`: frequency difference between atomic transition and light frequency (unit: MHz)  
`delta_cl`: frequency difference between cavity resonance and light frequency (unit: MHz)  
`f_fc`: Franck-Condon factor
`P_in`: Photon number generation rate at the input side of waveguide
`N_p`: maximum photon number in Fock state
`N_s`: number of molecule states: |0>: |g>; |1>:|e>; |2>:|s>
`N_m`: initial number of molecules  
`save`: if `save == False`, disable all saving data option
`save_path`: the path to save data. Default value (on linux): "../data/localtime"

### Methods
`simulation(self, show_progress=None, save_info=True, save_data=False)`: calculate the Lindbladian master equation. `show_progress`(None/True) determines whether to show progress bar of calculation.; `save_info` determines whether to write system parameters into a txt file; `save_data` determines whether to save the simulation result into the `save_path` (same as below).   
`photon_simulated(self, save_data=False)`: Use simulation result to calculate real-time transmission and reflection as well as the transmitted and reflected photon number as a function of time  
`photon_theory(self, save_data=False)`: Use analytical solution to calculate real-time transmission and reflection as well as the transmitted and reflected photon number as a function of time  
`draw_population(self)`: draw the graph of molecular state population evolution  
`draw_comparison_tran(self)`: draw the real-time transmission curve with the data from simulation and analytical solution. (Used for calibration)

```python
util.tran_spectrum(Mol, delta_max, delta_step, sampletlist, method, calculation="parallel", save_data=False)
```
a function to calculate the average transmission spectrum and draw a graph of transmission spectrum

### Parameters
`Mol`: an instance of `molSystem` to provide background parameters (expect detuning)  
`delta_max`: maximum detuning frequency (unit: MHz)
`delta_step`: the increasing step of detuning frequency from zero (unit: MHz)
`sampletlist`: a list of sampling time points to calculate average transmission  
`method`: `'theory'` or `'simulation'`. Select the method to derive transmission. Default value: `'theory'`  
`calculation`: `'serial'` or `'parallel'`. Select serial or parallel computation. Default value: `'parallel'`  
`save_data`: determines whether to save the raw data into the `Mol._path`. 

## Examples
```python
from util import molSystem, tran_spectrum

Mol = molSystem(duration=201, spacing=0.01, cooperativity=50, kappa_i=50, kappa_e=50)

Mol.simulation(show_progress=True, save_data=True)
Mol.photon_simulated(save_data=True)
Mol.photon_theory(save_data=True)
Mol.draw_population()
Mol.draw_comparison_tran()

tran_spectrum(
    Mol,
    delta_max=1000,
    delta_step=10,
    sampletlist=[20, 100, 200],
    method="theory",
    calculation="parallel",
    save_data=True,
)

del Mol

```



