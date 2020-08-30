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
