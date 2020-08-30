from qutip import qeye, create, destroy, basis, tensor, fock, mesolve, expect, Options
from qutip.parallel import parallel_map
from numpy import sqrt, pi, linspace, arange, save, array, copy, exp, concatenate, flip, transpose
from numpy.linalg import eig, solve
import matplotlib.pyplot as plt
from time import strftime, localtime
from os import makedirs
from os.path import exists, abspath
from functools import partial
from copy import deepcopy
from typing import List


def mkdir(path: str):

    if not exists(path):  # determine the existance of folder
        makedirs(path)  # makedirs create folder if the path does not exist
        print("---  create new folder...  ---")
        print("--- path:", path, "---")

    else:
        print("---  There is a folder!  ---")


class molSystem(object):
    def __init__(
        self,
        duration: int,
        spacing: int,
        cooperativity,
        kappa_i,
        kappa_e,
        gamma=12,
        delta_al=0,
        delta_cl=0,
        f_fc=0.37,
        P_in=1,
        N_p=3,
        N_s=3,
        N_m=1,
        save=True,
        save_path="../data/" + strftime("%Y%m%d_%H%M%S", localtime()) + "/",
    ):
        """
        Unit of kappa_i, kappa_e, gamma, P_in: MHz
        """
        self.duration = duration
        self.spacing = spacing
        self.tlist = arange(0, self.duration, self.spacing)
        self.C = cooperativity  # Cooperativity (Note: not take into account Franck-Condon factor)
        self.kappa_i = 2 * pi * kappa_i  # Unit: Mega rad/s, intrinsic resonator loss rate
        self.kappa_e = (
            2 * pi * kappa_e
        )  # Unit: Mega rad/s, coupling rate between waveguide and resonator
        self.kappa = self.kappa_i + self.kappa_e  # Unit:Mega rad/s cavity photon decay rate
        self.gamma = 2 * pi * gamma  # Unit: Mega rad/s excited molecule spontaneous decay rate
        self.g = sqrt(self.C * self.gamma * self.kappa)  # Unit: MHz
        self.delta_al = (
            2 * pi * delta_al
        )  # difference between atomic transition and light frequency
        self.delta_cl = 2 * pi * delta_cl  # difference between cavity resonance and light frequency
        self.f_fc = f_fc  # Franck Condon factor
        self.gamma_g = self.gamma * self.f_fc  # decay rate to vibrational ground state
        self.gamma_s = self.gamma * (1 - self.f_fc)  # decay rate to higher vibrational states
        self.P_in = P_in  # Unit: MHz Photon number generation rate at input side of waveguide
        self.E = 1j * sqrt(2 * self.kappa_e * self.P_in)  # Intracavity field
        self.Omega = -2 * sqrt(2 * self.kappa_e * self.P_in)  # Rabi frequency
        self.N_p = N_p  # maximum photon number in Fock space
        self.N_s = N_s  # Number of molecule states: |0>: |g>; |1>:|e>; |2>:|s>
        self.N_m = N_m  # number of molecules
        self._path = save_path  # saving path of data
        self._result = False  # flag of simulation
        self._save = save

        if self._save == True:
            mkdir(self._path)

    ### numerical simulation ###
    def _init_simulation(self):
        """
        create the Hamiltonian, initial state and collapse operator of the molecule-cavity system
        """
        # define initial state

        self.psi0 = tensor(
            [fock(self.N_p, 0), fock(self.N_p, 0)] + [basis(self.N_s, 0) for _ in range(self.N_m)]
        )  # CW mode + CCW mode + molecular internal state

        # ----------------define operators-------------------
        # define identity operator for all molecular internal state
        iden = [qeye(self.N_s) for m in range(self.N_m)]  # identity operator for all molecules

        # ----------------Hamiltonian-----------------------
        # annihilation operator of whispering gallery mode
        self.aCW = tensor([destroy(self.N_p), qeye(self.N_p)] + iden)
        self.aCCW = tensor([qeye(self.N_p), destroy(self.N_p)] + iden)

        H0 = (
            self.delta_cl * self.aCW.dag() * self.aCW
            + self.delta_cl * self.aCCW.dag() * self.aCCW
            + 1j * (self.E * self.aCW.dag() - self.E.conjugate() * self.aCW)
        )

        # excited state energy
        temp = []
        for m in range(self.N_m):
            op_temp = iden[:]
            op_temp[m] = self.delta_al * (basis(self.N_s, 1) * basis(self.N_s, 1).dag())
            temp.append(op_temp)
        H1 = sum([tensor([qeye(self.N_p), qeye(self.N_p)] + temp[m]) for m in range(self.N_m)])

        # Coupled to cavity
        temp = []
        for m in range(self.N_m):
            op_temp = iden[:]
            op_temp[m] = basis(self.N_s, 0) * basis(self.N_s, 1).dag()
            temp.append(op_temp)
        op = (
            1j
            * self.g
            * sum([tensor([create(self.N_p), qeye(self.N_p)] + temp[m]) for m in range(self.N_m)])
        )
        op += (
            1j
            * self.g
            * sum([tensor([qeye(self.N_p), create(self.N_p)] + temp[m]) for m in range(self.N_m)])
        )
        H1 += op

        temp = []
        for m in range(self.N_m):
            op_temp = iden[:]
            op_temp[m] = basis(self.N_s, 1) * basis(self.N_s, 0).dag()
            temp.append(op_temp)
        op = (
            -1j
            * self.g
            * sum([tensor([destroy(self.N_p), qeye(self.N_p)] + temp[m]) for m in range(self.N_m)])
        )
        op += (
            -1j
            * self.g
            * sum([tensor([qeye(self.N_p), destroy(self.N_p)] + temp[m]) for m in range(self.N_m)])
        )
        H1 += op

        # total Hamiltionian
        self.H = H0 + H1

        # -----------------collapse operators--------------
        self.c_ops = []
        if self.kappa > 0:
            self.c_ops.append(sqrt(2 * self.kappa) * self.aCW)
            self.c_ops.append(sqrt(2 * self.kappa) * self.aCCW)

        # spontaneous decay to higher vibrational state molecule
        if self.gamma_s > 0:
            for m in range(self.N_m):
                op_temp = iden[:]
                op_temp[m] = basis(self.N_s, 2) * basis(self.N_s, 1).dag()
                self.c_ops.append(
                    sqrt(self.gamma_s) * tensor([qeye(self.N_p), qeye(self.N_p)] + op_temp)
                )

        # spontaneous decay to vibrational ground state molecule
        if self.gamma_g > 0:
            for m in range(self.N_m):
                op_temp = iden[:]
                op_temp[m] = basis(self.N_s, 0) * basis(self.N_s, 1).dag()
                self.c_ops.append(
                    sqrt(self.gamma_g) * tensor([qeye(self.N_p), qeye(self.N_p)] + op_temp)
                )

    def simulation(self, show_progress=None, save_info=True, save_data=False):
        """
        Numerically calculate the evolution of system via Lindblad master equation
        The unit of duration, spacing: us 
        show_progress (None/True): show progress bar for the master equation calculation
        if self._save = True
            save_info: record system information in a txt file
            save_data: save the simulation results in npy files
        """
        if self._result == True:
            print(
                "The system has been simulated. Create a new folder to save data if self._save = True."
            )
            self._path = ("../data/" + strftime("%Y%m%d_%H%M%S", localtime()) + "/",)
            if self._save == True:
                mkdir(self._path)

        if (save_info and self._save) == True:
            # write setting to a txt file
            try:
                para = open(self._path + "parameters.txt", "a")
                para.write(
                    " P_in = %f MHz\n C = %f\n g = %fMHz\n kappa_i=%fMHz\n kappa_e=%fMHz\n gamma = %fMHz\n duration = %fus\n sampling point number = %d\n f_fc = %f\n N_m = %d\n N_p = %d\n file path: %s\n save path: %s"
                    % (
                        self.P_in,
                        self.C,
                        self.g / (2 * pi),
                        self.kappa_i / (2 * pi),
                        self.kappa_e / (2 * pi),
                        self.gamma / (2 * pi),
                        self.duration,
                        self.spacing,
                        self.f_fc,
                        self.N_m,
                        self.N_p,
                        abspath(__file__),
                        self._path,
                    )
                )
            except IOError:
                print("File error")
            finally:
                para.close()

        self._init_simulation()

        # -----------------projection operators------------------
        # define identity operator for all molecular internal state
        iden = [qeye(self.N_s) for m in range(self.N_m)]  # identity operator for all molecules

        # (i-1)th element in operator list is the projection operator for i-th molecule
        # ground state
        temp = []
        for m in range(self.N_m):
            op_temp = iden[:]
            op_temp[m] = basis(self.N_s, 0).proj()
            temp.append(op_temp)
        s1_proj = [tensor([qeye(self.N_p), qeye(self.N_p)] + temp[m]) for m in range(self.N_m)]

        # excited state
        temp = []
        for m in range(self.N_m):
            op_temp = iden[:]
            op_temp[m] = basis(self.N_s, 1).proj()
            temp.append(op_temp)
        s2_proj = [tensor([qeye(self.N_p), qeye(self.N_p)] + temp[m]) for m in range(self.N_m)]

        # other states
        temp = []
        for m in range(self.N_m):
            op_temp = iden[:]
            op_temp[m] = basis(self.N_s, 2).proj()
            temp.append(op_temp)
        s3_proj = [tensor([qeye(self.N_p), qeye(self.N_p)] + temp[m]) for m in range(self.N_m)]

        exp_list = []
        exp_list.append(s1_proj[0])  # ground state (first molecule)
        exp_list.append(s2_proj[0])  # excited state (first molecule)
        exp_list.append(s3_proj[0])  # other states (first molecule)
        exp_list.append(
            self.aCW.dag() * self.aCW + self.aCCW.dag() * self.aCCW
        )  # photon population in cavity
        exp_list.append(self.aCW)  # expectation value of aCW
        exp_list.append(self.aCCW)  # expectation value of aCCW
        exp_list.append(self.aCW.dag() * self.aCW)  # CW mode photon population in cavity
        exp_list.append(self.aCCW.dag() * self.aCCW)  # CCW mode photon population in cavity

        # ------------------calcualte master equation-----------------
        # set absolute tolerance for solver. Check qutip.solver.Options
        # options = Options(atol=1e-14, nsteps=10000)
        output = mesolve(
            self.H,
            self.psi0,
            self.tlist,
            self.c_ops,
            exp_list,
            # options=options,
            progress_bar=show_progress,
        )

        self.gState = output.expect[0]
        self.eState = output.expect[1]
        self.sState = output.expect[2]
        self.cavity_photon = output.expect[3]
        self.exp_aCW = output.expect[4]
        self.exp_aCCW = output.expect[5]

        if (save_data and self._save) == True:
            save(self._path + "tlist.npy", self.tlist)
            save(self._path + "gState.npy", output.expect[0])
            save(self._path + "eState.npy", output.expect[1])
            save(self._path + "sState.npy", output.expect[2])
            save(self._path + "cavity_photon.npy", output.expect[3])
            save(self._path + "aCW.npy", output.expect[4])
            save(self._path + "aCCW.npy", output.expect[5])

        self._result = True

    ### theoretical derivation ###
    """
    Calculate auxilary functions for theoretical derivation
    """

    def _init_theory(self):
        """
        output:
        N_m + 1 dimensional array: population ratio of expectation of aCW in each manifold
        """
        g_list = array([self.g * sqrt(i) for i in range(1, self.N_m + 1)])
        # write a program for excited state population
        pop31n = list(
            -(self.Omega / 2 * 1j)
            * (
                (g_list ** 2)
                + (1j * self.delta_cl + self.kappa) * (1j * self.delta_al + self.gamma / 2)
            )
            / (
                2 * (g_list ** 2)
                + (1j * self.delta_cl + self.kappa) * (1j * self.delta_al + self.gamma / 2)
            )
            / (1j * self.delta_cl + self.kappa)
        )  # proportion coefficient between excited state and ground state for n>0

        pop0 = [(-1j * self.Omega / 2 / (1j * self.delta_cl + self.kappa))]
        # proportion coefficient between excited state and ground state for n=0

        self.pop31 = (
            pop0 + pop31n
        )  # N_m + 1 dimensional array: population ratio of expectation of aCW in each manifold
        # print("g list:", g_list)
        # print("excited state coefficient:", pop)

        self.pop41 = list(
            -(self.Omega / 2 * 1j)
            * (-(g_list ** 2))
            / (
                2 * (g_list ** 2)
                + (1j * self.delta_cl + self.kappa) * (1j * self.delta_al + self.gamma / 2)
            )
            / (1j * self.delta_cl + self.kappa)
        )  # N_m dimensional array: population ratio of expectation of aCCW in each manifold (N>0)

        """
        calculate the population distribution in n-molecule manifolds. 
        return: pop_dynamics = [rho0(t), rho1(t), rho2(t), ... , rhoN(t) ] N+1 dimension array
        """

        def decayRate(N: int):
            """
            N: the number of molecules coupled to cavity
            Return decay rate of N-molecule manifold
            """
            g_eff = sqrt(N) * self.g
            return ((g_eff ** 2) * (self.Omega ** 2) * self.gamma_s) / (
                16 * (g_eff ** 4)
                + 8 * (g_eff ** 2) * (-2 * self.delta_al * self.delta_cl + self.gamma * self.kappa)
                + ((self.gamma ** 2) + 4 * (self.delta_al ** 2))
                * ((self.delta_cl ** 2) + (self.kappa ** 2))
            )

        dimension = self.N_m + 1
        decayRateList = [decayRate(i) for i in range(1, dimension)]
        matrix = []
        temp = [0 for _ in range(dimension)]
        temp[1] = decayRateList[0]
        matrix.append(temp)
        for i in range(1, self.N_m):
            temp = [0 for _ in range(dimension)]
            temp[i] = -decayRateList[i - 1]
            temp[i + 1] = decayRateList[i]
            matrix.append(temp)
        temp = [0 for _ in range(dimension)]
        temp[self.N_m] = -decayRateList[self.N_m - 1]
        matrix.append(temp)

        matrix = array(matrix)
        w, v = eig(matrix)
        # print("eigenvalues:", w)
        # print("eigenvectors:\n", v)

        initial_state = array([0 for _ in range(dimension)])
        initial_state[-1] = 1
        coeffient = solve(v, initial_state)
        # print("coefficient:\n", coeffient)
        func_coff = copy(v)
        for i in range(dimension):
            func_coff[:, i] *= coeffient[i]
        # print("func_coff:\n", func_coff
        self.pop_dynamics = [
            sum([exp(w[i] * self.tlist) * coff[i] for i in range(len(w))]) for coff in func_coff
        ]

    ### calculate transmitted and reflected photons
    def photon_simulated(self, save_data=False):
        if not self._result:
            print("Error: have not simulated system yet.")
            return

        # calculate transmission and reflection
        self.tran = (
            abs(1 + 1j * sqrt(2 * self.kappa_e / self.P_in) * self.exp_aCW) ** 2
        )  # real-time transmission
        self.refl = (
            abs(1j * sqrt(2 * self.kappa_e / self.P_in) * self.exp_aCCW) ** 2
        )  # real-time reflection

        # calculate photon number
        time_step = self.tlist[1] - self.tlist[0]
        self.tran_photon_nums = []
        self.refl_photon_nums = []
        p_num = 0
        for transmission in self.tran:
            p_num += transmission * time_step * self.P_in
            # transmission photon number at specific time
            self.tran_photon_nums.append(p_num)
        p_num = 0
        for reflection in self.refl:
            p_num += reflection * time_step * self.P_in
            # reflection photon number at specific time
            self.refl_photon_nums.append(p_num)
        if self._save and save_data:
            save(self._path + "tran.npy", self.tran)
            save(self._path + "refl.npy", self.refl)
            save(self._path + "tran_photon_nums.npy", self.tran_photon_nums)
            save(self._path + "refl_photon_nums.npy", self.refl_photon_nums)

    def photon_theory(self, save_data=False):
        self._init_theory()
        e_pop = self.pop31
        g_pop_dynamics = self.pop_dynamics

        # calculate transmission and reflection
        self.exp_aCW_theory = sum([e_pop[i] * g_pop_dynamics[i] for i in range(self.N_m + 1)])
        self.tran_theory = (
            abs(1 + 1j * sqrt(2 * self.kappa_e / self.P_in) * self.exp_aCW_theory) ** 2
        )

        e_pop = self.pop41
        self.exp_aCCW_theory = sum([e_pop[i] * g_pop_dynamics[i + 1] for i in range(self.N_m)])
        self.refl_theory = abs(1j * sqrt(2 * self.kappa_e / self.P_in) * self.exp_aCCW_theory) ** 2

        # calculate photon number
        time_step = self.tlist[1] - self.tlist[0]
        self.tran_photon_nums_theory = []
        self.refl_photon_nums_theory = []
        p_num = 0
        for transmission in self.tran_theory:
            p_num += transmission * time_step * self.P_in
            # transmission photon number at specific time
            self.tran_photon_nums_theory.append(p_num)
        p_num = 0
        for reflection in self.refl_theory:
            p_num += reflection * time_step * self.P_in
            # reflection photon number at specific time
            self.refl_photon_nums_theory.append(p_num)

        if self._save and save_data:
            save(self._path + "tran_theory.npy", self.tran_theory)
            save(self._path + "refl_theory.npy", self.refl_theory)
            save(self._path + "tran_photon_nums_theory.npy", self.tran_photon_nums_theory)
            save(self._path + "refl_photon_nums_theory.npy", self.refl_photon_nums_theory)

    # -------------------------drawing-------------------------------------
    def draw_population(self):
        if not self._result:
            print("Error: have not simulated system yet.")
            return

        fig, axes = plt.subplots(1, 1, figsize=(10, 8))

        axes.plot(self.tlist, self.gState, color="k", label=r"$|g\rangle$", linewidth=5)
        axes.plot(self.tlist, self.eState, color="b", label=r"$|e\rangle$", linewidth=5)
        axes.plot(self.tlist, self.sState, color="r", label=r"$|s\rangle$", linewidth=5)

        axes.legend(loc=6, fontsize=20)
        # axes.set_xscale("log")
        axes.set_xlabel("Time/us", fontsize=28)
        axes.set_ylabel("Population", fontsize=28)
        axes.tick_params(axis="both", which="major", labelsize=25)
        axes.tick_params(axis="both", which="minor", labelsize=12)
        if not exists(self._path + "figures"):  # detect figures folder
            makedirs(self._path + "figures")
        plt.savefig(self._path + "figures/Population_Dynamics.png")

    def draw_comparison_tran(self):
        if not hasattr(self, "tran"):
            print("Error: call photon_simulated method first")
            return

        if not hasattr(self, "tran_theory"):
            print("Error: call photon_theory method first")
            return

        fig, axes = plt.subplots(1, 1, figsize=(10, 8))

        axes.plot(self.tlist, self.tran, color="k", label="simulation", linewidth=5)
        axes.plot(
            self.tlist,
            self.tran_theory,
            color="b",
            label="theoretical",
            linestyle="dashed",
            linewidth=5,
        )

        axes.set_xlim(left=0.5)
        axes.set_ylim(top=0.3)
        axes.legend(loc=1, fontsize=18)
        axes.set_xscale("log")
        axes.set_xlabel("Time/us", fontsize=28)
        axes.tick_params(axis="both", which="major", labelsize=25)
        axes.tick_params(axis="both", which="minor", labelsize=12)
        if not exists(self._path + "figures"):  # detect figures folder
            makedirs(self._path + "figures")
        plt.savefig(self._path + "figures/Transmission_Comparison.png")


### calculate transmission spectrum
def _helper(delta: float, Mol_original: molSystem, sampletlist: List[float], method: str):
    """
    input: method: 'simulation' or 'theory'
    output: transmission at sampling time spot
    """
    Mol_temp = deepcopy(Mol_original)
    Mol_temp.delta_al = Mol_temp.delta_cl = delta
    Mol_temp._save = False
    if method == "simulation":
        Mol_temp._init_simulation()
        Mol_temp.simulation(show_progress=None, save_info=False, save_data=False)
        Mol_temp.photon_simulated()
        tran = Mol_temp.tran
    elif method == "theory":
        Mol_temp._init_theory()
        Mol_temp.photon_theory()
        tran = Mol_temp.tran_theory
    else:
        raise ValueError("method value should be 'simulation' or 'theory'")

    average_tran = []
    time_step = Mol_temp.tlist[1] - Mol_temp.tlist[0]
    t_idx_list = list(array(sampletlist) // time_step)
    t_flag = t_idx_list.pop(0)
    p_num = 0
    for t_idx, transmission in enumerate(tran):
        # average tranmission=total transmission photon number / total input photon number
        p_num += transmission
        if t_idx == t_flag:
            # t_idx + 1 represents time
            average_tran.append(p_num / (t_idx + 1))
            if len(t_idx_list) > 0:
                t_flag = t_idx_list.pop(0)
            else:
                break
    del Mol_temp
    return average_tran


def tran_spectrum(
    Mol: molSystem,
    delta_max: float,
    delta_step: float,
    sampletlist: List[float],
    method="theory",
    calculation="parallel",
    save_data=False,
):
    """
    input:
    an instance of molSystem (neglect the detuning setting)
    delta_max: Unit: MHz. positive maximum detuning frequency. 
    delta_step: Unit: MHz. Increasing step of detuning frequency from zero
    sampletlist: The list of sampling time to draw spectrum
    output:
    transmission spectrum data
    """
    if Mol.tlist[-1] < sampletlist[-1]:
        raise ValueError("tlist range does not cover sampletlist!")
    if method == "simulation" or method == "theory":
        print("The way to derive transmission spectrum is " + method)
    else:
        raise ValueError("method value should be 'simulation' or 'theory'")
    if calculation == "serial" or calculation == "parallel":
        print("The way to derive transmission spectrum is " + calculation)
    else:
        raise ValueError("calculation value should be 'serial' or 'parallel'")

    deltalist = arange(0, delta_max, delta_step)

    if calculation == "serial":
        # serial version
        spectrum_data = []
        for delta in deltalist:
            spectrum_data.append(_helper(2 * pi * delta, Mol, sampletlist, method))
    elif calculation == "parallel":
        # parallel version
        spectrum_data = parallel_map(
            _helper,
            2 * pi * array(deltalist),  # convert the unit to Mega rad/s
            task_kwargs=dict(Mol_original=Mol, sampletlist=sampletlist, method=method),
            progress_bar=True,
        )

    if save_data:
        save(Mol._path + "tran_spectrum.npy", spectrum_data)
        save(Mol._path + "deltalist.npy", array(deltalist))
        save(Mol._path + "sampletlist.npy", array(sampletlist))

    # draw graph
    delta_minus = -copy(deltalist)
    delta_data = concatenate([flip(delta_minus), deltalist])
    spec_minus = flip(copy(spectrum_data), axis=0)
    spec_data = transpose(concatenate([spec_minus, spectrum_data]))

    fig, axes = plt.subplots(1, 1, figsize=(10, 8))

    for t_idx, spec in enumerate(spec_data):
        axes.plot(delta_data, spec, label=str(sampletlist[t_idx]) + r"$\mu s$")

    axes.legend(loc=0)
    if not exists(Mol._path + "figures"):  # detect figures folder
        makedirs(Mol._path + "figures")
    plt.savefig(Mol._path + "figures/Transmission_Spectrum.png")


if __name__ == "__main__":
    Mol = molSystem(duration=201, spacing=0.01, cooperativity=50, kappa_i=50, kappa_e=50)

    # Mol.simulation(show_progress=True, save_data=True)
    # Mol.photon_simulated(save_data=True)
    # Mol.photon_theory(save_data=True)
    # Mol.draw_population()
    # Mol.draw_comparison_tran()

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

