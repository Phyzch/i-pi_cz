"""Deals with creating the ensembles class.

Copyright (C) 2013, Joshua More and Michele Ceriotti

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http.//www.gnu.org/licenses/>.

Classes:
   InputNebInstGPR: Deals with creating the nebInstGPR object from a file, and
      writing the checkpoints.
"""

import numpy as np 
from ipi.utils.inputvalue import *

class InputNebInstGPR(InputDictionary):
    """ Use nudged elastic band to find minimum action path (MAP). The nudged elastic band method is accelerated by Gaussian Process Regression.
    Adopted from neb_instanton module.
    Contains options related with neb path optimization algorithm, spring force for neb etc.
    """
    attribs = {
        "mode" : (
            InputAttribute, 
            {
                "dtype" : str,
                "default" : "verlet",
                "help" : "Defines the method to evolve the nudged elastic band",
                "options": ["verlet"]
            },
        )
    }

    fields = {
        "tolerances":(
            InputDictionary,
            {
                "dtype": float,
                "options": ["gradient"],
                 "default": [5e-3],
                 "help": "Convergence criteria for neb optimization",
                 "dimension": ["undefined"]
            },
        ),

        "energy_shift":(
            InputValue,
            {
                "dtype": float,
                "default": 0.000,  
                "help": "Set the zero of energy (unit Hatree). Choose it for energy of reactant state.",
                "dimension": "energy"
            }
        ),

        "time_step":(
            InputValue,
            {
                "dtype": float,
                "default" : 4.00,  # = 0.1 fs
                "help": "time step for evolve neb beads",
                "dimension": "time"
            }
        ),

        "instanton_time_step":(
            InputValue,
            {
                "dtype": float,
                "default": 4.00,  # = 0.1 fs
                "help": "time step to evolve dynamics along Minimum action path to generate ring-polymer instanton",
                "dimension": "time"
            }
        ),

        "stage":(
            InputValue,
            {
                "dtype": str,
                "default": "neb",
                "options": ["neb", "instanton", "converged"],
                "help": """stage for neb pipline for finding instanton path.
                neb: using nudged elastic band to find minimum action path
                instanton: evolve dynamics to find the temperature of the found minimum action path and output ring polymer"""
            }
        ),

        # for instanton that we got from minimum action path doing MD along the path.
        "instanton_bead_number": (
            InputValue,
            {
               "dtype": int,
               "default": 20,
                "help": "number of ring polymers for instanton at the instanton stage of calculation." 
            }
        ),

        "instanton_path_energy":(
            InputValue,
            {
                "dtype": float,
                "default": 0.00,
                "help": "the end beads energy for minimum action path. with respect to the energy shift.",
                "dimension": "energy"
            }
        ),

        "instanton_temperature":(
            InputValue,
            {
                "dtype": float,
                "default": 1.00,
                "help": "the final calculated temperature for minimum action path: inverse of period (beta hbar) for periodic motion. (Used for saving result in RESTART)",
                "dimension": "temperature"
            }
        ),

        "instanton_bead_q":(
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "the bead coordinate for instanton beads (spaced equally along imaginary time) on the minimum action path. (Used for recording result in RESTART)",
                "dimension": "length"
            }
        ),

        "instanton_bead_pot":(
            InputArray,
            {
                "dtype": float,
                "default" : input_default(factory=np.zeros, args=(0,)),
                "help": "the potential energy for instanton beads (spaced equally along imaginary time) on the minimum action path. (Used for recording result in RESTART)"
            }
        ),
        
        "instanton_hessian":(
            InputArray,
            {
                "dtype": float,
                "default" : input_default(factory=np.eye, args=(0,)),
                "help": "the calculated Hessian for instanton beads. (Used for recording result in RESTART)"
            }
        ),

        # for MD along minimum actioin path.
        "path_interpolation_bead_number":(
            InputValue,
            {
                "dtype": int,
                "default": 20,
                "help": "the number of interpolation point along the minimum action path to compute imaginary time t = beta * hbar"
            }
        ),

        # for spring force term and energy constraint energy in nudged elastic band (NEB) algorithm
        "spring_k":(
            InputValue,
            {
                "dtype": float,
                "default": 0.1,
                "help": """
                        the spring constant for internal beads k(|r'' - r| - |r - r'|). unit (angstrom^-2 * atomic_mass^-1/2). 
                        Spring_k term will be adjusted according to time step dt: spring_k * dt^2 = 0.25 (empirical choice.)
                        Therefore, we do not need this specify this value. The value will be used for restarting the algorithm.
                        """
            }
        ),

        "kappa":(
            InputDictionary,
            {
                "dtype": float,
                "options":["left", "right"], 
                "default": [50, 50],
                "help": """the energy constraint term for beads at two ends to confine beads at iso-energy contour. 
                unit: (eV^(-1) angstrom^(-1) * atomic_mass^-1/2)
                See eq.(19) of THE JOURNAL OF CHEMICAL PHYSICS 148, 102334 (2018).
                We need two different values of kappa for asymmetric potential. 
                For symmetric potential, we can set left and right kappa to the same value.
                Two kappa values will be updated accordingly during the simulation using the force information |dV/dx|. So, we do not need to specify this value.
                        |dV/dx| * kappa / sqrt(m_H) * dt^2 = 0.5 (empirical value). The value will be used for restarting the algorithm. 
                        """
            },
        ),

        "final_hessian_bool":(
            InputValue,
            {
                "dtype" : bool,
                "default": False,
                "help": "Bool variable. whether to compute final hessian when we get the instanton trajectory."
            }
        ),

        "alt_out":(
            InputValue,
            {
                "dtype": int,
                "default" : 5,
                "help": "output instanton bead energy and geometry every alt_out step"
            }
        ),
        "prefix":(
            InputValue,
            {
                "dtype": str,
                "default": "neb_instanton",
                "help": "prefix of output file"
            }
        ),
        
        "gpr_force_criterion":(
            InputValue,
            {
                "dtype": float,
                "default": 0.05,
                "help": "convergence criterion for gpr outer loop. |f^GPR - f|/|f| < gpr_force_criterion means GPR prediction is reliable for force is reliable. Stop the outer loop."
            }

        ),

        "gpr_trust_region_ratio":(
            InputValue,
            {
                "dtype": float,
                "default": 0.03,
                "help": "trust region for Gaussian Process Regression r_max = gpr_trust_region_ratio * NEB_path_length. If the distance between NEB beads and nearest GPR point exceed r_max, we stop the NEB inner loop and evaluate ab-initio force on that point."
            }
        ),

        "gpr_kernel_outputscale":(
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "Gaussian Process Regression hyperparameter. This is an array, with each element corresponds to one Squared Exponential kernel we use to construct covariance function. Mean value for output scale prior of Gaussian process regression kernel. Typically it is the variance of potential energy"
            }
        ),

        "gpr_kernel_lengthscale_ratio":(
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "Gaussian Process Regression hyperparameter. This is an array, with each element corresponds to one Squared Exponential kernel we use to construct covariance function. Ratio of the mean value for the prior of the lengthscale of Gaussian Process regression model / length of bead in non-redundant internal coordinate. Typically this should be in the same order of the range of input data.",
                "dimension": "length"
            }
        ),

        "gpr_noise_std":(
            InputDictionary,
            {
                "dtype": float,
                "options": [ "pot_noise_prior", "force_noise_prior"],
                 "default": [1e-6, 1e-4],
                 "help": "constraint for the variance of noise in the Gaussian Process Regression",
            },
        ),

        "gpr_SE_kernel_number":(
            InputValue,
            {
                "dtype": int,
                "default": 1,
                "help" : "Number of Squared Exponential (SE) kernel used in Gaussian Process kernel. For symmetric potential system, 1 Squared Exponential kernel will fit the potential well. However, for asymmetric system, 2 Squared Exponential kernel is recommended."
            }
        ),

        "read_initial_gpr_training_data":(
            InputValue,
            {
                "dtype": bool,
                "default": False,
                "help" : "Bool variable to decide whether to read the stored training data."
            }
        )

    }

    dynamic = {}

    default_help = "A class for nudged elastic band to find minimum action path. Accelerated by Nudged Elastic band method."
    default_label = "NEB_INSTANTON_GPR"

    def store(self, geop):
        '''
        this function corresponds to how we name variables in neb_instanton.py
        '''
        if geop == {}:
            return 
        
        options = geop.options
        optarrays = geop.optarrays
        
        # options
        self.mode.store(options["mode"])
        self.stage.store(options["stage"])
        self.tolerances.store(options["tolerances"])
        self.alt_out.store(options["alt_out_step"])
        self.prefix.store(options["prefix"])
        self.final_hessian_bool.store(options["final_hessian_bool"])

        # options for GPR kernel
        self.gpr_SE_kernel_number.store(options["gpr_SE_kernel_number"])
        self.read_initial_gpr_training_data.store(options["read_initial_gpr_training_data"])

        # optarrays
        self.energy_shift.store(optarrays["energy_shift"])
        self.spring_k.store(optarrays["spring_k"])
        self.kappa.store(optarrays["kappa"])
        self.time_step.store(optarrays["time_step"])
        self.instanton_time_step.store(optarrays["instanton_time_step"])

        self.instanton_path_energy.store(optarrays["instanton_path_energy"])
        self.instanton_bead_number.store(optarrays["instanton_bead_number"])
        self.path_interpolation_bead_number.store(optarrays["path_interpolation_bead_number"]) 

        # store parameters about gaussian process regression
        self.gpr_force_criterion.store(optarrays["gpr_force_criterion"])
        self.gpr_trust_region_ratio.store(optarrays["gpr_trust_region_ratio"])
        self.gpr_kernel_outputscale.store(optarrays["gpr_kernel_outputscale"])
        self.gpr_kernel_lengthscale_ratio.store(optarrays["gpr_kernel_lengthscale_ratio"])
        self.gpr_noise_std.store(optarrays["gpr_noise_std"])


        # store result of instanton calculation
        self.instanton_temperature.store(optarrays["instanton_temperature"])
        self.instanton_bead_q.store(optarrays["instanton_bead_q"]) 
        self.instanton_bead_pot.store(optarrays["instanton_bead_pot"])
        self.instanton_hessian.store(optarrays["instanton_hessian"])

        

    def fetch(self):
        rv = super(InputNebInstGPR, self).fetch()
        rv["mode"] = self.mode.fetch()
        return rv 
