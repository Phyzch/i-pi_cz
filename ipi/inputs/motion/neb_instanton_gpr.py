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
    """Use nudged elastic band to find minimum action path (MAP). The nudged elastic band method is accelerated by Gaussian Process Regression.
    Adopted from neb_instanton module.
    Contains options related with neb path optimization algorithm, spring force for neb etc.
    """

    attribs = {
        "mode": (
            InputAttribute,
            {
                "dtype": str,
                "default": "verlet",
                "help": "Defines the method to evolve the nudged elastic band",
                "options": ["verlet", "cg", "FIRE"],
            },
        ),

        "cal_type": (
            InputAttribute,
            {
                "dtype": str,
                "default": "rate",
                "help": "Define the type of calculation. rate / tunneling_splitting.",
                "options": ["rate", "splitting"]
            }
        )
    }

    fields = {
        "fix_dofs":(
            InputArray,
            {
                "dtype": int,
                "default": input_default(factory= np.zeros, args=(0,)),
                "help": "degrees of freedom in molecules to be fixed. Used for planar molecule case",
            }
        ),

        "asr": (
            InputValue,
            {
                "dtype": str,
                "default": "none",
                "options": ["none", "poly", "lin", "crystal"],
                "help": "Removes the zero frequency vibrational modes depending on the symmerty of the system.",
            },
        ),

        "tolerances": (
            InputDictionary,
            {
                "dtype": float,
                "options": ["gradient",
                            "gradient_end_bead", 
                            "action_forces_sum",
                            "action"],
                "default": [5e-3,
                            1e-2,
                            5e-3,
                            1e-4],
                "help": "Convergence criteria for neb optimization\
                    gradient: gradient for internal beads. \
                    gradient_end_bead: gradient for end beads \
                    action_forces_sum: sum of transverse gradient of internal beads. \
                    action: the decrease of action when we drift all beads.",
            },
        ),
        "energy_shift": (
            InputValue,
            {
                "dtype": float,
                "default": 0.000,
                "help": "Set the zero of energy (unit Hatree). Choose it for energy of reactant state.",
                "dimension": "energy",
            },
        ),
        "FIRE":(
            InputDictionary,
            {
                "dtype": float,
                "options": ["tmax",
                            "tmin",
                            "Ndelay",
                            "finc",
                            "fdec", 
                            "alpha0", 
                            "alpha_shrink",
                            "Nmax",
                            "maxstep",
                            "neb_step_update_kappa"],
                "default": [4.0,
                            0.1,
                            5,
                            1.1,
                            0.5,
                            0.15,
                            0.99,
                            100,
                            100, 
                            20
                            ],
                "help":"""
                Parameters for FIRE (Fast Inertial relaxation engine).
                tmax: the maximum time step is : tmax * dt. Be careful about this tmax value, you have to make sure MD converge.
                tmin: the minimum time step is : tmin * dt 
                Ndelay: Number of steps to wait after P<0 before we accelerate downhill
                finc: factor to increase dt when going downhill
                fdec: factor to decrease dt when going uphill
                alpha0: initial coefficient for mixing velocity and force vectors.
                alpha_shrink: factor to decrease alpha when going downhill.
                Nmax: maximum uphill step before we end the program.
                maxstep: maximum dx (in mass scaled coordinate) for each FIRE step.
                neb_step_update_kappa: number of neb steps until we refresh the energy constraint term kappa value.
                """
            }
        ),

        "time_step": (
            InputValue,
            {
                "dtype": float,
                "default": 4.00,  # = 0.1 fs
                "help": """"time step for evolve neb beads with projected velocity verlet. 
                If mode == "verlet": this is the time step for projected velocity verlet algorithm.
                If mode == "FIRE", this is the initial time step for the Fast Inertial relaxation engine method.
                """,
                "dimension": "time",
            },
        ),

        "cg_big_step": (
            InputValue,
            {
                "dtype": float,
                "default": 1.0,
                "help": """ time step for conjugate gradient search.
                If mode == "cg", this is the largest time step for line search using conjugate gradient method.
                """,
            }
        ),

        "instanton_time_step": (
            InputValue,
            {
                "dtype": float,
                "default": 4.00,  # = 0.1 fs
                "help": "time step to evolve dynamics along Minimum action path to generate ring-polymer instanton",
                "dimension": "time",
            },
        ),
        "stage": (
            InputValue,
            {
                "dtype": str,
                "default": "neb",
                "options": ["neb", "instanton", "converged"],
                "help": """stage for neb pipline for finding instanton path.
                neb: using nudged elastic band to find minimum action path
                instanton: evolve dynamics to find the temperature of the found minimum action path and output ring polymer""",
            },
        ),
        "opt": (
            InputValue,
            {
                "dtype": str,
                "default": "neb",
                "options": ["neb", "string"],
                "help": """ optimization method for instanton path searching.
                We provide 2 different methods: LINEB (J. Chem. Phys. 148, 102334 (2018)) and
                string method (Phys. Rev. B 66, 052301) 
                """
            }
        ),
        # for instanton that we got from minimum action path doing MD along the path.
        "instanton_bead_number": (
            InputValue,
            {
                "dtype": int,
                "default": 20,
                "help": "number of ring polymers for instanton at the instanton stage of calculation.",
            },
        ),
        "instanton_path_energy": (
            InputValue,
            {
                "dtype": float,
                "default": 0.00,
                "help": "the end beads energy for minimum action path. with respect to the energy shift.",
                "dimension": "energy",
            },
        ),
        "instanton_temperature": (
            InputValue,
            {
                "dtype": float,
                "default": 1.00,
                "help": """ the final calculated temperature for minimum action path: inverse of period (beta hbar) for periodic motion. 
                (Used for saving result in RESTART) """,
                "dimension": "temperature",
            },
        ),
        "instanton_bead_q": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": """the bead coordinate for instanton beads (spaced equally along imaginary time)
                on the minimum action path. (Used for recording result in RESTART)""",
                "dimension": "length",
            },
        ),
        "instanton_bead_pot": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": """the potential energy for instanton beads (spaced equally along imaginary time) 
                on the minimum action path. (Used for recording result in RESTART)""",
            },
        ),
        "instanton_hessian": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.eye, args=(0,)),
                "help": "the calculated Hessian for instanton beads. (Used for recording result in RESTART)",
            },
        ),

        "neb_inner_loop_step_max": (
            InputValue,
            {
                "dtype": int,
                "default": 100, 
                "help" : """
                The maximum step number in the inner loop before we claim the algorithm fails to converge.
                """
            }
        ),

        # for spring force term and energy constraint energy in nudged elastic band (NEB) algorithm
        "spring_k": (
            InputValue,
            {
                "dtype": float,
                "default": 0.1,
                "help": """
                        the spring constant for internal beads k(|r'' - r| - |r - r'|). unit (angstrom^-2 * atomic_mass^-1/2). 
                        Spring_k term will be adjusted according to time step dt: spring_k * dt^2 = 0.25 (empirical choice.)
                        Therefore, we do not need this specify this value. The value will be used for restarting the algorithm.
                        """,
            },
        ),
        "kappa": (
            InputDictionary,
            {
                "dtype": float,
                "options": ["left", "right"],
                "default": [50, 50],
                "help": """the energy constraint term for beads at two ends to confine beads at iso-energy contour. 
                unit: (eV^(-1) angstrom^(-1) * atomic_mass^-1/2)
                See eq.(19) of THE JOURNAL OF CHEMICAL PHYSICS 148, 102334 (2018).
                We need two different values of kappa for asymmetric potential. 
                For symmetric potential, we can set left and right kappa to the same value.
                Two kappa values will be updated accordingly during the simulation using the force information |dV/dx|. 
                So, we do not need to specify this value.
                |dV/dx| * kappa / sqrt(m_H) * dt^2 = 0.5 (empirical value). The value will be used for restarting the algorithm. 
                        """,
            },
        ),

        "ENO_order": (
            InputValue,
            {
                "dtype": int,
                "default" : 3,
                "help": """
                order of essentially non-oscillatory method for approximating the tangent direction of the path.
                """
            }
        ),

        "end_bead_energy_converge_value":(
            InputValue,
            {
                "dtype": float,
                "default": 1e-4,
                "dimension": "energy",
                "help": """
                If energy of end beads is within the converge value around the instanton path energy, we then assume the end beads 
                is close to converge. This is used to set the kappa (energy constraint term) value.
            """
            }
        ),

        "dynamical_adjust_ratio": (
            InputDictionary,
            {
                "dtype": float,
                "options": ["spring_k", "kappa"],
                "default": [0.1, 0.2],
                "help": """
                The parameter that dynamically adjust the spring constant and energy constraint term (kappa) to make
                sure the MD converges.
                Adjust spring_k and kappa such that: 
                spring_k * (dt)^2 = spring_k_dynamical_adjust_ratio,
                kappa * (dt)^2 = kappa_dynamical_adjust_ratio.
                """
            }
        ),

        "final_hessian_bool": (
            InputValue,
            {
                "dtype": bool,
                "default": False,
                "help": "Bool variable. whether to compute final hessian when we get the instanton trajectory.",
            },
        ),
        "ab_initio_hessian_bool": (
            InputValue,
            {
                "dtype": bool,
                "default": False,
                "help": "Bool variable. if True, compute hessians of all beads ab initio. If false, use GPR to predict hessians.",
            },
        ),
        "Hessian_interpolation":(
            InputValue,
            {
                "dtype": str,
                "options": ["GPR", "CubicSpline"],
                "default": "GPR",
                "help": "Method used to interpolate Hessians of ring polymers along the instanton path."
            }
        ),

        "alt_out": (
            InputValue,
            {
                "dtype": int,
                "default": 5,
                "help": "output instanton bead energy and geometry every alt_out step",
            },
        ),
        "prefix": (
            InputValue,
            {"dtype": str, "default": "neb_instanton", "help": "prefix of output file"},
        ),
        "gpr_relative_force_error_criterion": (
            InputValue,
            {
                "dtype": float,
                "default": 0.05,
                "help": "convergence criterion for gpr outer loop. \
                |f^GPR - f|/|f| < gpr_relative_force_error_criterion means GPR prediction is reliable for force is reliable. \
                Stop the outer loop.",
            },
        ),
        "gpr_absolute_force_error_criterion": (
            InputValue,
            {
                "dtype": float,
                "default": 0.001,
                "help": """
                convergence criterion for gpr outer loop. 
                When |f^GPR - f| < gpr_absolute_force_error_criterion, 
                this means the GPR prediction is already reliable for that bead.
                This value should be larger or equal to the noise of force_noise_prior * sqrt(d) in the GPR model.
                here d is degrees of freedom in the model.
                """,
            },
        ),
        "gpr_absolute_potential_error_criterion":(
            InputValue,
            {
                "dtype": float,
                "default":1e-4,
                "dimension": "energy",
                "help":"""
                The error of potential prediction.
                For tunneling splitting calculation. The low energy region near the potential 
                minimum is approximated by the Taylor expansion.
                """
            }
        ),
        "gpr_force_uncertainty_criterion":(
            InputValue,
            {
                "dtype": float,
                "default": 0.001,
                "help": """
                convergence criterion for gpr outer loop.
                The std from GPR prediction is compared with the criterion.
                If all images' uncertainty is smaller than criterion, the algorithm converge.
                otherwise, the bead with large uncertainty is selected, its potential and force are computed,
                then it's used to update the model. 
                """
            }
        ),

        "gpr_trust_region": (
            InputValue,
            {
                "dtype": float,
                "default": 0.1,
                "help": "trust region for Gaussian Process Regression r_max = gpr_trust_region. \
                  If the distance between NEB beads and nearest GPR point exceed r_max, \
                  we stop the NEB inner loop and evaluate ab-initio force on that point.",
            },
        ),


        "minimum_trust_region": (
            InputValue,
            {
                "dtype": float,
                "default": 0.1,
                "help": """ The trust region will be adjusted when we find the current trust region is too 
                large to make the optimization algorithm unstable. The minimum trust region avoids we makes
                the trust region too small.
                 """
            }
        ),

        "distance_cutoff_for_training_data": (
            InputValue,
            {
                "dtype": float,
                "default": 0.1,
                "help": """ To avoid ill-conditioning of covariance matrix in the Gaussian Process Regression 
                model, we have to avoid adding data points too close to existing training data points in the 
                model. This distance cutoff will throw away data points too close to existing data. 
                """
            }
        ),

        "gpr_kernel_outputscale": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.ones, args=(0,)),
                "help": "Gaussian Process Regression hyperparameter. \
                  Each element corresponds to one Squared Exponential kernel we use to construct covariance function. \
                  Mean value for output scale prior of Gaussian process regression kernel. \
                  Typically it is the variance of potential energy",
            },
        ),

        "gpr_kernel_outputscale_constraint":(
            InputDictionary,
            {
                "dtype": float,
                "options" :[
                    "min",
                    "max"
                ],
                "default":[
                    0.0,
                    1.0
                ],
                "help": """constraint for the output scale of Gaussian Process regression kernel.
                output_scale = min + (max - min) * sigmoid(output_scale_unconstrained)"""
            }
        ),

        "gpr_kernel_lengthscale_ratio": (
            InputArray,
            {
                "dtype": float,
                "default": input_default(factory=np.ones, args=(0,)),
                "help": "Gaussian Process Regression hyperparameter. \
                  Each element corresponds to one Squared Exponential kernel we use to construct covariance function. \
                    Ratio of the mean value for the prior of the lengthscale of Gaussian Process regression model / length of bead in non-redundant internal coordinate. \
                      Typically this should be in the same order of the range of input data.",
                "dimension": "length",
            },
        ),

        "gpr_kernel_lengthscale_ratio_constraint":(
            InputDictionary,
            {
                "dtype": float,
                "options": [
                    "min",
                    "max"
                ],
                "default": [
                    0.5,
                    5.0
                ],
                "help": """constraint for the lengthscale of Gaussian Process regression kernel.
                lengthscale = min + (max - min) * sigmoid(lengthscale_unconstrained)"""
            }
        ),

        "gpr_noise_std": (
            InputDictionary,
            {
                "dtype": float,
                "options": [
                    "pot_noise_prior",
                    "force_noise_prior",
                    "hessian_noise_prior",
                ],
                "default": [1e-6, 1e-4, 1e-3],
                "help": "constraint for the variance of noise in the Gaussian Process Regression",
            },
        ),
        "gpr_SE_kernel_number": (
            InputValue,
            {
                "dtype": int,
                "default": 1,
                "help": "Number of Squared Exponential (SE) kernel used in Gaussian Process kernel.",
            },
        ),

        "gpr_fix_internal_dofs_bool":(
            InputValue,
            {
                "dtype": bool,
                "default": True,
                "help": """
                bool variable to decide whether we fix certain internal dofs.
                This is for the case that certain internal dofs of training data is fixed.
                The criterion is |q_max - q_min| < cutoff (given by gpr_fix_internal_dofs_cutoff)
            """
            }
        ),

        "gpr_fix_internal_dofs_cutoff":(
            InputValue,
            {
                "dtype": float,
                "default": 1e-3,
                "help": """
                cutoff value for fixing internal dofs.
            """
            }
        ),

        "gpr_rigid_internal_dofs_cutoff":(
            InputValue,
            {
                "dtype": float,
                "default": 1e-2,
                "help":
                """
                cutoff value for rigid internal dofs. We will use Linear Regression fitting for gradient along rigid internal dofs.
                """
            }
        ),

        "read_initial_gpr_training_data": (
            InputValue,
            {
                "dtype": bool,
                "default": False,
                "help": "Bool variable to decide whether to read the stored training data.",
            },
        ),

        "test_gpr_model_along_instanton_path":(
            InputValue,
            {
                "dtype": bool, 
                "default": False,
                "help": """
                bool variable to decide whether test the gpr model for data point along LINEB path.
                default is False, but can turn it on if you suspect the GPR model prediction along force is not accurate,
                which causes the temperature predicted to be wrong.
                """
            }
        ),

        "read_gpr_hessian_folder": (
            InputValue,
            {
                "dtype": str,
                "default": "None",
                "help": """
                Provide the name of folder. Read coordinate, potential, 
                gradient & hessians for the gpr_hessian model from a given folder. 
                The data can also be used for the cubic spline interpolation of Hessians. 
                """,
            },
        ),
        "add_new_hessian_data_bool": (
            InputValue,
            {
                "dtype": bool,
                "default": False,
                "help": "Bool variable to decide whether we will add new hessian training data in the training set.",
            },
        ),
        "candidate_hessian_data_number": (
            InputValue,
            {
                "dtype": int,
                "default": 20,
                "help": "number of ab initio hessian data we can potentially compute along the path. \
                    we can choose indices from these data points and use them to construct gpr_hessian model.",
            },
        ),
        "new_hessian_data_index": (
            InputArray,
            {
                "dtype": int,
                "default": input_default(factory=np.zeros, args=(0,)),
                "help": "The index for new data point which we will compute hessian. \
                  These hessian data will be added to the gpr_hessian model.",
            },
        ),

        "add_new_grad_data_bool":(
            InputValue,
            {
                "dtype": bool,
                "default": False,
                "help": "Bool variable to decide whether we will add new gradient training data in the training set."
            }
        ),
        
        "candidate_grad_data_number":(
            InputValue,
            {
                "dtype": int,
                "default": 100,
                "help": "number of ab initio gradient data we can potentially compute and add to the path. \
                    we can choose indices from these data points and use them to construct gpr_hessian model"
            },
        ),
        
        "new_grad_data_index": (
            InputArray,
            {
                "dtype": int,
                "default": input_default(factory= np.zeros, args=(0,)),
                "help": "The index for new data point which we will compute gradient. \
                    These gradient data will be added to the gpr_hessian model"
            }
        ),
        "train_grad_model_bool":(
            InputValue,
            {
                "dtype": bool,
                "default": True, 
                "help": """
                option to train GPR gradient model.
                Training GPR model can be expensive. If we are happy with model hyper-parameter,
                we can turn the training option off here.
                """
            }
        ),

        "train_hessian_model_bool": (
            InputValue,
            {
                "dtype": bool,
                "default": True,
                "help": """
                option to train GPR hessian model. 
                Training GPR model can be expensive when model size gets large. ~O(N^3).
                Because we have to compute inverse of the matrix & log determinant of the matrix 
                at each step of optimization. 
                It's not necessary to train GPR model each time we add more data, only do it 
                when the performance of gpr model degrades.
                """
            }
        ),
        "selective_hessian_bool":(
            InputValue,
            {
                "dtype": bool,
                "default": False,
                "help": """ Bool variable. if True, we will compute hessians in internal coordinate, and 
                save computational cost by only compute 1 hessian for components along rigid modes.
                If false, we compute hessians in Cartesian coordinate.
                """
            }
        ),
        "new_hessian_data_index_rigid_mode":(
            InputArray,
            {
                "dtype": int,
                "default": input_default(factory= np.zeros, args=(0,)),
                "help": """The index for new data point in which we will compute hessian along
                        rigid modes. 
                """
            }
        ),

        "internal_coord":(
            InputValue,
            {
                "dtype": str,
                "default": "bond",
                "options": ["bond", "Coulomb", "IRZ"],
                "help": """
                The option to construct primitive internal coordinate.
                We provide three choices: bond, Coulomb and IRZ. 
                For Coulomb choice, we build redundant internal coordinate as 1/|ri - rj|. 
                This is for hydrogen extract reaction, including CH4 + H.
                For bond choice, we build redundant internal coordinate by building connectivity graph 
                between atoms in the molecule, then add angle, distance & dihedral angle as redundant internal coordinate.
                This is for intra-molecular proton transfer reaction like malonaldehyde and aminopropenal.
                For IRZ, it stands for inverse radial Z matrix coordinates. It replace the bond length with inverse bond length
                in the redundant internal coordinate.
                """
            }
        ),
        "cross_validation_bool":(
            InputValue,
            {
                "dtype": bool,
                "default": False,
                "help":
                """
                The option to perform cross validation for the gpr model that predict hessians.
                To do so, we need to load the already computed hessian data.
                The number of hessian data should be >= 5 for cross validation to work.
                """
            }
        ),
        "ridge_regularization_alpha":(
            InputDictionary,
            {
                "dtype": float,
                "options": ["force", "hessian"],
                "default": [0.1, 0.5],
                "help":
                """
                The regularization amplitude for ridge (linear regression model) for fitting hessians
                along stiff dofs.
                """
            }
        ),
        "gpr_covar_inverse_nugget": (
            InputValue,
            {
                "dtype": float,
                "default": 1e-8,
                "help":
                """
                Nugget value add to the pseudo-inverse of covariance matrix in gpr model.
                See: https://arxiv.org/abs/1602.00853.
                """
            }
        )
    }

    dynamic = {}

    default_help = "A class for nudged elastic band to find minimum action path. Accelerated by Nudged Elastic band method."
    default_label = "NEB_INSTANTON_GPR"

    def store(self, geop):
        """
        this function corresponds to how we name variables in neb_instanton.py
        """
        if geop == {}:
            return

        options = geop.options
        optarrays = geop.optarrays

        # options
        self.mode.store(options["mode"])
        self.cal_type.store(options["cal_type"])
        self.opt.store(options["opt"])
        self.asr.store(options["asr"])
        self.stage.store(options["stage"])
        self.tolerances.store(options["tolerances"])
        self.alt_out.store(options["alt_out_step"])
        self.prefix.store(options["prefix"])
        self.final_hessian_bool.store(options["final_hessian_bool"])
        self.ab_initio_hessian_bool.store(options["ab_initio_hessian_bool"])
        self.Hessian_interpolation.store(options["Hessian_interpolation"])
        # options for GPR kernel
        self.gpr_SE_kernel_number.store(options["gpr_SE_kernel_number"])
        self.read_initial_gpr_training_data.store(
            options["read_initial_gpr_training_data"]
        )
        self.test_gpr_model_along_instanton_path.store(
            options["test_gpr_model_along_instanton_path"]
        )
        # about computing hessian for gpr model & rate calculation.
        self.read_gpr_hessian_folder.store(options["read_gpr_hessian_folder"])
        self.add_new_hessian_data_bool.store(options["add_new_hessian_data_bool"])
        self.candidate_hessian_data_number.store(
            options["candidate_hessian_data_number"]
        )

        self.add_new_grad_data_bool.store(options["add_new_grad_data_bool"])
        self.candidate_grad_data_number.store(
            options["candidate_grad_data_number"]
        )

        self.train_grad_model_bool.store(options["train_grad_model_bool"])
        self.train_hessian_model_bool.store(options["train_hessian_model_bool"])

        # for stability of gaussian process regression model
        self.minimum_trust_region.store(
            options["minimum_trust_region"]
        )

        self.distance_cutoff_for_training_data.store(
            options["distance_cutoff_for_training_data"]
        )

        self.gpr_fix_internal_dofs_bool.store(
            options["gpr_fix_internal_dofs_bool"]
        )

        self.gpr_fix_internal_dofs_cutoff.store(
            options["gpr_fix_internal_dofs_cutoff"]
        )

        self.gpr_rigid_internal_dofs_cutoff.store(
            options["gpr_rigid_internal_dofs_cutoff"]
        )

        self.selective_hessian_bool.store(
            options["selective_hessian_bool"]
        )

        self.internal_coord.store(
            options["internal_coord"]
        )

        self.cross_validation_bool.store(
            options["cross_validation_bool"]
        )

        # optarrays
        self.fix_dofs.store(optarrays["fix_dofs"])
        self.energy_shift.store(optarrays["energy_shift"])
        self.spring_k.store(optarrays["spring_k"])
        self.kappa.store(optarrays["kappa"])
        self.ENO_order.store(optarrays["ENO_order"])
        self.dynamical_adjust_ratio.store(optarrays["dynamical_adjust_ratio"])
        self.end_bead_energy_converge_value.store(optarrays["end_bead_energy_converge_value"])
        self.neb_inner_loop_step_max.store(optarrays["neb_inner_loop_step_max"])

        self.time_step.store(optarrays["time_step"])
        self.cg_big_step.store(optarrays["cg_big_step"])
        self.instanton_time_step.store(optarrays["instanton_time_step"])

        self.instanton_path_energy.store(optarrays["instanton_path_energy"])
        self.instanton_bead_number.store(optarrays["instanton_bead_number"])

        # store parameters about gaussian process regression
        self.gpr_relative_force_error_criterion.store(
            optarrays["gpr_relative_force_error_criterion"]
        )
        self.gpr_absolute_force_error_criterion.store(
            optarrays["gpr_absolute_force_error_criterion"]
        )
        self.gpr_absolute_potential_error_criterion.store(
            optarrays["gpr_absolute_potential_error_criterion"]
        )
        self.gpr_force_uncertainty_criterion.store(
            optarrays["gpr_force_uncertainty_criterion"]
        )
        self.gpr_trust_region.store(optarrays["gpr_trust_region"])
        self.gpr_kernel_outputscale.store(optarrays["gpr_kernel_outputscale"])
        self.gpr_kernel_outputscale_constraint.store(
            optarrays["gpr_kernel_outputscale_constraint"]
        )
        self.gpr_kernel_lengthscale_ratio.store(
            optarrays["gpr_kernel_lengthscale_ratio"]
        )
        self.gpr_kernel_lengthscale_ratio_constraint.store(
            optarrays["gpr_kernel_lengthscale_ratio_constraint"]
        )
        self.gpr_noise_std.store(optarrays["gpr_noise_std"])

        # store result of instanton calculation
        self.instanton_temperature.store(optarrays["instanton_temperature"])
        self.instanton_bead_q.store(optarrays["instanton_bead_q"])
        self.instanton_bead_pot.store(optarrays["instanton_bead_pot"])
        self.instanton_hessian.store(optarrays["instanton_hessian"])

        # about computing hessian for gpr model & rate calculation.
        self.new_hessian_data_index.store(optarrays["new_hessian_data_index"])
        self.new_grad_data_index.store(optarrays["new_grad_data_index"])
        self.new_hessian_data_index_rigid_mode.store(optarrays["new_hessian_data_index_rigid_mode"])
        
        # regularization factor for ridge model : modeling hessians along stiff modes.
        self.ridge_regularization_alpha.store(optarrays["ridge_regularization_alpha"])
        self.gpr_covar_inverse_nugget.store(optarrays["gpr_covar_inverse_nugget"])
        
    def fetch(self):
        rv = super(InputNebInstGPR, self).fetch()
        rv["mode"] = self.mode.fetch()
        rv["cal_type"] = self.cal_type.fetch()
        return rv
