from .internalcoord import DelocalizedInternalCoordinates
import numpy as np
import ipi.utils.internal.molecule 

class non_redundant_coordinate_transformer:
    """
    perform transformation between Cartesian coordinate and internal coordinate.
    """
    def __init__(self, 
                 natoms: int,
                 ref_x_list: np.ndarray,
                 elem: list = None, 
                 ):
        """
        :param: natoms: number of atoms in molecule
        :param: ref_x_list: a list of reference coordinate x, which we will use to create the topology of molecule.
                By default, we choose the first ref_x in ref_x_list as the coordinate of the created molecule object.
        :param: elem: elements of atoms in molecules: in ipi, this is self.beads.names.
        """
        self.natom = natoms
        self.ref_x = ref_x_list[0]

        if np.size(self.ref_x) != 3 * natoms:
            raise (
                "The size of reference point for initializing non redundant coordinate is not 3 * natom: size of ref_x: {} , natom: {}".format(
                    np.size(self.ref_x), natoms
                )
            )

        # create molecule object
        molecule = ipi.utils.internal.molecule.create_molecule(
            natoms,
            elem,
            ref_x_list,
            molecule_index= 0  # we choose the first ref_x as the coordinate of the newly created molecule object.
        )

        # create delocalized internal coordinate object
        self.dlc_coord = DelocalizedInternalCoordinates(molecule, connect= True, addcart= False)

        # record unitary matrix U and singular value matrix S for the reference point.
        self.ref_U = self.dlc_coord.ref_U 
        self.ref_UT = self.ref_U.T 
        self.ref_S = self.dlc_coord.S
        self.ref_Vh = self.dlc_coord.ref_Vh  
        self.nonzero_S_index_len = len(self.ref_S)
        print(f"Number of nonzero dofs: {self.nonzero_S_index_len}")
    
    def compute_delocalized_wilson_matrix_Bq(self, x):
        """
        compute how changes in Cartesian coordinate x will affect delocalized non-redundant coordinate q.
        Bq == \\ partial q / \\partial x.
        :param: x: Cartesian coordinate. size: [nbatch, 3 * natom]

        :return: matrix Bq: delocalized non-redundant Wilson B matrix.  size: [nbatch, 3 * natom - 6, 3 * natom]
        """
        nbatch = np.shape(x)[0]
        Bq = []
        for i in range(nbatch):
            coord = x[i]
            ders = self.dlc_coord.derivatives(coord).reshape((-1, 3 * self.natom))
            Bq.append(ders)
        
        Bq = np.array(Bq)

        return Bq 
    
    def get_internal_coordinate_q(self, x):
        """
        transform Cartesian coordinate x to internal coordinate q.
        :param: x: Cartesian coordinate. shape: [nbatch, 3 * natom]
        
        :return: q: internal coordinate. shape: [nbatch, # of internal dof]
        """
        nbatch = np.shape(x)[0]
        q = [] 
        for i in range(nbatch):
            coord = x[i]
            dlc_q = self.dlc_coord.calculate(coord)
            q.append(dlc_q)
        
        q = np.array(q)

        return q 

    # transformation between gradients and hessian. g_x <-> g_q.  h_x <-> h_q
    # g_x -> g_q
    def transform_cartesian_gradient_to_internal_gradient(self, x_list, g_x_list):
        """
        transform from Cartesian coordinate system's gradient g_x to internal coordinate gradient g_q.
        :param: x_list: Cartesian coordinate. shape: [nbatch, 3 * natom]
        :param: g_x_list: gradient in Cartesian coordinate. shape:[nbatch, 3 * natom]

        :return g_q_list: gradient in internal coordinate. shape:[nbatch, 3 * natom - 6]
        """
        g_q_list = []
        for (x, g_x) in zip(x_list, g_x_list):
            g_q = self.dlc_coord.calcGrad(x, g_x)
            g_q_list.append(g_q)
        
        g_q_list = np.array(g_q_list)

        return g_q_list 
    
    # g_q - > g_x.
    def transform_internal_gradient_to_cartesian_gradient(self, x_list, g_q_list):
        """
        transform from internal coordinate's gradient g_q to cartesian coordinate's gradient g_x. 
        :param: x_list: Cartesian coordinate. size: [nbatch, 3 * natom]
        :param: g_q_list: gradient in nonredundant internal coordinate. shape: [nbatch, internal_dof_#]

        :return: g_x_list: gradient in cartesian coordinate. shape: [nbatch, 3 * natom]
        """
        g_x_list = []
        for (x, g_q) in zip(x_list, g_q_list):
            g_x = self.dlc_coord.calcGradCart(x, g_q)
            g_x_list.append(g_x)
        
        g_x_list = np.array(g_x_list)

        return g_x_list 
    
    # H_x -> H_q. transform hessian.
    def transform_cartesian_hessian_to_internal_hessian(self, x_list, g_x_list, H_x_list):
        """
        transform from Cartesian hessian H to internal dofs hessian H.
        In this case, we need hessian in all dofs.

        :param: x_list: Cartesian coordinates. shape: [nbatch, 3 * natom]
        :param: g_x: gradient in Cartesian coordinates. shape:[nbatch, 3 * natom]
        :param: H_x: hessian in Cartesian coordinates.  shape:[nbatch, 3 * natom, 3 * natom]

        :return: H_q: (only if hessian_bool = True). hessian in internal coordinate. shape:[nbatch, 3 * natom - 6, 3 * natom - 6]
        """
        nbatch = np.shape(x_list)[0]
        assert (
            nbatch == np.shape(g_x_list)[0]
        ), "the number of data points with gradients for internal coordinate transform is wrong."
        assert (
            nbatch == np.shape(H_x_list)[0]
        ), "the number of data points with hessians for internal coordinate transform is wrong."

        Hq_list = []
        for (x, g_x, H_x) in zip(x_list, g_x_list, H_x_list):
            Hq = self.dlc_coord.calcHess(x, g_x, H_x)
            Hq_list.append(Hq)
        
        Hq_list = np.array(Hq_list)

        return Hq_list 
    
    # H_q -> H_x. transform hessian.
    def transform_internal_hessian_to_cartesian_hessian(self, x_list, g_q_list, H_q_list):
        """
        transform from internal coordinate's hessian H to cartesian coordinate hessian H.
        In this case, we need hessian in all dofs.
        :param: x_list: Cartesian coordinate. size: [nbatch, 3 * natom]
        :param: g_q_list: gradients in non-redundant internal coordinates. shape: [nbatch, internal_dof_#]
        :param: H_q_list: Hessians in non-redundant internal coordinates. shape: [nbatch, internal_dof_#, internal_dof_#]

        :return: H_x_list: Hessian in Cartesian dofs.
        """
        nbatch = np.shape(x_list)[0]
        assert (
            nbatch == np.shape(g_q_list)[0]
        ), "the number of data points with gradients for internal coordinate transform is wrong."
        assert (
            nbatch == np.shape(H_q_list)[0]
        ), "the number of data points with hessians for internal coordinate transform is wrong."

        Hx_list = []
        for (x, g_q, H_q) in zip(x_list, g_q_list, H_q_list):
            Hx = self.dlc_coord.calcHessCart(x, g_q, H_q)
            Hx_list.append(Hx)
        
        Hx_list = np.array(Hx_list)

        return Hx_list 
    
    def compute_x_hessian_q(self, x_list):
        """
        compute the hessian of Cartesian coordinate x for the reference point with respect to internal coordinate q.
        d^2 x/ dq^2
        
        :param: x_list: Cartesian coordinate. size: [nbatch, 3 * natom]
        """
        hessian_x_qq_list = []
        for x in x_list:
            hessian_x_qq = self.dlc_coord.inverse_second_derivatives(x)
            hessian_x_qq_list.append(hessian_x_qq)
        
        hessian_x_qq_list = np.array(hessian_x_qq_list)

        return hessian_x_qq_list 
    