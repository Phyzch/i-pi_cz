"""
Modified from geomeTRIC internal.py file. Copyright info below:
internal.py: Internal coordinate systems

Copyright 2016-2020 Regents of the University of California and the Authors

Authors: Lee-Ping Wang, Chenchen Song

Contributors: 

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import itertools
import time, sys
from collections import OrderedDict, defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np
from numpy.linalg import multi_dot
from ipi.utils.messages import warning, info 

from internal.nifty import click, commadash, ang2bohr, bohr2ang, logger, pvec1d, pmat2d
from internal.molecule import Molecule, PeriodicTable, Elements, Radii 

## Some vector calculus functions
def unit_vector(a):
    """
    Vector function: Given a vector a, return the unit vector
    """
    return a / np.linalg.norm(a)

def d_unit_vector(a, ndim=3):
    term1 = np.eye(ndim)/np.linalg.norm(a)
    term2 = np.outer(a, a)/(np.linalg.norm(a)**3)
    answer = term1-term2
    return answer

def d_cross(a, b):
    """
    Given two vectors a and b, return the gradient of the cross product axb w/r.t. a.
    (Note that the answer is independent of a.)
    Derivative is on the first axis.
    """
    d_cross = np.zeros((3, 3), dtype=float)
    for i in range(3):
        ei = np.zeros(3, dtype=float)
        ei[i] = 1.0
        d_cross[i] = np.cross(ei, b)
    return d_cross

def d_cross_ab(a, b, da, db):
    """
    Given two vectors a, b and their derivatives w/r.t. a parameter, return the derivative
    of the cross product
    """
    answer = np.zeros((da.shape[0], 3), dtype=float)
    for i in range(da.shape[0]):
        answer[i] = np.cross(a, db[i]) + np.cross(da[i], b)
    return answer

def ncross(a, b):
    """
    Scalar function: Given vectors a and b, return the norm of the cross product
    """
    cross = np.cross(a, b)
    return np.linalg.norm(cross)

def d_ncross(a, b):
    """
    Return the gradient of the norm of the cross product w/r.t. a
    """
    ncross = np.linalg.norm(np.cross(a, b))
    term1 = a * np.dot(b, b)
    term2 = -b * np.dot(a, b)
    answer = (term1+term2)/ncross
    return answer

def nudot(a, b):
    r"""
    Given two vectors a and b, return the dot product (\hat{a}).b.
    """
    ev = a / np.linalg.norm(a)
    return np.dot(ev, b)
    
def d_nudot(a, b):
    r"""
    Given two vectors a and b, return the gradient of 
    the norm of the dot product (\hat{a}).b w/r.t. a.
    """
    return np.dot(d_unit_vector(a), b)

def ucross(a, b):
    r"""
    Given two vectors a and b, return the cross product (\hat{a})xb.
    """
    ev = a / np.linalg.norm(a)
    return np.cross(ev, b)
    
def d_ucross(a, b):
    r"""
    Given two vectors a and b, return the gradient of 
    the cross product (\hat{a})xb w/r.t. a.
    """
    ev = a / np.linalg.norm(a)
    return np.dot(d_unit_vector(a), d_cross(ev, b))

def nucross(a, b):
    r"""
    Given two vectors a and b, return the norm of the cross product (\hat{a})xb.
    """
    ev = a / np.linalg.norm(a)
    return np.linalg.norm(np.cross(ev, b))
    
def d_nucross(a, b):
    r"""
    Given two vectors a and b, return the gradient of 
    the norm of the cross product (\hat{a})xb w/r.t. a.
    """
    ev = a / np.linalg.norm(a)
    return np.dot(d_unit_vector(a), d_ncross(ev, b))
## End vector calculus functions

# List of Primitive Internal Coordinate.
class PrimitiveCoordinate(object):
    """
    Parent class for primitive internal coordinate objects with common methods.
    """
    def calcDiff(self, xyz1, xyz2=None, val2=None):
        """
        Return the difference of the internal coordinate
        calculated for c(xyz1) - c(xyz2) or c(xyz1) - val2.

        Parameters
        ----------
        xyz1 : np.ndarray
            xyz coordinates of first structure in Bohr
        xyz2 : np.ndarray
            If provided, xyz coordinates of second structure in Bohr
        val2 : float
            If provided, this is the value to subtract
        """
        if xyz2 is None and val2 is None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        elif xyz2 is not None and val2 is not None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        if xyz2 is not None:
            val2 = self.value(xyz2)
        diff = self.value(xyz1) - val2
        if hasattr(self, 'w'):
            w = self.w
        else:
            w = 1.0
        # Divide by the weight, if exists, to get the "base" number
        diff /= w
        # Subtract out any differences of 2*pi for periodic degrees of freedom
        # (rotation ICs handled separately)
        if hasattr(self, 'isPeriodic') and self.isPeriodic:
            Plus2Pi = diff + 2*np.pi
            Minus2Pi = diff - 2*np.pi
            if np.abs(diff) > np.abs(Plus2Pi):
                diff = Plus2Pi
            if np.abs(diff) > np.abs(Minus2Pi):
                diff = Minus2Pi
        diff *= w
        return diff

class CartesianX(PrimitiveCoordinate):
    def __init__(self, a, w=1.0):
        self.a = a   # atoms index.
        self.w = w   # weights
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        #return "Cartesian-X %i : Weight %.3f" % (self.a+1, self.w)
        return "Cartesian-X %i" % (self.a+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = self.a == other.a
        if eq and self.w != other.w:
            logger.warning("Warning: CartesianX same atoms, different weights (%.4f %.4f)\n" % (self.w, other.w))
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        return xyz[a][0]*self.w

    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        derivatives[self.a][0] = self.w
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2
    
class CartesianY(PrimitiveCoordinate):
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Cartesian-Y %i : Weight %.3f" % (self.a+1, self.w)
        return "Cartesian-Y %i" % (self.a+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = self.a == other.a
        if eq and self.w != other.w:
            logger.warning("Warning: CartesianY same atoms, different weights (%.4f %.4f)\n" % (self.w, other.w))
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        return xyz[a][1]*self.w
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        derivatives[self.a][1] = self.w
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class CartesianZ(PrimitiveCoordinate):
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Cartesian-Z %i : Weight %.3f" % (self.a+1, self.w)
        return "Cartesian-Z %i" % (self.a+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = self.a == other.a
        if eq and self.w != other.w:
            logger.warning("Warning: CartesianZ same atoms, different weights (%.4f %.4f)\n" % (self.w, other.w))
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        return xyz[a][2]*self.w

    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        derivatives[self.a][2] = self.w
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class TranslationX(PrimitiveCoordinate):
    def __init__(self, a, w):
        self.a = a
        self.w = w
        assert len(a) == len(w)
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Translation-X %s : Weights %s" % (' '.join([str(i+1) for i in self.a]), ' '.join(['%.2e' % i for i in self.w]))
        return "Translation-X %s" % (commadash(self.a))
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.w-other.w)**2) > 1e-6:
            logger.warning("Warning: TranslationX same atoms, different weights\n")
            eq = False
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,0]*self.w)

    def calcDiff(self, xyz1, xyz2=None, val2=None):
        # Translation ICs require an explicit implementation of calcDiff
        # because self.w is not a float but an array
        if xyz2 is None and val2 is None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        elif xyz2 is not None and val2 is not None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        if xyz2 is not None:
            val2 = self.value(xyz2)
        diff = self.value(xyz1) - val2
        return diff

    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        for i, a in enumerate(self.a):
            derivatives[a][0] = self.w[i]
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2
    
class TranslationY(object):
    def __init__(self, a, w):
        self.a = a
        self.w = w
        assert len(a) == len(w)
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Translation-Y %s : Weights %s" % (' '.join([str(i+1) for i in self.a]), ' '.join(['%.2e' % i for i in self.w]))
        return "Translation-Y %s" % (commadash(self.a))
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.w-other.w)**2) > 1e-6:
            logger.warning("Warning: TranslationY same atoms, different weights\n")
            eq = False
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,1]*self.w)

    def calcDiff(self, xyz1, xyz2=None, val2=None):
        # Translation ICs require an explicit implementation of calcDiff
        # because self.w is not a float but an array
        if xyz2 is None and val2 is None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        elif xyz2 is not None and val2 is not None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        if xyz2 is not None:
            val2 = self.value(xyz2)
        diff = self.value(xyz1) - val2
        return diff
    
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        for i, a in enumerate(self.a):
            derivatives[a][1] = self.w[i]
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class TranslationZ(PrimitiveCoordinate):
    def __init__(self, a, w):
        self.a = a
        self.w = w
        assert len(a) == len(w)
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Translation-Z %s : Weights %s" % (' '.join([str(i+1) for i in self.a]), ' '.join(['%.2e' % i for i in self.w]))
        return "Translation-Z %s" % (commadash(self.a))
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.w-other.w)**2) > 1e-6:
            logger.warning("Warning: TranslationZ same atoms, different weights\n")
            eq = False
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,2]*self.w)
        
    def calcDiff(self, xyz1, xyz2=None, val2=None):
        # Translation ICs require an explicit implementation of calcDiff
        # because self.w is not a float but an array
        if xyz2 is None and val2 is None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        elif xyz2 is not None and val2 is not None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        if xyz2 is not None:
            val2 = self.value(xyz2)
        diff = self.value(xyz1) - val2
        return diff
    
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        for i, a in enumerate(self.a):
            derivatives[a][2] = self.w[i]
        return derivatives
    
    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class CentroidDistance(PrimitiveCoordinate):
    def __init__(self, a, b):
        # a & b is index for atoms.
        self.a = a
        self.b = b
        self.isAngular = False
        self.isPeriodic = False
        self.xa = TranslationX(a, w=np.ones(len(a))/len(a))
        self.ya = TranslationY(a, w=np.ones(len(a))/len(a))
        self.za = TranslationZ(a, w=np.ones(len(a))/len(a))
        self.xb = TranslationX(b, w=np.ones(len(b))/len(b))
        self.yb = TranslationY(b, w=np.ones(len(b))/len(b))
        self.zb = TranslationZ(b, w=np.ones(len(b))/len(b))

    def __repr__(self):
        return "CentroidDistance %s --- %s" % (commadash(self.a), commadash(self.b))
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = (set(self.a) == set(other.a) and set(self.b) == set(other.b))
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        (xa, ya, za, xb, yb, zb) = (p.value(xyz) for p in (self.xa, self.ya, self.za,
                                                           self.xb, self.yb, self.zb))
        ra = np.array([xa, ya, za])
        rb = np.array([xb, yb, zb])
        rab = (rb-ra)
        nab = np.linalg.norm(rab)
        return nab
        
    def calcDiff(self, xyz1, xyz2=None, val2=None):
        if xyz2 is None and val2 is None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        elif xyz2 is not None and val2 is not None:
            raise RuntimeError("Provide exactly one of xyz2 and val2")
        if xyz2 is not None:
            val2 = self.value(xyz2)
        diff = self.value(xyz1) - val2
        return diff
    
    def derivative(self, xyz):
        (xa, ya, za, xb, yb, zb) = (p.value(xyz) for p in (self.xa, self.ya, self.za,
                                                           self.xb, self.yb, self.zb))
        ra = np.array([xa, ya, za])
        rb = np.array([xb, yb, zb])
        rab = (rb-ra)
        nab = np.linalg.norm(rab)
        
        dxa = self.xa.derivative(xyz)
        dya = self.ya.derivative(xyz)
        dza = self.za.derivative(xyz)
        dxb = self.xb.derivative(xyz)
        dyb = self.yb.derivative(xyz)
        dzb = self.zb.derivative(xyz)
        
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        derivatives += rab[0]*dxb
        derivatives -= rab[0]*dxa
        derivatives += rab[1]*dyb
        derivatives -= rab[1]*dya
        derivatives += rab[2]*dzb
        derivatives -= rab[2]*dza
        derivatives /= nab
        return derivatives
        # for i, a in enumerate(self.a):
        #     derivatives[a][2] = self.w[i]
        # return derivatives

    def second_derivative(self, xyz):
        (xa, ya, za, xb, yb, zb) = (p.value(xyz) for p in (self.xa, self.ya, self.za,
                                                           self.xb, self.yb, self.zb))
        ra = np.array([xa, ya, za])
        rb = np.array([xb, yb, zb])
        rab = (rb-ra)
        nab = np.linalg.norm(rab)
        
        # Finite difference for now - I don't expect we'll be using this anyway,
        # since second derivs. are usually for TS optimization and CentroidDistance
        # is used for constraints
        h = 1e-4 

        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        for i in range(xyz.shape[0]):
            for j in range(xyz.shape[1]):
                xyz[i, j] += h
                dplus = self.derivative(xyz)
                xyz[i, j] -= 2*h
                dminus = self.derivative(xyz)
                xyz[i, j] += h
                deriv2[i, j, :, :] = (dplus-dminus)/(2*h)

        return deriv2
    
class Distance(PrimitiveCoordinate):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        if a == b:
            raise RuntimeError('a and b must be different')
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        return "Distance %i-%i" % (self.a+1, self.b+1)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.a == other.a:
            if self.b == other.b:
                return True
        if self.a == other.b:
            if self.b == other.a:
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        return np.sqrt(np.sum((xyz[a]-xyz[b])**2))
    
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = self.a
        n = self.b
        u = (xyz[m] - xyz[n]) / np.linalg.norm(xyz[m] - xyz[n])
        derivatives[m, :] = u
        derivatives[n, :] = -u
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        m = self.a
        n = self.b
        l = np.linalg.norm(xyz[m] - xyz[n])
        u = (xyz[m] - xyz[n]) / l
        mtx = (np.outer(u, u) - np.eye(3))/l
        deriv2[m, :, m, :] = -mtx
        deriv2[n, :, n, :] = -mtx
        deriv2[m, :, n, :] = mtx
        deriv2[n, :, m, :] = mtx
        return deriv2
    
class Angle(PrimitiveCoordinate):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.isAngular = True
        self.isPeriodic = False
        if len({a, b, c}) != 3:
            raise RuntimeError('a, b, and c must be different')

    def __repr__(self):
        return "Angle %i-%i-%i" % (self.a+1, self.b+1, self.c+1)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.b == other.b:
            if self.a == other.a:
                if self.c == other.c:
                    return True
            if self.a == other.c:
                if self.c == other.a:
                    return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        # vector from first atom to central atom
        vector1 = xyz[a] - xyz[b]
        # vector from last atom to central atom
        vector2 = xyz[c] - xyz[b]
        # norm of the two vectors
        norm1 = np.sqrt(np.sum(vector1**2))
        norm2 = np.sqrt(np.sum(vector2**2))
        dot = np.dot(vector1, vector2)
        # Catch the edge case that very rarely this number is -1.
        if dot / (norm1 * norm2) <= -1.0:
            if (np.abs(dot / (norm1 * norm2)) + 1.0) < -1e-6:
                raise RuntimeError('Encountered invalid value in angle')
            return np.pi
        if dot / (norm1 * norm2) >= 1.0:
            if (np.abs(dot / (norm1 * norm2)) - 1.0) > 1e-6:
                raise RuntimeError('Encountered invalid value in angle')
            return 0.0
        return np.arccos(dot / (norm1 * norm2))

    def normal_vector(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        # vector from first atom to central atom
        vector1 = xyz[a] - xyz[b]
        # vector from last atom to central atom
        vector2 = xyz[c] - xyz[b]
        # norm of the two vectors
        norm1 = np.sqrt(np.sum(vector1**2))
        norm2 = np.sqrt(np.sum(vector2**2))
        crs = np.cross(vector1, vector2)
        crs /= np.linalg.norm(crs)
        return crs
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = self.a
        o = self.b
        n = self.c
        # Unit displacement vectors
        u_prime = (xyz[m] - xyz[o])
        u_norm = np.linalg.norm(u_prime)
        v_prime = (xyz[n] - xyz[o])
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        v = v_prime / v_norm
        VECTOR1 = np.array([1, -1, 1]) / np.sqrt(3)
        VECTOR2 = np.array([-1, 1, 1]) / np.sqrt(3)
        if np.linalg.norm(u + v) < 1e-10 or np.linalg.norm(u - v) < 1e-10:
            # if they're parallel
            if ((np.linalg.norm(u + VECTOR1) < 1e-10) or
                    (np.linalg.norm(u - VECTOR2) < 1e-10)):
                # and they're parallel o [1, -1, 1]
                w_prime = np.cross(u, VECTOR2)
            else:
                w_prime = np.cross(u, VECTOR1)
        else:
            w_prime = np.cross(u, v)
        w = w_prime / np.linalg.norm(w_prime)
        term1 = np.cross(u, w) / u_norm
        term2 = np.cross(w, v) / v_norm
        derivatives[m, :] = term1
        derivatives[n, :] = term2
        derivatives[o, :] = -(term1 + term2)
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        m = self.a
        o = self.b
        n = self.c
        # Unit displacement vectors
        u_prime = (xyz[m] - xyz[o])
        u_norm = np.linalg.norm(u_prime)
        v_prime = (xyz[n] - xyz[o])
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        v = v_prime / v_norm
        # Deriv2 derivatives are set to zero in the case of parallel or antiparallel vectors
        if np.linalg.norm(u + v) < 1e-10 or np.linalg.norm(u - v) < 1e-10:
            return deriv2
        # cosine and sine of the bond angle
        cq = np.dot(u, v)
        sq = np.sqrt(1-cq**2)
        uu = np.outer(u, u)
        uv = np.outer(u, v)
        vv = np.outer(v, v)
        de = np.eye(3)
        term1 = (uv + uv.T - (3*uu - de)*cq)/(u_norm**2*sq)
        term2 = (uv + uv.T - (3*vv - de)*cq)/(v_norm**2*sq)
        term3 = (uu + vv - uv*cq   - de)/(u_norm*v_norm*sq)
        term4 = (uu + vv - uv.T*cq - de)/(u_norm*v_norm*sq)
        der1 = self.derivative(xyz)
        def zeta(a_, m_, n_):
            return (int(a_==m_) - int(a_==n_))
        for a in [m, n, o]:
            for b in [m, n, o]:
                deriv2[a, :, b, :] = (zeta(a, m, o)*zeta(b, m, o)*term1
                                      + zeta(a, n, o)*zeta(b, n, o)*term2
                                      + zeta(a, m, o)*zeta(b, n, o)*term3
                                      + zeta(a, n, o)*zeta(b, m, o)*term4
                                      - (cq/sq) * np.outer(der1[a], der1[b]))
        return deriv2

class LinearAngle(PrimitiveCoordinate):
    def __init__(self, a, b, c, axis):
        self.a = a
        self.b = b
        self.c = c
        self.axis = axis  # which linear bend angle to return. See value()
        self.isAngular = False
        self.isPeriodic = False
        if len({a, b, c}) != 3:
            raise RuntimeError('a, b, and c must be different')
        self.e0 = None
        self.stored_dot2 = 0.0

    def reset(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        # Unit vector pointing from a to c.
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        # Cartesian axes.
        ex = np.array([1.0,0.0,0.0])
        ey = np.array([0.0,1.0,0.0])
        ez = np.array([0.0,0.0,1.0])
        self.e0 = [ex, ey, ez][np.argmin([np.dot(i, ev)**2 for i in [ex, ey, ez]])]  # the axis that is most likely perpendicular to ev.
        self.stored_dot2 = 0.0

    def reposition_e0(self, xyz):
        """
        Project out the component of e0 that is parallel to ev. 
        This prevents linear angles from becoming parallel to e0, 
        which requires resetting the coordinate system.
        This function should be called at the end of each accepted step.
        """
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        dot = np.dot(ev, self.e0)
        self.e0 -= dot*ev
        self.e0 /= np.linalg.norm(self.e0)

    def __repr__(self):
        return "LinearAngle%s %i-%i-%i" % (["X","Y"][self.axis], self.a+1, self.b+1, self.c+1)

    def __eq__(self, other):
        if not hasattr(other, 'axis'): return False
        if self.axis is not other.axis: return False
        if type(self) is not type(other): return False
        if self.b == other.b:
            if self.a == other.a:
                if self.c == other.c:
                    return True
            if self.a == other.c:
                if self.c == other.a:
                    return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def value(self, xyz):
        """
        This function measures the displacement of the BA and BC unit
        vectors in the linear angle "ABC". The displacements are measured
        along two axes that are perpendicular to the AC unit vector.
        """
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        # Unit vector pointing from a to c.
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        if self.e0 is None: self.reset(xyz)
        e0 = self.e0
        self.stored_dot2 = np.dot(ev, e0)**2
        # Now make two unit vectors that are perpendicular to this one.
        c1 = np.cross(ev, e0)
        e1 = c1 / np.linalg.norm(c1)
        c2 = np.cross(ev, e1)
        e2 = c2 / np.linalg.norm(c2)
        # BA and BC unit vectors in ABC angle
        vba = xyz[a]-xyz[b]
        eba = vba / np.linalg.norm(vba)
        vbc = xyz[c]-xyz[b]
        ebc = vbc / np.linalg.norm(vbc)
        if self.axis == 0:
            answer = np.dot(eba, e1) + np.dot(ebc, e1)
        else:
            answer = np.dot(eba, e2) + np.dot(ebc, e2)
        return answer

    def visualize(self, xyz):
        xyz = xyz.reshape(-1,3)
        xsel = xyz[[self.a, self.b, self.c], :]
        xmean = np.mean(xsel,axis=0)
        a = self.a
        b = self.b
        c = self.c
        # Unit vector pointing from a to c.
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        if self.e0 is None: self.reset(xyz)
        e0 = self.e0
        self.stored_dot2 = np.dot(ev, e0)**2
        # Now make two unit vectors that are perpendicular to this one.
        c1 = np.cross(ev, e0)
        e1 = c1 / np.linalg.norm(c1)
        c2 = np.cross(ev, e1)
        e2 = c2 / np.linalg.norm(c2)
        # Visualize the rotated unit vectors.
        answer = np.zeros((3, 3), dtype=float)
        answer[0, :] = xmean
        answer[1, :] = xmean + e1*ang2bohr
        answer[2, :] = xmean + e2*ang2bohr
        return answer

    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        derivatives = np.zeros_like(xyz)
        ## Finite difference derivatives
        ## fderivatives = np.zeros_like(xyz)
        ## h = 1e-6
        ## for u in range(xyz.shape[0]):
        ##     for v in range(3):
        ##         xyz[u, v] += h
        ##         vPlus = self.value(xyz)
        ##         xyz[u, v] -= 2*h
        ##         vMinus = self.value(xyz)
        ##         xyz[u, v] += h
        ##         fderivatives[u, v] = (vPlus-vMinus)/(2*h)
        # Unit vector pointing from a to c.
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        if self.e0 is None: self.reset(xyz)
        e0 = self.e0
        c1 = np.cross(ev, e0)
        e1 = c1 / np.linalg.norm(c1)
        c2 = np.cross(ev, e1)
        e2 = c2 / np.linalg.norm(c2)
        # BA and BC unit vectors in ABC angle
        vba = xyz[a]-xyz[b]
        eba = vba / np.linalg.norm(vba)
        vbc = xyz[c]-xyz[b]
        ebc = vbc / np.linalg.norm(vbc)
        # Derivative terms
        de0 = np.zeros((3, 3), dtype=float)
        dev = d_unit_vector(v)
        dc1 = d_cross_ab(ev, e0, dev, de0)
        de1 = np.dot(dc1, d_unit_vector(c1))
        dc2 = d_cross_ab(ev, e1, dev, de1)
        de2 = np.dot(dc2, d_unit_vector(c2))
        deba = d_unit_vector(vba)
        debc = d_unit_vector(vbc)
        if self.axis == 0:
            derivatives[a, :] = np.dot(deba, e1) + np.dot(-de1, eba) + np.dot(-de1, ebc)
            derivatives[b, :] = np.dot(-deba, e1) + np.dot(-debc, e1)
            derivatives[c, :] = np.dot(de1, eba) + np.dot(de1, ebc) + np.dot(debc, e1)
        else:
            derivatives[a, :] = np.dot(deba, e2) + np.dot(-de2, eba) + np.dot(-de2, ebc)
            derivatives[b, :] = np.dot(-deba, e2) + np.dot(-debc, e2)
            derivatives[c, :] = np.dot(de2, eba) + np.dot(de2, ebc) + np.dot(debc, e2)
        ## Finite difference derivatives
        ## if np.linalg.norm(derivatives - fderivatives) > 1e-6:
        ##     print np.linalg.norm(derivatives - fderivatives)
        ##     raise Exception()
        return derivatives
    
    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        deriv2 = np.zeros((xyz.shape[0], 3, xyz.shape[0], 3), dtype=float)
        h = 1.0e-3
        for i in range(3):
            for j in range(3):
                ii = [a, b, c][i]
                xyz[ii, j] += h
                FPlus = self.derivative(xyz)
                xyz[ii, j] -= 2*h
                FMinus = self.derivative(xyz)
                xyz[ii, j] += h
                fderiv = (FPlus-FMinus)/(2*h)
                deriv2[ii, j, :, :] = fderiv
        return deriv2
    
class MultiAngle(PrimitiveCoordinate): # pragma: no cover
    def __init__(self, a, b, c):
        if type(a) is int:
            a = (a,)
        if type(c) is int:
            c = (c,)
        self.a = tuple(a)
        self.b = b
        self.c = tuple(c)
        self.isAngular = True
        self.isPeriodic = False
        if len({a, b, c}) != 3:
            raise RuntimeError('a, b, and c must be different')

    def __repr__(self):
        stra = ("("+','.join(["%i" % (i+1) for i in self.a])+")") if len(self.a) > 1 else "%i" % (self.a[0]+1)
        strc = ("("+','.join(["%i" % (i+1) for i in self.c])+")") if len(self.c) > 1 else "%i" % (self.c[0]+1)
        return "%sAngle %s-%i-%s" % ("Multi" if (len(self.a) > 1 or len(self.c) > 1) else "", stra, self.b+1, strc)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.b == other.b:
            if set(self.a) == set(other.a):
                if set(self.c) == set(other.c):
                    return True
            if set(self.a) == set(other.c):
                if set(self.c) == set(other.a):
                    return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        b = self.b
        c = np.array(self.c)
        xyza = np.mean(xyz[a], axis=0)
        xyzc = np.mean(xyz[c], axis=0)
        # vector from first atom to central atom
        vector1 = xyza - xyz[b]
        # vector from last atom to central atom
        vector2 = xyzc - xyz[b]
        # norm of the two vectors
        norm1 = np.sqrt(np.sum(vector1**2))
        norm2 = np.sqrt(np.sum(vector2**2))
        dot = np.dot(vector1, vector2)
        # Catch the edge case that very rarely this number is -1.
        if dot / (norm1 * norm2) <= -1.0:
            if (np.abs(dot / (norm1 * norm2)) + 1.0) < -1e-6:
                raise RuntimeError('Encountered invalid value in angle')
            return np.pi
        return np.arccos(dot / (norm1 * norm2))

    def normal_vector(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        b = self.b
        c = np.array(self.c)
        xyza = np.mean(xyz[a], axis=0)
        xyzc = np.mean(xyz[c], axis=0)
        # vector from first atom to central atom
        vector1 = xyza - xyz[b]
        # vector from last atom to central atom
        vector2 = xyzc - xyz[b]
        # norm of the two vectors
        norm1 = np.sqrt(np.sum(vector1**2))
        norm2 = np.sqrt(np.sum(vector2**2))
        crs = np.cross(vector1, vector2)
        crs /= np.linalg.norm(crs)
        return crs
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = np.array(self.a)
        o = self.b
        n = np.array(self.c)
        xyzm = np.mean(xyz[m], axis=0)
        xyzn = np.mean(xyz[n], axis=0)
        # Unit displacement vectors
        u_prime = (xyzm - xyz[o])
        u_norm = np.linalg.norm(u_prime)
        v_prime = (xyzn - xyz[o])
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        v = v_prime / v_norm
        VECTOR1 = np.array([1, -1, 1]) / np.sqrt(3)
        VECTOR2 = np.array([-1, 1, 1]) / np.sqrt(3)
        if np.linalg.norm(u + v) < 1e-10 or np.linalg.norm(u - v) < 1e-10:
            # if they're parallel
            if ((np.linalg.norm(u + VECTOR1) < 1e-10) or
                    (np.linalg.norm(u - VECTOR2) < 1e-10)):
                # and they're parallel o [1, -1, 1]
                w_prime = np.cross(u, VECTOR2)
            else:
                w_prime = np.cross(u, VECTOR1)
        else:
            w_prime = np.cross(u, v)
        w = w_prime / np.linalg.norm(w_prime)
        term1 = np.cross(u, w) / u_norm
        term2 = np.cross(w, v) / v_norm
        for i in m:
            derivatives[i, :] = term1/len(m)
        for i in n:
            derivatives[i, :] = term2/len(n)
        derivatives[o, :] = -(term1 + term2)
        return derivatives
    
    def second_derivative(self, xyz):
        raise NotImplementedError("Second derivatives have not been implemented for IC type %s" % self.__name__)

class Dihedral(PrimitiveCoordinate):
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.isAngular = True
        self.isPeriodic = True
        if len({a, b, c, d}) != 4:
            raise RuntimeError('a, b, c and d must be different')

    def __repr__(self):
        return "Dihedral %i-%i-%i-%i" % (self.a+1, self.b+1, self.c+1, self.d+1)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.a == other.a:
            if self.b == other.b:
                if self.c == other.c:
                    if self.d == other.d:
                        return True
        if self.a == other.d:
            if self.b == other.c:
                if self.c == other.b:
                    if self.d == other.a:
                        return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        vec1 = xyz[b] - xyz[a]
        vec2 = xyz[c] - xyz[b]
        vec3 = xyz[d] - xyz[c]
        cross1 = np.cross(vec2, vec3)
        cross2 = np.cross(vec1, vec2)
        arg1 = np.sum(np.multiply(vec1, cross1)) * \
               np.sqrt(np.sum(vec2**2))
        arg2 = np.sum(np.multiply(cross1, cross2))
        answer = np.arctan2(arg1, arg2)
        return answer
    
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = self.a
        o = self.b
        p = self.c
        n = self.d
        u_prime = (xyz[m] - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyz[n] - xyz[p])
        u_norm = np.linalg.norm(u_prime)
        w_norm = np.linalg.norm(w_prime)
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        w = w_prime / w_norm
        v = v_prime / v_norm
        if (1 - np.dot(u, w)**2) < 1e-6:
            term1 = np.cross(u, w) * 0
            term3 = np.cross(u, w) * 0
        else:
            term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
            term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        if (1 - np.dot(v, w)**2) < 1e-6:
            term2 = np.cross(v, w) * 0
            term4 = np.cross(v, w) * 0
        else:
            term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
            term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        # term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        # term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        # term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        # term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        derivatives[m, :] = term1
        derivatives[n, :] = -term2
        derivatives[o, :] = -term1 + term3 - term4
        derivatives[p, :] = term2 - term3 + term4
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        m = self.a
        o = self.b
        p = self.c
        n = self.d
        u_prime = (xyz[m] - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyz[n] - xyz[p])
        lu = np.linalg.norm(u_prime)
        lw = np.linalg.norm(w_prime)
        lv = np.linalg.norm(v_prime)
        u = u_prime / lu
        w = w_prime / lw
        v = v_prime / lv
        cu = np.dot(u, w)
        su = (1 - np.dot(u, w)**2)**0.5
        su4 = su**4
        cv = np.dot(v, w)
        sv = (1 - np.dot(v, w)**2)**0.5
        sv4 = sv**4
        if su < 1e-6 or sv < 1e-6 : return deriv2
        
        uxw = np.cross(u, w)
        vxw = np.cross(v, w)

        term1 = np.outer(uxw, w*cu - u)/(lu**2*su4)
        term2 = np.outer(vxw, -w*cv + v)/(lv**2*sv4)
        term3 = np.outer(uxw, w - 2*u*cu + w*cu**2)/(2*lu*lw*su4)
        term4 = np.outer(vxw, w - 2*v*cv + w*cv**2)/(2*lv*lw*sv4)
        term5 = np.outer(uxw, u + u*cu**2 - 3*w*cu + w*cu**3)/(2*lw**2*su4)
        term6 = np.outer(vxw,-v - v*cv**2 + 3*w*cv - w*cv**3)/(2*lw**2*sv4)
        term1 += term1.T
        term2 += term2.T
        term3 += term3.T
        term4 += term4.T
        term5 += term5.T
        term6 += term6.T
        def mk_amat(vec):
            amat = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    if i == j: continue
                    k = 3 - i - j
                    amat[i, j] = vec[k] * (j-i) * ((-0.5)**np.abs(j-i))
            return amat
        term7 = mk_amat((-w*cu + u)/(lu*lw*su**2))
        term8 = mk_amat(( w*cv - v)/(lv*lw*sv**2))
        def zeta(a_, m_, n_):
            return (int(a_==m_) - int(a_==n_))
        # deriv2_terms = [np.zeros_like(deriv2) for i in range(9)]
        # Accumulate the second derivative
        for a in [m, n, o, p]:
            for b in [m, n, o, p]:
                deriv2[a, :, b, :] = (zeta(a, m, o)*zeta(b, m, o)*term1 +
                                      zeta(a, n, p)*zeta(b, n, p)*term2 +
                                      (zeta(a, m, o)*zeta(b, o, p) + zeta(a, p, o)*zeta(b, o, m))*term3 +
                                      (zeta(a, n, p)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, n, p))*term4 +
                                      zeta(a, o, p)*zeta(b, p, o)*term5 +
                                      zeta(a, p, o)*zeta(b, o, p)*term6)
                if a != b:
                    deriv2[a, :, b, :] += ((zeta(a, m, o)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, o, m))*term7 +
                                           (zeta(a, n, o)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, o, n))*term8)
        return deriv2
                    
        # Accumulate a dictionary of contributions to the second derivatives by term (for debugging)
        #             deriv2_terms[7][a, :, b, :] = (zeta(a, m, o)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, o, m))*term7
        #             deriv2_terms[8][a, :, b, :] = (zeta(a, n, o)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, o, n))*term8
        #         deriv2_terms[1][a, :, b, :] = zeta(a, m, o)*zeta(b, m, o)*term1
        #         deriv2_terms[2][a, :, b, :] = zeta(a, n, p)*zeta(b, n, p)*term2
        #         deriv2_terms[3][a, :, b, :] = (zeta(a, m, o)*zeta(b, o, p) + zeta(a, p, o)*zeta(b, o, m))*term3
        #         deriv2_terms[4][a, :, b, :] = (zeta(a, n, p)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, n, p))*term4
        #         deriv2_terms[5][a, :, b, :] = zeta(a, o, p)*zeta(b, p, o)*term5
        #         deriv2_terms[6][a, :, b, :] = zeta(a, p, o)*zeta(b, o, p)*term6
        # deriv2_terms[0] = deriv2.copy()
        # 
        #=======
        # Term-by-term checking of the second derivative.
        # Produces output such as:
        # 1x1x a:  0.0000 n:  0.0000 e:  0.0000 Terms: NNNNNNNN  0.0000  0.0000 -0.0000 -0.0000 -0.0000 -0.0000  0.0000  0.0000
        # 1x1y a:  0.3337 n:  0.3337 e:  0.0000 Terms: YNNNNNNN  0.3337  0.0000 -0.0000 -0.0000 -0.0000 -0.0000  0.0000  0.0000
        # 1x1z a:  0.0590 n:  0.0590 e: -0.0000 Terms: YNNNNNNN  0.0590  0.0000 -0.0000  0.0000 -0.0000  0.0000  0.0000  0.0000
        # 
        # def printTerm(strin, num):
        #     i = int(strin[0])-1
        #     j = 'xyz'.index(strin[1])
        #     k = int(strin[2])-1
        #     l = 'xyz'.index(strin[3])
        #     ana = deriv2_terms[0][i,j,k,l]
        #     err = ana-num
        #     correct = np.abs(num-ana) < 1e-5
        #     color = '\x1b[92m' if correct else '\x1b[91m'
        #     print('%i%s%i%s a: % .4f n: % .4f e: % .4f Terms: ' % (i+1, 'xyz'[j], k+1, 'xyz'[l], ana, num, err) +
        #           ''.join(["Y" if np.abs(deriv2_terms[m][i,j,k,l]) > 1e-5 else "N" for m in range(1, 9)]) + ' ' +
        #           ' '.join(["%s% .4f\x1b[0m" % (color if np.abs(deriv2_terms[m][i,j,k,l]) > 1e-5 else '',
        #                                         deriv2_terms[m][i,j,k,l]) for m in range(1, 9)]))
        # print("LP checking single term:")
        # printTerm('1x1x',  5.55112e-09)
        # printTerm('1x1y',  3.33702e-01)
        # printTerm('1x1z',  5.90389e-02)

class MultiDihedral(PrimitiveCoordinate): # pragma: no cover
    def __init__(self, a, b, c, d):
        if type(a) is int:
            a = (a, )
        if type(d) is int:
            d = (d, )
        self.a = tuple(a)
        self.b = b
        self.c = c
        self.d = tuple(d)
        self.isAngular = True
        self.isPeriodic = True
        if len({a, b, c, d}) != 4:
            raise RuntimeError('a, b, c and d must be different')

    def __repr__(self):
        stra = ("("+','.join(["%i" % (i+1) for i in self.a])+")") if len(self.a) > 1 else "%i" % (self.a[0]+1)
        strd = ("("+','.join(["%i" % (i+1) for i in self.d])+")") if len(self.d) > 1 else "%i" % (self.d[0]+1)
        return "%sDihedral %s-%i-%i-%s" % ("Multi" if (len(self.a) > 1 or len(self.d) > 1) else "", stra, self.b+1, self.c+1, strd)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if set(self.a) == set(other.a):
            if self.b == other.b:
                if self.c == other.c:
                    if set(self.d) == set(other.d):
                        return True
        if set(self.a) == set(other.d):
            if self.b == other.c:
                if self.c == other.b:
                    if set(self.d) == set(other.a):
                        return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        b = self.b
        c = self.c
        d = np.array(self.d)
        xyza = np.mean(xyz[a], axis=0)
        xyzd = np.mean(xyz[d], axis=0)
        
        vec1 = xyz[b] - xyza
        vec2 = xyz[c] - xyz[b]
        vec3 = xyzd - xyz[c]
        cross1 = np.cross(vec2, vec3)
        cross2 = np.cross(vec1, vec2)
        arg1 = np.sum(np.multiply(vec1, cross1)) * \
               np.sqrt(np.sum(vec2**2))
        arg2 = np.sum(np.multiply(cross1, cross2))
        answer = np.arctan2(arg1, arg2)
        return answer
    
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = np.array(self.a)
        o = self.b
        p = self.c
        n = np.array(self.d)
        xyzm = np.mean(xyz[m], axis=0)
        xyzn = np.mean(xyz[n], axis=0)
        
        u_prime = (xyzm - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyzn - xyz[p])
        u_norm = np.linalg.norm(u_prime)
        w_norm = np.linalg.norm(w_prime)
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        w = w_prime / w_norm
        v = v_prime / v_norm
        if (1 - np.dot(u, w)**2) < 1e-6:
            term1 = np.cross(u, w) * 0
            term3 = np.cross(u, w) * 0
        else:
            term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
            term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        if (1 - np.dot(v, w)**2) < 1e-6:
            term2 = np.cross(v, w) * 0
            term4 = np.cross(v, w) * 0
        else:
            term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
            term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        # term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        # term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        # term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        # term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        for i in self.a:
            derivatives[i, :] = term1/len(self.a)
        for i in self.d:
            derivatives[i, :] = -term2/len(self.d)
        derivatives[o, :] = -term1 + term3 - term4
        derivatives[p, :] = term2 - term3 + term4
        return derivatives
    
    def second_derivative(self, xyz):
        raise NotImplementedError("Second derivatives have not been implemented for IC type %s" % self.__name__)

class OutOfPlane(PrimitiveCoordinate):
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.isAngular = True
        self.isPeriodic = True
        if len({a, b, c, d}) != 4:
            raise RuntimeError('a, b, c and d must be different')

    def __repr__(self):
        return "Out-of-Plane %i-%i-%i-%i" % (self.a+1, self.b+1, self.c+1, self.d+1)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.a == other.a:
            if {self.b, self.c, self.d} == {other.b, other.c, other.d}:
                if [self.b, self.c, self.d] != [other.b, other.c, other.d]:
                    logger.warning("Warning: OutOfPlane atoms are the same, ordering is different\n")
                return True
        #     if self.b == other.b:
        #         if self.c == other.c:
        #             if self.d == other.d:
        #                 return True
        # if self.a == other.d:
        #     if self.b == other.c:
        #         if self.c == other.b:
        #             if self.d == other.a:
        #                 return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        vec1 = xyz[b] - xyz[a]
        vec2 = xyz[c] - xyz[b]
        vec3 = xyz[d] - xyz[c]
        cross1 = np.cross(vec2, vec3)
        cross2 = np.cross(vec1, vec2)
        arg1 = np.sum(np.multiply(vec1, cross1)) * \
               np.sqrt(np.sum(vec2**2))
        arg2 = np.sum(np.multiply(cross1, cross2))
        answer = np.arctan2(arg1, arg2)
        return answer
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = self.a
        o = self.b
        p = self.c
        n = self.d
        u_prime = (xyz[m] - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyz[n] - xyz[p])
        u_norm = np.linalg.norm(u_prime)
        w_norm = np.linalg.norm(w_prime)
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        w = w_prime / w_norm
        v = v_prime / v_norm
        if (1 - np.dot(u, w)**2) < 1e-6:
            term1 = np.cross(u, w) * 0
            term3 = np.cross(u, w) * 0
        else:
            term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
            term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        if (1 - np.dot(v, w)**2) < 1e-6:
            term2 = np.cross(v, w) * 0
            term4 = np.cross(v, w) * 0
        else:
            term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
            term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        # term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        # term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        # term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        # term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        derivatives[m, :] = term1
        derivatives[n, :] = -term2
        derivatives[o, :] = -term1 + term3 - term4
        derivatives[p, :] = term2 - term3 + term4
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        deriv2 = np.zeros((xyz.shape[0], 3, xyz.shape[0], 3), dtype=float)
        h = 1.0e-3
        for i in range(4):
            for j in range(3):
                ii = [a, b, c, d][i]
                xyz[ii, j] += h
                FPlus = self.derivative(xyz)
                xyz[ii, j] -= 2*h
                FMinus = self.derivative(xyz)
                xyz[ii, j] += h
                fderiv = (FPlus-FMinus)/(2*h)
                deriv2[ii, j, :, :] = fderiv
        return deriv2

# List of Primitive Internal Coordinate.

def convert_bohrs_degrees(prims, values):
    """ Convert values of primitive ICs (or differences) from
    weighted atomic units to Bohrs and degrees.
    The unit of translation will be Bohrs.
    The unit of rotation will be degrees. """
    converted = np.array(values).copy()
    for ic, c in enumerate(prims):
        if type(c) in [TranslationX, TranslationY, TranslationZ]:
            w = 1.0
        elif hasattr(c, 'w'):
            w = c.w
        else:
            w = 1.0
        if type(c) in [TranslationX, TranslationY, TranslationZ, CartesianX, CartesianY, CartesianZ, Distance, LinearAngle, CentroidDistance]:
            factor = 1  # here we assume the unit of translation will be Bohrs.
        elif c.isAngular:
            factor = 180.0 / np.pi
        converted[ic] /= w
        converted[ic] *= factor
    return converted

# Parent class for PrimitiveInternalCoordinates and DelocalizedInternalCoordinates
class InternalCoordinates(object):
    def __init__(self):
        self.stored_wilsonB = OrderedDict()  # orderedDict, key: xyz. value: Wilson B matrix.

    def addConstraint(self, cPrim, cVal):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def haveConstraints(self):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def augmentGH(self, xyz, G, H):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def calcGradProj(self, xyz, gradx):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def clearCache(self):
        self.stored_wilsonB = OrderedDict()

    def wilsonB(self, xyz, invMW=False):
        """
        Given Cartesian coordinates xyz, return the Wilson B-matrix
        given by dq_i/dx_j where x is flattened (i.e. x1, y1, z1, x2, y2, z2)
        """
        global CacheWarning
        t0 = time.time()
        xyz = xyz.flatten()
        xhash = hash(xyz.tobytes())
        ht = time.time() - t0
        if xhash in self.stored_wilsonB:
            ans = self.stored_wilsonB[xhash]
            return ans
        WilsonB = []
        Der = self.derivatives(xyz)
        for i in range(Der.shape[0]):
            WilsonB.append(Der[i].flatten())
        self.stored_wilsonB[xhash] = np.array(WilsonB)
        if len(self.stored_wilsonB) > 1000 and not CacheWarning:
            logger.warning("\x1b[91mWarning: more than 1000 B-matrices stored, memory leaks likely\x1b[0m\n")
            CacheWarning = True
        ans = np.array(WilsonB)
        if invMW:
            ans /= np.tile(np.sqrt(self.mass), (len(self.Internals), 1))
        return ans

    def GMatrix(self, xyz, invMW=False):
        """
        Given Cartesian coordinates xyz, return the G-matrix
        given by G = BuBt where u is an arbitrary matrix (default to identity)
        """
        Bmat = self.wilsonB(xyz, invMW)
        BuBt = np.dot(Bmat,Bmat.T)
        return BuBt

    def GInverse(self, xyz, sqrt=False, invMW=False):
        """
        Compute inverse of G: G^{-1}. 
        Here G= B * B^T.  
        G^{-1} = (B * B^T)^{-1}.
        We perform the general inverse of matrix G using SVD.
        """
        xyz = xyz.reshape(-1,3)
        # Perform singular value decomposition
        click()
        loops = 0
        while True:
            try:
                G = self.GMatrix(xyz, invMW)
                time_G = click()
                U, S, VT = np.linalg.svd(G)
                time_svd = click()
            except np.linalg.LinAlgError:
                logger.warning("\x1b[1;91m SVD fails, perturbing coordinates and trying again\x1b[0m\n")
                xyz = xyz + 1e-2*np.random.random(xyz.shape)
                loops += 1
                if loops == 10:
                    raise RuntimeError('SVD failed too many times')
                continue
            break
        # print "Build G: %.3f SVD: %.3f" % (time_G, time_svd),
        V = VT.T
        UT = U.T
        Sinv = np.zeros_like(S)
        Ssqrt = np.zeros_like(S)
        LargeVals = 0
        for ival, value in enumerate(S):
            # print "%.5e % .5e" % (ival,value)
            if np.abs(value) > 1e-6:
                if sqrt: value = np.sqrt(value)
                LargeVals += 1
                Sinv[ival] = 1/value
                Ssqrt[ival] = value

        # print "%i atoms; %i/%i singular values are > 1e-6" % (xyz.shape[0], LargeVals, len(S))
        Sinv = np.diag(Sinv)
        Inv = multi_dot([V, Sinv, UT])
       
        # When "sqrt" is True, return the sqrt of the G matrix along with its inverse.
        # Sqrt of the G matrix is used to calculate gradients and Hessian in mass-weighted IC.
        if sqrt:
            Ssqrt = np.diag(Ssqrt)
            Sqrt = multi_dot([V, Ssqrt, UT])
            return Inv, Sqrt
        return Inv

 
    def calcGrad(self, xyz, gradx):
        """
        calculate the gradient in internal coordinate. df/dq.
        """
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        # Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx).T
        Gq = multi_dot([Ginv, Bmat, gradx.T])
        return Gq.flatten()

    def calcHess(self, xyz, gradx, hessx):
        """
        Compute the internal coordinate Hessian. 
        Expects Cartesian coordinates to be provided in a.u.
        """
        xyz = xyz.flatten()
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        Gq = self.calcGrad(xyz, gradx)
        deriv2 = self.second_derivatives(xyz)
        Bmatp = deriv2.reshape(deriv2.shape[0], xyz.shape[0], xyz.shape[0])
        Hx_BptGq = hessx - np.einsum('pmn,p->mn',Bmatp,Gq)
        Hq = np.einsum('ps,sm,mn,nr,rq', Ginv, Bmat, Hx_BptGq, Bmat.T, Ginv, optimize=True)
        return Hq

    def calcGradCart(self, xyz, gradq):
        """
        calculate the gradient in Cartesian coordinate. df/dx.
        Gx = B^T * gradq.
        """
        Bmat = self.wilsonB(xyz)
        # Cartesian coordinate gradient.
        Gx = np.transpose(Bmat) @ gradq 
        return Gx 
        

    def calcHessCart(self, xyz, gradq, hessq):
        """
        Compute the Cartesian Hessian given internal coordinate gradient and Hessian. 
        Returns the answer in a.u.
        """
        xyz = xyz.flatten()
        Bmat = self.wilsonB(xyz)
        deriv2 = self.second_derivatives(xyz)
        Bmatp = deriv2.reshape(deriv2.shape[0], xyz.shape[0], xyz.shape[0])
        BptGq = np.einsum('pmn,p->mn',Bmatp,gradq)
        Hx = np.einsum('ai,ab,bj->ij', Bmat, hessq, Bmat, optimize=True)
        Hx += BptGq
        return Hx
    
    @property
    def conmethod(self):
        ''' algorithm for constraint satisfaction

        Notes:
            - `0`: Original algorithm implemented in 2016
            - `1`: Updated algorithm implemented on 2019-03-20

        Returns:
            None | int: integer if the algorithm is applicable and indicate the revision of method,
                        or else `None` is returned
        '''
        if hasattr(self, '_conmethod'):
            return self._conmethod
        return None

    @conmethod.setter
    def conmethod(self, val):
        ''' set the algorithm for constraint satisfaction

        Args:
            val (None | int): algorithm revision
        '''
        self._conmethod = val

    @property
    def rigid(self):
        ''' Flag for rigid optimizations (valid for DLC only)

        Notes:
            - `0`: Rigid optimizations off
            - `1`: Rigid optimizations on

        Returns:
            None | bool: True if rigid optimizations are enabled, or else `None` is returned
        '''
        if hasattr(self, '_rigid'):
            return self._rigid
        return None
        
    @rigid.setter
    def rigid(self, val):
        ''' set the flag for rigid optimizations

        Args:
            val (None | bool): Whether rigid optimizations are on, off or undefined
        '''
        self._rigid = val

class PrimitiveInternalCoordinates(InternalCoordinates):
    """
    Primitive Redundant Internal Coordinate.
    We do not implement TRIC and constraint in the current class. 
    This is to simplify the implementation.
    """
    def __init__(self, molecule: Molecule, connect=False, addcart=False, **kwargs):
        super(PrimitiveInternalCoordinates, self).__init__()
        # connect = True corresponds to "traditional" internal coordinates with minimum spanning bonds
        # connect = False, addcart = True corresponds to HDLC
        # connect = False, addcart = False corresponds to TRIC
        self.connect = connect
        self.addcart = addcart

        self.Internals = []  # List of Primitive Internal coordinates.

        self.elem = molecule.elem
        # Atomic mass array
        self.mass = np.repeat([PeriodicTable[i] for i in self.elem], 3)
        # Here each molecule object corresponds to one frame in geomeTRIC.
        self.makePrimitives(molecule, connect, addcart)

    def makePrimitives(self, molecule: Molecule, connect, addcart):
        """
        Make primitive internal coordinates based on atom connectivity topology of the molecule.
        """
        # build topology for molecules based on atom radius and distance between atoms. 
        molecule.build_topology() 

        # coordinates in Angstrom
        coords = molecule.xyz.flatten()
        # Make a distance matrix mapping atom pairs to inter-atomic distances.
        AtomIterator, dxij = molecule.distance_matrix(pbc= False)

        # record the distance between atoms (this belongs to Internal Coordinate.)
        D = {} 
        for index, r in zip(AtomIterator, dxij[0]):
            assert index[0] < index[1]
            D[tuple(index)] = r 
        
        # create non-covalent bond to characterize inter-molecular interaction.
        # Or use HDIC. The TRIC currently is not implemented.
        # Add non-covalent bonds into self.Internal.
        noncov = self.add_noncov(molecule, D)

        # Add an internal coordinate for bonded atom pairs. (Distance between bonded atom pairs)
        for (a,b) in molecule.topology.edges():
            self.add(Distance(a,b))
        
        # Linear Angle threshold -- corresponds about 162 degrees.
        self.LinThre = 0.95

        # Add angle between bonded molecules.
        # Or bending internal coords corresponding to linear 
        # The angular internal coordinates is added to self.Internals
        AngDict = self.add_angle(molecule, noncov, coords)

        # Out of Plane internal coordinate.
        # The out of plane internal coordinate is added to self.Internals.
        self.add_out_of_plane(molecule, noncov, coords)

        # Creating dihedrals
        # The dihedral angle internal coordinate is added to self.Internals.
        self.add_dihedral(molecule, noncov, coords)

    def add_noncov(self, molecule: Molecule, D: dict):
        """
        add non-covalent bond between molecular fragments.
        This is to treat inter-molecular interaction.
        Transition rotation Internal Coordinate (TRIC), Hybrid Delocalized Internal Coordinate (HDIC)
        and traditional approach (all connected method) differs from each other 
        in how we generate internal coordinates that characterize inter-molecular interaction.
        Here we only implement the traditional way.

        :param: molecule: Molecule class object. topology of atom connectivity.
        :param: D: dictionary of distance between atom pairs.
        """
        # create a weighted graph. The weight of edge is intr-atom distance r.
        dgraph = nx.Graph()
        
        for i in range(molecule.na):
            dgraph.add_node(i)

        for k, v in D.items():
            # key is AtomIterator tuple (i,j)
            # v is distance r.
            dgraph.add_edge(k[0], k[1], weight= v)
        
        # minimum spanning trees.
        mst = sorted(
            list(
                    nx.minimum_spanning_edges(dgraph, data= False)
                )
            )
        
        # build a list of noncovalent distances.
        noncov = []
        # Connect all non-bonded fragments together.
        if self.connect:
            for edge in mst:
                if edge not in list(molecule.topology.edges()):
                    # This edge is between non-bonded fragments.
                    molecule.topology.add_edge(edge[0], edge[1])
                    noncov.append(edge)
        else:
            if self.addcart:
                for i in range(molecule.na):
                    self.add(CartesianX(i, w= 1.0))
                    self.add(CartesianY(i, w= 1.0))
                    self.add(CartesianZ(i, w= 1.0))
            else:
                raise NotImplementedError("We do not implement Translation Rotation Internal Coordinate Here.\
                                           This may change in the future.")

        return noncov 
    
    def add_angle(self, molecule: Molecule, noncov, coords):
        """
        Add angles and linear angles for bonded atoms.
        :param: molecule: Molecular object. Contains information about connectivity between atoms.
        :param: noncov: list of noncov bonds (record edge (a,b))
        :param: coords: coordinates of molecule (3 * Natom.)
        """
        AngDict = defaultdict(list)
        for b in molecule.topology.nodes():
            for a in molecule.topology.neighbors(b):
                for c in molecule.topology.neighbors(b):
                    if a < c:
                        Ang = Angle(a, b, c)

                        # check if bond is non-covalent bond.
                        is_ab_noncov = (min(a,b), max(a,b)) in noncov
                        is_bc_noncov = (min(b,c), max(b,c)) in noncov 
                        # results for not noncov.
                        nnc = (is_ab_noncov & is_bc_noncov)

                        if np.abs(np.cos(Ang.value(coords))) < self.LinThre:
                            # not linear.
                            self.add(Angle(a, b, c))
                            AngDict[b].append(Ang)
                        elif self.connect or (not self.addcart):
                            # Add linear angle IC's
                            # LPW 2019-02-16: Linear angle ICs work well for "very" linear angles in molecules (e.g. HCCCN)
                            # but do not work well for "almost" linear angles in noncovalent systems (e.g. H2O6).
                            # Bringing back old code to use "translations" for the latter case, but should be investigated
                            # more deeply in the future.
                            # LPW 2022-02-15: Linear angle ICs have been improved, and should no longer require resetting if the
                            # atoms in the angle go through a large rotation. They are currently being used.
                            if nnc == 0:
                                # both (a,b) and (b,c) is not in noncov, which means that are covalent bond.
                                # Add two internal coordinates corresponding to the bending of the linear angle.
                                self.add(LinearAngle(a, b, c, 0))
                                self.add(LinearAngle(a, b, c, 1))
                            else:
                                # Use translation in case there is non-covalent linear bond.
                                # Unit vector connecting atoms a and c
                                nac= molecule.xyz[c] - molecule.xyz[a]
                                nac = nac / np.linalg.norm(nac)
                                # Dot products of this vector with the Cartesian axes
                                dots = [np.abs(np.dot(ei, nac)) for ei in np.eye(3)]
                                # Functions for adding Cartesian coordinate
                                # carts = [CartesianX, CartesianY, CartesianZ]
                                trans = [TranslationX, TranslationY, TranslationZ]
                                w = np.array([-1.0, 2.0, -1.0])
                                # Add two of the most perpendicular Cartesian coordinates
                                for i in np.argsort(dots)[:2]:
                                    self.add(trans[i]([a, b, c], w=w))
        
        return AngDict 

    def add_out_of_plane(self, 
                         molecule:Molecule, 
                         noncov,
                         coords):
        """
        Add out of plane internal coordinate.
        :param: molecule: Molecular object. Contains information about connectivity between atoms.
        :param: noncov: list of noncov bonds (record edge (a,b))
        :param: coords: coordinates of molecule (3 * Natom.)
        """
        # out of plane internal coordinate.
        for b in molecule.topology.nodes():
            for a in molecule.topology.neighbors(b):
                for c in molecule.topology.neighbors(b):
                    for d in molecule.topology.neighbors(b):
                        if a < c < d:
                            nnc = (min(a, b), max(a, b)) in noncov
                            nnc += (min(b, c), max(b, c)) in noncov
                            nnc += (min(b, d), max(b, d)) in noncov
                            # if nnc >= 1: continue
                            for i, j, k in sorted(list(itertools.permutations([a, c, d], 3))):
                                Ang1 = Angle(b,i,j)
                                Ang2 = Angle(i,j,k)
                                if np.abs(np.cos(Ang1.value(coords))) > self.LinThre: continue
                                if np.abs(np.cos(Ang2.value(coords))) > self.LinThre: continue
                                # This is to take care the case that 4 molecules is in a plane (BH3), the possible out of plane bending.
                                if np.abs(np.dot(Ang1.normal_vector(coords), Ang2.normal_vector(coords))) > self.LinThre:  # linear.
                                    self.delete(Angle(i, b, j))
                                    self.add(OutOfPlane(b, i, j, k))
                                    break
        

    def add_dihedral(self,
                     molecule: Molecule,
                     noncov,
                     coords):
        """
        Add dihedral angle between 4 atoms as internal coordinate.
        :param: molecule: Molecular object. Contains information about connectivity between atoms.
        :param: noncov: list of noncov bonds (record edge (a,b))
        :param: coords: coordinates of molecule (3 * Natom.)
        """
        # Find groups of atoms that are in straight lines. This is for creating dihedrals.
        atom_lines = [list(i) for i in molecule.topology.edges()]
        # This will make atom_lines include line object (a,b,c,d,e,..) where atom aligns in a line.
        while True:
            # For a line of two atoms (one bond):
            # AB-AC
            # AX-AY
            # i.e. AB is the first one, AC is the second one
            # AX is the second-to-last one, AY is the last one
            # AB-AC-...-AX-AY
            # AB-(AC, AX)-AY
            atom_lines0 = deepcopy(atom_lines)
            for aline in atom_lines:
                # Imagine a line of atoms going like ab-ac-ax-ay.
                # Our job is to extend the line until there are no more
                ab = aline[0]  # node 0 of edge aline
                ay = aline[-1] # node 1 of edge aline.
                for aa in molecule.topology.neighbors(ab):
                    if aa not in aline:
                        # If the angle that AA makes with AB and ALL other atoms AC in the line are linear:
                        # Add AA to the front of the list
                        if all([np.abs(np.cos(Angle(aa, ab, ac).value(coords))) > self.LinThre for ac in aline[1:] if ac != ab]):
                            aline.insert(0, aa)
                for az in molecule.topology.neighbors(ay):
                    if az not in aline:
                        if all([np.abs(np.cos(Angle(ax, ay, az).value(coords))) > self.LinThre for ax in aline[:-1] if ax != ay]):
                            aline.append(az)

            # If no further atoms add to liner line, break.
            if atom_lines == atom_lines0: break
        
        # unique set of atom lines.
        atom_lines_uniq = []
        for atom_line in atom_lines:     
            if tuple(atom_line) not in set(atom_lines_uniq):
                atom_lines_uniq.append(tuple(atom_line))
        
        # Normal dihedral code
        for aline in atom_lines_uniq:
            # Go over ALL pairs of atoms in a line
            for (b, c) in itertools.combinations(aline, 2):  # generate a pair of atom indices.
                if b > c: (b, c) = (c, b)
                # Go over all neighbors of b
                for a in molecule.topology.neighbors(b):
                    # Go over all neighbors of c
                    for d in molecule.topology.neighbors(c):
                        # Make sure the end-atoms are not in the line and not the same as each other
                        if a not in aline and d not in aline and a != d:
                            nnc = (min(a, b), max(a, b)) in noncov
                            nnc += (min(b, c), max(b, c)) in noncov
                            nnc += (min(c, d), max(c, d)) in noncov
                            # print aline, a, b, c, d
                            Ang1 = Angle(a,b,c)
                            Ang2 = Angle(b,c,d)
                            # Eliminate dihedrals containing angles that are almost linear
                            # (should be eliminated already)
                            if np.abs(np.cos(Ang1.value(coords))) > self.LinThre: continue
                            if np.abs(np.cos(Ang2.value(coords))) > self.LinThre: continue
                            self.add(Dihedral(a, b, c, d))

    # Return internal coordinates:
    def calculate(self, xyz):
        """
        Return Internal coordinates corresponds to xyz.
        """
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.value(xyz))
        return np.array(answer)

    # Return derivatives of internal coordinates.
    def derivatives(self, xyz):
        # self.calculate(xyz)
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.derivative(xyz))
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        return np.array(answer)

    def second_derivatives(self, xyz):
        # self.calculate(xyz)
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.second_derivative(xyz))
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        # 4) Number of atoms
        # 5) 3
        return np.array(answer)

    # Auxiliary function.
    def add(self, dof):
        if dof not in self.Internals:
            self.Internals.append(dof)
    
    def delete(self, dof):
        for ii in range(len(self.Internals))[::-1]:
            if dof == self.Internals[ii]:
                del self.Internals[ii]
    
    def __repr__(self):
        lines = ["Internal coordinate system (atoms numbered from 1):"]
        typedict = OrderedDict()
        for Internal in self.Internals:
            lines.append(Internal.__repr__())
            if str(type(Internal)) not in typedict:
                typedict[str(type(Internal))] = 1
            else:
                typedict[str(type(Internal))] += 1
        if len(lines) > 1000:
            # Print only summary if too many
            lines = []
        for k, v in typedict.items():
            lines.append("%s : %i" % (k, v))
        return '\n'.join(lines)


    def __eq__(self, other):
        answer = True
        for i in self.Internals:
            if i not in other.Internals:
                answer = False
        for i in other.Internals:
            if i not in self.Internals:
                answer = False
        return answer

    def __ne__(self, other):
        return not self.__eq__(other)

class DelocalizedInternalCoordinates(InternalCoordinates):
    def __init__(self, molecule: Molecule,  connect=False, addcart=False):
        super(DelocalizedInternalCoordinates, self).__init__()
        # HDLC is given by (connect = False, addcart = True)
        # Standard DLC is given by (connect = True, addcart = False)
        # TRIC is given by (connect = False, addcart = False)
        self.connect = connect
        # Add Cartesian coordinates to all.
        self.addcart = addcart

        self.molecule = molecule
        # The DLC contains an instance of primitive internal coordinates.
        self.Prims = PrimitiveInternalCoordinates(molecule, connect=connect, addcart=addcart)
        self.frags = self.Prims.frags
        self.na = molecule.na
        # Atomic mass array
        self.mass = np.repeat([PeriodicTable[i] for i in molecule.elem], 3)

        # Build the DLC's. This takes some time, so we have the option to turn it off.
        # xyz in molecule.xyz is already in bohr unit.
        xyz = molecule.xyz.flatten()
        
        self.build_dlc(xyz)
    
    def build_dlc(self, xyz):
        """
        Build delocalized internal coordinate.
        param: xyz: Cartesian coordinate.
        """
        Bmat = self.wilsonB(xyz)
        # SVD decomposition of Bmat
        U, S, Vh = np.linalg.svd(Bmat, full_matrices= False)

        natom = self.na
        # If we do not include information about the position of 
        # the center of mass and the orientation of molecular system.
        # The number of nonzero eigenvalue should be 3N-6.
        dlc_na = 3 * natom - 6
        
        assert (
            np.size(S) >= dlc_na
        ), "number of nonzero singular value of B is smaller than 3n-6. Wrong"

        # sort singular value according to their absolute values. descending order
        s_index = np.array(range(len(S)))
        nonzero_S_index = s_index[: dlc_na]
        nonzero_S = S[nonzero_S_index]

        # sanity check in case we have zero sinuglar value number larger than 3n-6.
        zero_S_index = s_index[dlc_na :]
        zero_S = S[zero_S_index]
        if np.size(zero_S) != 0:
            zero_s_max = np.max(np.abs(zero_S))
            if zero_s_max > np.power(10.0, -4) * np.min(np.abs(nonzero_S)):
                # nonzero value is too large
                raise (
                    "zero singular value of matrix B is too large. zero_s_max: {}  min(nonzero_s): {}".format(
                        zero_s_max, np.min(np.abs(nonzero_S))
                    )
                )
            
        S_nonredundant = S[:-6]
        print(f"All non-redundant singular values: {S_nonredundant}")

        # truncate nonzero singular value.
        U = U[:, :dlc_na]
        Vh = Vh[:dlc_na, :]
        S = S[:dlc_na]

        # record U matrix and singular value matrix S.
        self.ref_U = U  
        self.ref_UT = U.T
        self.S = S
    

    def calculate(self, coords):
        """
        Calculate Delocalized Internal Coordinate given the Cartesian coordinate.
        """
        # Primitive Internal Coordinate.
        PrimVals = self.Prims.calculate(coords)
        
        vals = PrimVals @ self.ref_U 
        
        return vals 
    
    def derivatives(self, coords):
        """
        Calculate the change of the DLCs with respect to the Cartesian coordinates. 
        The returned array has dimensions:
        1) Number of delocalized internal coordinates
        2) Number of atoms
        3) 3
        """
        # shape: [Ninternal, natom, 3]
        PrimDers = self.Prims.derivatives(coords)

        ders = np.tensordot(self.ref_U, PrimDers, axes= (0,0))
        return np.array(ders)
    
    def second_derivatives(self, coords):
        """
        Calculate the second derivatives of DLCs with respect to the Cartesian coordinate.
        This array has dimensions:
        1) Number of delocalized internal coordinates
        2) Number of atoms
        3) 3
        4) Number of atoms
        5) 3
        """
        PrimDers = self.Prims.second_derivatives(coords)
        ders = np.tensordot(self.ref_U, PrimDers, axes= (0,0))

        return ders

    def inverse_second_derivatives(self, coords):
        """
        Calculate the d^x/dq^2. here q is DLC, x is cartesian coordinate.
        d^x/dq^2 = (-1) (dx/dq)^3 (dq/dx)^2 = (-1) * ((B)^{-1})^3 * second_derivative
        """
        # d^2 q/ dx^2, shape: [3n-6, n, 3, n, 3]
        ders = self.second_derivatives(coords)
        
        natom = self.na 
        # shape: [3n-6, 3n, 3n]
        hessian_q_xx = ders.reshape((ders.shape[0], natom * 3, natom * 3))

        # dq/dx : shape[3n-6, 3n]
        Bq = self.wilsonB(coords)
        # shape: [3n, 3n-6]
        inverse_Bq = np.linalg.pinv(Bq)
        # shape: [3n-6, 3n-6, 3n]
        h1 = np.einsum('ijk, jl -> ilk', hessian_q_xx, inverse_Bq)
        # shape: [3n-6, 3n-6, 3n-6]
        h2 = np.einsum('ijk, kl -> ijl', h1, inverse_Bq)
        # shape: [3n, 3n-6, 3n -6]
        h3 = np.einsum('ijk, li -> ljk', h2, inverse_Bq)

        hessian_x_qq = (-1) * h3 

        return hessian_x_qq 

    def __eq__(self, other):
        return self.Prims == other.Prims

    def __ne__(self, other):
        return not self.__eq__(other)
