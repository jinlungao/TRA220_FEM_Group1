import cupy as cp
from abc import ABC, abstractmethod

class MatrixAssembler(ABC):
    """Abstract base class for matrix assembly operators"""
    def __init__(self, m, dx, dtype=cp.float64):
        self.m = m
        self.dx = dx
        self.dtype = dtype

    @abstractmethod
    def assemble(self):
        """
        RETURN (C_hat, K_hat)
        C_hat: mass matrix (m x m)
        K_hat: stiffness (m x m)
        """
        pass