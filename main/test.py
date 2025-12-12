import numpy as np
import scipy as sp
from sympy import *
from matplotlib import pyplot as plt

class IBVP:
    
    def __init__(self):
        self.num = 5
    
    def test(self):
        print(self.num)