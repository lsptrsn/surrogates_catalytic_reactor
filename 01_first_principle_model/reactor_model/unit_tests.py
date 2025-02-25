#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:28:16 2023

@author: peterson
"""

import unittest
import numpy as np
from parameter import data
from reaction_rate import reaction_rate
import thermodynamics as thermo
import pressure_drop as pd


class TestReactionRateFunction(unittest.TestCase):
    """Test if reaction rate is correct for given inputs."""

    def test_reaction_rate(self):
        num_points = 300
        T = np.full((num_points,), 300)  # Temperature in K, same entry repeated 300 times
        x = np.full((4, num_points), 0.5)  # Mole fraction, same entry repeated 300 times for each of the 4 species
        p_R_loss_entry = 3 * 1e6
        p_R_loss = np.full((num_points,), p_R_loss_entry)  # Pressure in Pa, same entry repeated 300 times
        expected_rate = 4.21706608e-09
        expected_rate = np.full((num_points,), expected_rate)
        calculated_rate = reaction_rate(T, x, p_R_loss)
        # Add a tolerance for numerical imprecision
        tolerance = 1e-5
        np.testing.assert_allclose(calculated_rate, expected_rate, rtol=tolerance)


class TestThermodynamicsFunctions(unittest.TestCase):
    """Test thermodynamics functions."""

    def test_get_viscosity_fluid(self):
            T = 523.15  # Temperature in K
            x = np.array([0.8, 0.2, 0.0, 0.0])  # Mole fraction
            expected_viscosity = 2.20768547e-05
            calculated_viscosity = thermo.get_viscosity_fluid(T, x)
            # Add a tolerance for numerical imprecision
            tolerance = 1e-8
            np.testing.assert_allclose(calculated_viscosity, expected_viscosity, rtol=tolerance)

    def test_get_cp_fluid(self):
            T = 650  # Temperature in K
            x = np.array([0.536585, 0.134146, 0.109756, 0.219512]).reshape(-1, 1)  # Mole fraction
            expected_cp = 2863.71290617
            calculated_cp = thermo.get_cp_fluid(T, x)
            # Add a tolerance for numerical imprecision
            tolerance = 1e-6
            np.testing.assert_allclose(calculated_cp, expected_cp, rtol=tolerance)


    def test_pressure_drop(self):
            T = 523.15  # Temperature in K
            x = np.array([0.8, 0.2, 0.0, 0.0])  # Mole fraction
            rho_fluid = data["rho_fluid_in"]
            expected_pressure_drop = 5852.96158045
            calculated_pressure_drop = pd.pressure_drop(T, x, rho_fluid)
            # Add a tolerance for numerical imprecision
            tolerance = 1
            np.testing.assert_allclose(calculated_pressure_drop, expected_pressure_drop, rtol=tolerance)

if __name__ == '__main__':
    unittest.main()
