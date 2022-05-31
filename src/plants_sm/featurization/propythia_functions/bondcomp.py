# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
This function gives the sum of the bond composition for each type of bond
For bond composition four types of bonds are
considered total number of bonds (including aromatic), hydrogen bond, single bond and double
bond. The number of values for each kind of bond is provided as bonds.csv file

The code is based on the package Pfeature :
Pande, Akshara & Patiyal, Sumeet & Lathwal, Anjali & Arora, Chakit & Kaur, Dilraj & Dhall, Anjali & Mishra, Gaurav & Kaur,
Harpreet & Sharma, Neelam & Jain, Shipra & Usmani, Salman & Agrawal, Piyush & Kumar, Rajesh & Kumar, Vinod & Raghava, Gajendra.
(2019). Computing wide range of protein/peptide features from their sequence and structure. 10.1101/599126.


It returns a dictionary form with the values

Authors: Ana Marta Sequeira

Date: 05/2019

Email:


##############################################################################
"""
import pandas as pd
import os


# import sys

def boc_wp(seq):
    """
    Sum of the bond composition for each type of bond: total number of bonds (including aromatic), hydrogen bond,
    single bond and double

    :param seq: protein sequence
    :return: dictionary with number of total, hydrogen, single and double bonds
    """
    bb = {}

    bonds_dictionary = {"G": [9, 5, 8, 1],
                        "S": [13, 7, 12, 1],
                        "A": [12, 7, 11, 1],
                        "D": [15, 7, 13, 2],
                        "N": [16, 8, 14, 2],
                        "T": [16, 9, 15, 1],
                        "P": [17, 9, 16, 1],
                        "E": [18, 9, 16, 2],
                        "V": [18, 11, 17, 1],
                        "Q": [19, 10, 17, 2],
                        "M": [19, 11, 18, 1],
                        "H": [20, 9, 17, 3],
                        "I": [21, 13, 20, 1],
                        "Y": [24, 11, 20, 4],
                        "L": [21, 13, 20, 1],
                        "K": [23, 14, 22, 1],
                        "W": [28, 12, 23, 5],
                        "F": [23, 11, 19, 4],
                        "C": [25, 12, 23, 2],
                        "R": [25, 14, 23, 2]
                        }
    total = 0
    h = 0
    s = 0
    d = 0
    for aa in seq:

        total += bonds_dictionary[aa][0]
        h += bonds_dictionary[aa][1]
        s += bonds_dictionary[aa][2]
        d += bonds_dictionary[aa][3]

    bb["total_bonds"] = total
    bb["hydrogen_bonds"] = h
    bb["single bonds"] = s
    bb["double bonds"] = d

    return bb


if __name__ == '__main__':
    print(boc_wp('MQGNGSALPNASQPVLRGDGARPSWLASALACVLIFTIVVDILGNLLVILSVYRNKKLRN'))
    print(boc_wp('MALPNAVIAAAALSVYRNKKLRN'))
    print(boc_wp('MQGNGSPALLNSRRRRRGDGARPSWLASALACVLIFTIVVDILGNLLVILSVYRNKKLRN'))
