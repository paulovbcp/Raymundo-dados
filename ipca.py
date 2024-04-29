# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:35:14 2024

@author: Paulo Maia
"""

from bcb import sgs

sgs.get({'ipca': 433}, start = '2019-01-01', end = '2024-12-31')

# Não tem previsão (trouxe os valores só até o mês de março)