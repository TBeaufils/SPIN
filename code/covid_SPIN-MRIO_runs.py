# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:25:17 2021

@author: beaufils
"""

import shutil
import os
from interface_SPIN import spin

projection_path = os.path.join('..','output')
base_table_path = os.path.join('..','data','MRIOT')

spin('covid_SPIN-MRIO_hist',2015,2019,'covid_SPIN-MRIO_base',trade_zones='world')

shutil.copyfile(os.path.join(projection_path,'covid_SPIN-MRIO_hist_2019_T.csv'),os.path.join(base_table_path,'covid_SPIN-MRIO_hist_2019_T.csv'))
shutil.copyfile(os.path.join(projection_path,'covid_SPIN-MRIO_hist_2019_FD.csv'),os.path.join(base_table_path,'covid_SPIN-MRIO_hist_2019_FD.csv'))
shutil.copyfile(os.path.join(projection_path,'covid_SPIN-MRIO_hist_2019_VA.csv'),os.path.join(base_table_path,'covid_SPIN-MRIO_hist_2019_VA.csv'))

spin('covid_SPIN-MRIO_baseline',2019,2026,'covid_SPIN-MRIO_hist',trade_zones='world')
spin('covid_SPIN-MRIO_counterfactual',2019,2026,'covid_SPIN-MRIO_hist',trade_zones='world')