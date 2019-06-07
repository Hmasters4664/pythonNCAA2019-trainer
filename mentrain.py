# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:33:00 2019

@author: Hassani
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:17:15 2019

@author: Hassani
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:34:37 2019

@author: Hassani
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV


d_TstatsDetailed = pd.read_csv('NCAATourneyDetailedResults.csv')
d_seeds = pd.read_csv('NCAATourneySeeds.csv')
#d_2019statsDetailed = pd.read_csv('Prelim2019_RegularSeasonDetailedResults.csv')
d_RstatsDetailed = pd.read_csv('RegularSeasonDetailedResults.csv')
d_Seasons = pd.read_csv('Teams.csv')

db=d_RstatsDetailed 
frames = [db, d_TstatsDetailed]
d_RstatsDetailed = pd.concat(frames)

d_Seasons.drop(labels=['TeamName', 'FirstD1Season', 'LastD1Season'], inplace=True, axis=1)
d_winseeds = d_seeds.rename(columns={'TeamID':'WTeamID', 'Seed':'WSeed'})
d_lossseeds = d_seeds.rename(columns={'TeamID':'LTeamID', 'Seed':'LSeed'})

df_dummy = pd.merge(left=d_TstatsDetailed, right=d_winseeds, how='left', on=['WTeamID','Season'])
df_concat = pd.merge(left=df_dummy, right=d_lossseeds, on=[ 'LTeamID', 'Season'])

d_stats = d_seeds

df=pd.DataFrame(columns=['TeamID','TFGM','TFGA','TFGM3','TFGA3','TFTM','TFTA','TOR','TDR','TAst','TTO','TStl','TBlk', 'OppDREB'])
df2=pd.DataFrame(columns=['TeamID','Shooting','Turnovers','Rebounds','Free Throws','Average Seed'])
#df['Shooting']=((d_RstatsDetailed['WFGM']-d_RstatsDetailed['WFGM3'])+0.5*d_RstatsDetailed['WFGM3'])/d_RstatsDetailed['WFGA']
#Shooting = (WFGM-WFGM3) +0.5*WFGM3

db=d_RstatsDetailed 
timer_out=0
for x in range(1101, 1467):
         TFGM= d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['WFGM']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['LFGM']
         TFGA = d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['WFGA']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['LFGA']
         TFGM3= d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['WFGM3']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['LFGM3']
         TFGA3 = d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['WFGA3']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['LFGA3']
         TFTM= d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['WFTM']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['LFTM']
         TFTA = d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['WFTA']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['LFTA']
         TOR= d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['WOR']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['LOR']
         TDR = d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['WDR']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['LDR']
         TAst= d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['WAst']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['LAst']
         TTO = d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['WTO']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['LTO']
         TStl = d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['WStl']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['LStl']
         TBlk= d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['WBlk']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['LBlk']
         OppDREB=  d_RstatsDetailed[(d_RstatsDetailed.WTeamID == x)].sum()['LDR']+d_RstatsDetailed[(d_RstatsDetailed.LTeamID == x)].sum()['WDR']
         if TFGA > 0:
             arr2=[x,TFGM,TFGA,TFGM3,TFGA3,TFTM,TFTA,TOR,TDR,TAst,TTO,TStl,TBlk,OppDREB]
             df.loc[timer_out]=arr2
             timer_out=timer_out+1
#d_regdet[(d_regdet.WTeamID == x) & (d_regdet.Season == y) ].sum()['WFGM']
         

         
df2['TeamID']=df['TeamID']
#df2['Season']=df['Season']

df2['Shooting']=((df['TFGM']-df['TFGM3'])+(0.5*df['TFGM3']))/df['TFGA']
df2['Turnovers']=df['TTO']/(df['TFGA']+(0.44*df['TFTA'])+df['TTO'])
df2['Rebounds']=df['TOR']/(df['TOR']+df['OppDREB'])
df2['Free Throws']=df['TFTM']/df['TFGA']
d_seeds['Seed'] = d_seeds['Seed'].str.strip( 'abwxyzWXYZ' )

df2.drop(labels=['Average Seed'], inplace=True, axis=1)
#d_seeds = d_seeds.drop(d_seeds[(d_seeds.Season < 2003) ].index)
d_seeds['Seed']=pd.to_numeric(d_seeds['Seed'])* 1.
a = []
for x in range(1101, 1467):
    #print((d_seeds[(d_seeds.TeamID == x) & (d_seeds.Season > 2003)].sum()['Seed']))
    avg=((d_seeds[(d_seeds.TeamID == x) & (d_seeds.Season > 2010)].sum()['Seed']/(d_seeds[(d_seeds.TeamID == x) & (d_seeds.Season > 2010)].count()['Seed'])))
    if avg > 0:    
        a.append(avg)
    else:
        a.append(100)
        
d_Seasons['Average Seed']=a
d_Seasons['Average Seed']=d_Seasons['Average Seed'].astype(int)
d_Seasons['TeamID']=d_Seasons['TeamID'].astype(int)
df2['TeamID']=df2['TeamID'].astype(int)
df2.astype(float)

df2=df2.merge(d_Seasons, on='TeamID', how='left')

    
d_Test = pd.read_csv('NCAATourneyCompactResults2018.csv')
d_Test = d_Test.drop(d_seeds[(d_seeds.Season < 2018) ].index)  
dt= pd.DataFrame()

dt['Team1']= d_Test['WTeamID']
dt['Team2']= d_Test['LTeamID']

d_wins = pd.DataFrame()
d_wins=df2.rename(columns={'TeamID': 'WTeamID','Shooting':'WShooting','Turnovers':'WTurnovers','Rebounds':'WRebounds','Free Throws': 'WFree_Throws','Average Seed':'WAverage_Seed'})

#= d_seeds15.rename(columns={'TeamID':'WTeamID', 'Rank':'Win_Rank','Rank_ADJD':'WDefensive_Rank'})

d_loss = pd.DataFrame()
d_loss=df2.rename(columns={'TeamID': 'LTeamID','Shooting':'LShooting','Turnovers':'LTurnovers','Rebounds':'LRebounds','Free Throws': 'LFree_Throws','Average Seed':'LAverage_Seed'})


