# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:25:39 2026

@author: usuario
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

years = np.arange(1990, 2025)

# ---------------------------------------------------------------
# 1) POPULATION (persons) - World Bank SP.POP.TOTL via FRED, EXACT
# ---------------------------------------------------------------
population = {
1990:32440069,1991:33098372,1992:33760571,1993:34441473,1994:35127410,
1995:35804662,1996:36462745,1997:37121149,1998:37792165,1999:38454863,
2000:39089934,2001:39709262,2002:40324680,2003:40937429,2004:41543171,
2005:42128977,2006:42691084,2007:43235374,2008:43758808,2009:44271541,
2010:44777319,2011:45259614,2012:45715810,2013:46151584,2014:46565429,
2015:46969940,2016:47437512,2017:48131078,2018:49024465,2019:49907985,
2020:50629997,2021:51188173,2022:51737944,2023:52321152,2024:52886363,
}

# ---------------------------------------------------------------
# 2) GDP PER CAPITA (constant 2010 US$) - World Bank NY.GDP.PCAP.KD via FRED, EXACT
# ---------------------------------------------------------------
gdp_pc = {
1990:3714.18,1991:3713.18,1992:3787.59,1993:3912.66,1994:4059.32,
1995:4189.73,1996:4198.69,1997:4265.69,1998:4213.83,1999:3967.11,
2000:4016.81,2001:4020.51,2002:4058.28,2003:4154.18,2004:4311.92,
2005:4457.28,2006:4694.03,2007:4947.25,2008:5048.57,2009:5046.97,
2010:5214.25,2011:5517.10,2012:5675.76,2013:5910.81,2014:6121.84,
2015:6248.51,2016:6316.07,2017:6309.68,2018:6353.55,2019:6439.96,
2020:5891.96,2021:6457.17,2022:6856.73,2023:6837.37,2024:6865.31,
}

pop = np.array([population[y] for y in years], dtype=float)
gdppc = np.array([gdp_pc[y] for y in years], dtype=float)
gdp = pop * gdppc  # constant 2010 US$

# ---------------------------------------------------------------
# 3) ENERGY USE PER CAPITA (kg oil equivalent) - APPROXIMATE
#    Anchors from World Bank / CEIC benchmark values; interpolated (PCHIP) in between.
# ---------------------------------------------------------------
energy_pc_anchors = {
1990:660, 1994:645, 1999:635, 2004:638, 2008:690, 2010:705,
2013:800, 2016:923, 2018:880, 2019:850, 2020:780, 2021:800,
2022:776, 2023:873, 2024:841,
}
ax_e = np.array(sorted(energy_pc_anchors.keys()))
ay_e = np.array([energy_pc_anchors[y] for y in ax_e])
energy_pc = PchipInterpolator(ax_e, ay_e)(years)
energy = pop * energy_pc / 1e9  # Mtoe (population * kgoe / 1e9)

# ---------------------------------------------------------------
# 4) FOSSIL CO2 EMISSIONS (Mt CO2) - APPROXIMATE
#    Anchors from World Bank/Climate Watch + Global Carbon Project (Worldometer); interpolated (PCHIP).
# ---------------------------------------------------------------
co2_anchors = {
1990:49.4, 1994:52, 1999:54, 2004:60, 2008:68, 2010:75,
2013:79, 2016:83.5, 2018:85, 2019:79.2, 2020:79.1, 2021:84.3,
2022:88.5, 2023:93, 2024:95,
}
ax_c = np.array(sorted(co2_anchors.keys()))
ay_c = np.array([co2_anchors[y] for y in ax_c])
co2 = PchipInterpolator(ax_c, ay_c)(years)

# ---------------------------------------------------------------
# KAYA IDENTITY: CO2 = Pop x (GDP/Pop) x (Energy/GDP) x (CO2/Energy)
# ---------------------------------------------------------------
gdp_per_pop = gdp / pop
energy_per_gdp = energy / gdp
co2_per_energy = co2 / energy

def growth_pct(x):
    """log-difference growth rate in %/yr"""
    g = np.full_like(x, np.nan, dtype=float)
    g[1:] = np.log(x[1:] / x[:-1]) * 100
    return g

g_pop = growth_pct(pop)
g_gdppc = growth_pct(gdp_per_pop)
g_energy_gdp = growth_pct(energy_per_gdp)
g_co2_energy = growth_pct(co2_per_energy)
g_co2_actual = np.full_like(co2, np.nan)
g_co2_actual[1:] = (co2[1:] / co2[:-1] - 1) * 100  # actual % change (matches "dot")

# sum of log-additive components (this should ~= log-growth of CO2)
g_co2_log_sum = g_pop + g_gdppc + g_energy_gdp + g_co2_energy
# interactions = difference between actual arithmetic % growth and the log-additive sum
interactions = g_co2_actual - g_co2_log_sum

# ---------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7))

plot_years = years[1:]
b_pop = g_pop[1:]
b_gdppc = g_gdppc[1:]
b_egdp = g_energy_gdp[1:]
b_ce = g_co2_energy[1:]
b_int = interactions[1:]
dots = g_co2_actual[1:]

colors = {
    'pop': '#b7d96a',
    'gdppc': '#f28ac4',
    'egdp': '#5bbf9e',
    'ce': '#8fa9d9',
    'int': '#b8b8b8',
}

bottom_pos = np.zeros(len(plot_years))
bottom_neg = np.zeros(len(plot_years))

def stack(ax, x, values, color, bottom_pos, bottom_neg, label):
    pos = np.where(values > 0, values, 0)
    neg = np.where(values < 0, values, 0)
    ax.bar(x, pos, bottom=bottom_pos, color=color, width=0.8, label=label)
    ax.bar(x, neg, bottom=bottom_neg, color=color, width=0.8)
    bottom_pos = bottom_pos + pos
    bottom_neg = bottom_neg + neg
    return bottom_pos, bottom_neg

bottom_pos, bottom_neg = stack(ax, plot_years, b_pop, colors['pop'], bottom_pos, bottom_neg, 'Población')
bottom_pos, bottom_neg = stack(ax, plot_years, b_gdppc, colors['gdppc'], bottom_pos, bottom_neg, 'PIB/Población')
bottom_pos, bottom_neg = stack(ax, plot_years, b_egdp, colors['egdp'], bottom_pos, bottom_neg, 'Energía/PIB')
bottom_pos, bottom_neg = stack(ax, plot_years, b_ce, colors['ce'], bottom_pos, bottom_neg, 'CO2/Energía')
bottom_pos, bottom_neg = stack(ax, plot_years, b_int, colors['int'], bottom_pos, bottom_neg, 'Interacciones')

ax.scatter(plot_years, dots, color='black', s=18, zorder=5, label='CO2 fósil (total)')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('%/año')
ax.set_title('Colombia — Descomposición de Kaya de las emisiones de CO2 fósil (1990–2024)', fontsize=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlim(1990.5, 2024.5)
ax.legend(loc='lower left', ncol=3, frameon=False, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.text(0.01, -0.02,
    "Nota: Población y PIB/Población son datos exactos del Banco Mundial (vía FRED, PIB en US$ constantes de 2010).\n"
    "Energía/PIB y CO2/Energía se estimaron interpolando entre valores de referencia (Banco Mundial, CEIC, Global Carbon Project); no son series anuales oficiales exactas.",
    fontsize=8, color='dimgray', ha='left')

plt.tight_layout()
plt.savefig('kaya_colombia.png', dpi=170, bbox_inches='tight')
print("saved")

# Print a small summary table for sanity check
for y, p, g, e, c, co in zip(years, pop, gdp, energy, co2, co2):
    pass

print("\nYear  Pop(M)  GDPpc(2010$)  Energy(Mtoe)  CO2(Mt)")
for y in [1990,2000,2010,2016,2020,2024]:
    i = list(years).index(y)
    print(f"{y}  {pop[i]/1e6:6.2f}  {gdppc[i]:10.1f}  {energy[i]:8.2f}  {co2[i]:6.1f}")