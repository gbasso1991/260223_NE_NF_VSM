#%% VSM NE@citrato 260203 & NF@citrato 260203 Febrero 2026
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 
from mlognormfit import fit3
from mvshtools import mvshtools as mt
import re
from uncertainties import ufloat
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
#%% Funciones
def lineal(x,m,n):
    return m*x+n

def coercive_field(H, M):
    """
    Devuelve los valores de campo coercitivo (Hc) donde la magnetización M cruza por cero.
    
    Parámetros:
    - H: np.array, campo magnético (en A/m o kA/m)
    - M: np.array, magnetización (en emu/g)
    
    Retorna:
    - hc_values: list de valores Hc (puede haber más de uno si hay múltiples cruces por cero)
    """
    H = np.asarray(H)
    M = np.asarray(M)
    hc_values = []

    for i in range(len(M)-1):
        if M[i]*M[i+1] < 0:  # Cambio de signo indica cruce por cero
            # Interpolación lineal entre (H[i], M[i]) y (H[i+1], M[i+1])
            h1, h2 = H[i], H[i+1]
            m1, m2 = M[i], M[i+1]
            hc = h1 - m1 * (h2 - h1) / (m2 - m1)
            hc_values.append(hc)

    return hc_values
#%% NE & NE@citrico
data_NE_core = np.loadtxt(os.path.join('data','NE@260203b.txt'), skiprows=12)
H_NE_core = data_NE_core[:, 0]  # Gauss
m_NE_core = data_NE_core[:, 1]*1.00435414  #con correccion de Flavio

conc_NE_core = 13.9 #mg/mL Magnetita

conc_NE_core_mm = conc_NE_core/1000 #mg np/mg de solvente
masa_NE_core = (0.1172-0.0666)*conc_NE_core_mm
m_NE_core_norm = m_NE_core/masa_NE_core #emu/g

data_NE = np.loadtxt(os.path.join('data','NECitrico@260203a.txt'), skiprows=12)
H_NE = data_NE[:, 0]  # Gauss
m_NE = data_NE[:, 1]  # emu

conc_NE = 9.7 #mg/mL

conc_NE_mm = conc_NE/1000
masa_NE = (0.1111-0.0602)*conc_NE_mm
m_NE_norm = m_NE/masa_NE

fig, a = plt.subplots( figsize=(8, 6), constrained_layout=True)
a.plot(H_NE_core, m_NE_core, '.-', label='NE core')
a.plot(H_NE, m_NE, '.-', label='NE')
a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title('NE - Coprecipitacion')
a.set_xlabel('H (G)')
a.set_ylabel('m (emu)')
plt.show()
#%% NE Normalizadas por masa

fig2, b = plt.subplots( figsize=(8, 6), constrained_layout=True)

b.plot(H_NE_core, m_NE_core_norm,'.-', label=f'NE core (norm con m = {masa_NE_core:.1e} g)')
b.plot(H_NE, m_NE_norm,'.-', label=f'NE@citrato (norm con m = {masa_NE:.1e} g)')

#b.plot(H_NE, m_NE_norm,'.-', label=f'NE ({masa_NE:.1e} g)')
b.set_ylabel('m (emu/g)')
b.legend()
b.grid()
b.set_title('NE - Coprecipitacion - Normalizado por masa')
b.set_xlabel('H (G)')
b.set_ylabel('m (emu/g)')

axins = inset_axes(b,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=b.transAxes,
    borderpad=0) 


# Volver a graficar las curvas en el inset
axins.plot(H_NE_core, m_NE_core_norm,'.-')
axins.plot(H_NE, m_NE_norm,'.-')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-20,20)   # rango eje X
axins.set_ylim(-4, 4)   # rango eje Y

axins.grid()
mark_inset(b, axins, loc1=1, loc2=3, fc="none", ec="gray",zorder=4)
plt.savefig('NE_NE@citrato.png',dpi=300)
plt.show()
#%% NF & NF@citrico
data_NF = np.loadtxt(os.path.join('data','NFCitrico@260203b.txt'), skiprows=12)
H_NF = data_NF[:, 0]  # Gauss
m_NF = data_NF[:, 1]  # emu

data_NF_conc = np.loadtxt(os.path.join('data','NFCitrico@conc260203.txt'), skiprows=12)
H_NF_conc = data_NF_conc[:, 0]  # Gauss
m_NF_conc = data_NF_conc[:, 1]  # emu

conc_NF = 1.59 #mg/mL
conc_NF_mm = conc_NF/1000 #mg np/mg de solvente 
masa_NF = (0.1148-0.0652)*conc_NF_mm  # (masa_muestra - masa sachet)/Concentracion_m/m  g
m_NF_norm = m_NF/masa_NF

conc_NF_conc = 13 #mg/mL Magnetita
conc_NF_conc_mm = conc_NF_conc/1000 #mg/mL de np/densidad del solvente 
masa_NF_conc = (0.1401-0.0900)*conc_NF_conc_mm  # g (masa_muestra - masa sachet)*C_mm =  mFF*mNP/mFF 
m_NF_conc_norm = m_NF_conc/masa_NF_conc

# PLOTEO crudos
fig3, a = plt.subplots( figsize=(8, 6), constrained_layout=True)
a.plot(H_NF, m_NF, '.-', label=f'NF ({conc_NF} mg/mL)')
a.plot(H_NF_conc, m_NF_conc, '.-', label=f'NF conc ({conc_NF_conc} mg/mL)')
a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title('NF - Solvotermal')
a.set_xlabel('H (G)')
a.set_ylabel('M (emu)')
plt.show()
#%% ploteo normalizados NF
fig4, b = plt.subplots(figsize=(8, 6), constrained_layout=True)
b.plot(H_NF, m_NF_norm, '.-', label=f'NF@citrato (norm con m = {masa_NF*1000:.3f} mg)')
b.plot(H_NF_conc, m_NF_conc_norm, '.-', label=f'NF@citrato concentrada (norm con m = {masa_NF_conc*1000:.3f} mg)')

#b.plot(H_NF_conc, m_NF_conc_norm, '.-', label=f'NF conc ({masa_NF_conc*1000:.3f} mg) ')
b.set_ylabel('m (emu)')
b.legend()
b.grid()
b.set_title('NF@citrico - Solvotermal - Normalizado por masa')
b.set_xlabel('H (G)')
b.set_ylabel('m (emu/g)')

axins = inset_axes(b,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=b.transAxes,
    borderpad=0) 

# Volver a graficar las curvas en el inset
axins.plot(H_NF, m_NF_norm,'.-')
axins.plot(H_NF_conc, m_NF_conc_norm,'.-')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-60, 60)   # rango eje X
axins.set_ylim(-10, 10)   # rango eje Y

axins.grid()
plt.savefig('NF_NFconcentrada.png',dpi=300)
plt.show()

#%% PLOTEO NE y NF concentrada ambas normalizadas

fig, ax = plt.subplots( figsize=(8, 6), constrained_layout=True)

ax.plot(H_NE, m_NE_norm, '.-', label=f'NE\nC = {conc_NE} mg/mL\nm = {masa_NE*1000:.3f} mg')
ax.plot(H_NF_conc, m_NF_conc_norm, '.-', label=f'NF conc\nC = {conc_NF_conc} mg/mL\nm = {masa_NF_conc*1000:.3f} mg')
ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend(ncol=2)
ax.grid()
ax.set_title('NE@citrato - NF@citrato concentrada')

axins = inset_axes(    ax,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=ax.transAxes,
    borderpad=0) 

# Volver a graficar las curvas en el inset
axins.plot(H_NE, m_NE_norm,'.-')
axins.plot(H_NF_conc, m_NF_conc_norm,'.-')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-20, 20)   # rango eje X
axins.set_ylim(-8, 8)   # rango eje Y
axins.grid()

plt.savefig('NE_NF_comparativa_VSM.png',dpi=300)
plt.show()
#%% Curvas Anhistéricas y fit para todas las muestras NE  

# H_anhist, m_anhist = mt.anhysteretic(H_NE, m_NE)
# fit = fit3.session(H_anhist, m_anhist, fname='NE', divbymass=True,mass=masa_NE)
# fit.fix('sig0')
# fit.fix('mu0')
# fit.free('dc')
# fit.fit()
# fit.update()
# fit.free('sig0')
# fit.free('mu0')
# # fit.set_yE_as('sep')
# fit.fit()
# fit.update()
# fit.save()
# fit.print_pars()
# # Obtengo la contribución lineal usando los parámetros del fit
# C = fit.params['C'].value
# dc = fit.params['dc'].value
# linear_contrib = lineal(fit.X, C, dc)
# m_fit_sin_lineal = fit.Y - linear_contrib

# H_fit_NE = fit.X
# m_fit_NE_norm = m_fit_sin_lineal

# #%% Curvas Anhistéricas y fit para todas las muestras NF  

# H_anhist, m_anhist = mt.anhysteretic(H_NF, m_NF)
# fit = fit3.session(H_anhist, m_anhist, fname='NF', divbymass=True,mass=masa_NF)
# fit.fix('sig0')
# fit.fix('mu0')
# fit.free('dc')
# fit.fit()
# fit.update()
# fit.free('sig0')
# fit.free('mu0')
# fit.set_yE_as('sep')
# fit.fit()
# fit.update()
# fit.save()
# fit.print_pars()
# # Obtengo la contribución lineal usando los parámetros del fit
# C = fit.params['C'].value
# dc = fit.params['dc'].value
# linear_contrib = lineal(fit.X, C, dc)
# m_fit_sin_lineal = fit.Y - linear_contrib

# H_fit_NF = fit.X
# m_fit_NF_norm = m_fit_sin_lineal
    
#%% Fitting para cada muestra individualmente

resultados_fit = {}
H_fit_arrays = {}
m_fit_arrays = {}

for nombre, H, m ,mass in [('NE', H_NE, m_NE, masa_NE)]:
    H_anhist, m_anhist = mt.anhysteretic(H, m)
    fit = fit3.session(H_anhist, m_anhist, fname=nombre, divbymass=True,mass=mass)
    fit.fix('sig0')
    fit.fix('mu0')
    fit.free('dc')
    fit.fit()
    fit.update()
    fit.free('sig0')
    fit.free('mu0')
    # fit.set_yE_as('sep')
    fit.fit()
    fit.update()
    fit.save()
    fit.print_pars()
    # Obtengo la contribución lineal usando los parámetros del fit
    C = fit.params['C'].value
    dc = fit.params['dc'].value
    linear_contrib = lineal(fit.X, C, dc)
    m_fit_sin_lineal = fit.Y - linear_contrib
    resultados_fit[nombre]={'H_anhist': H_anhist,
                            'm_anhist': m_anhist,
                            'H_fit': fit.X,
                            'm_fit': fit.Y,
                            'm_fit_sin_lineal': m_fit_sin_lineal,
                            'linear_contrib': linear_contrib,
                            'Ms':fit.derived_parameters()['m_s'],
                            'fit': fit}
    H_fit_arrays[nombre] = fit.X
    m_fit_arrays[nombre] = fit.Y

#%
for nombre, H, m ,mass in [('NF', H_NF, m_NF_conc, masa_NF_conc)]:
    H_anhist, m_anhist = mt.anhysteretic(H, m)
    fit = fit3.session(H_anhist, m_anhist, fname=nombre, divbymass=True,mass=mass)
    fit.fix('sig0')
    fit.fix('mu0')
    fit.free('dc')
    fit.fit()
    fit.update()
    fit.free('sig0')
    fit.free('mu0')
    fit.set_yE_as('sep')
    fit.fit()
    fit.update()
    fit.save()
    fit.print_pars()
    # Obtengo la contribución lineal usando los parámetros del fit
    C = fit.params['C'].value
    dc = fit.params['dc'].value
    linear_contrib = lineal(fit.X, C, dc)
    m_fit_sin_lineal = fit.Y - linear_contrib
    resultados_fit[nombre]={'H_anhist': H_anhist,
                            'm_anhist': m_anhist,
                            'H_fit': fit.X,
                            'm_fit': fit.Y,
                            'm_fit_sin_lineal': m_fit_sin_lineal,
                            'linear_contrib': linear_contrib,
                            'Ms':fit.derived_parameters()['m_s'],
                            'fit': fit}
    H_fit_arrays[nombre] = fit.X
    m_fit_arrays[nombre] = fit.Y


#%% Ploteo los fits
fig, ax = plt.subplots( figsize=(8, 6), constrained_layout=True)

ax.plot(H_NE, m_NE_norm,'o-', c='C0',alpha=0.4,label=f'Datos originales\n')

ax.plot(resultados_fit['NE']['H_fit'], resultados_fit['NE']['m_fit'],
        '.-', c='C1',label=f'NE fit\nC = {conc_NE} mg/mL\nm = {masa_NE*1000:.3f} mg\nMs = {resultados_fit["NE"]["Ms"]:.1uS} emu/g\n')


ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend(ncol=1)
ax.grid()
ax.set_title('NE@citrato ')

axins = inset_axes(    ax,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=ax.transAxes,
    borderpad=0) 

# Volver a graficar las curvas en el inset
axins.plot(H_NE, m_NE_norm,'o-', c='C0',alpha=0.4)
axins.plot(resultados_fit['NE']['H_fit'], resultados_fit['NE']['m_fit'],'.-',c='C1')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-20, 20)   # rango eje X
axins.set_ylim(-8, 8)   # rango eje Y
axins.grid()

plt.savefig('NE_fit.png',dpi=300)
plt.show()

fig2, ax = plt.subplots( figsize=(8, 6), constrained_layout=True)

ax.plot(H_NF, m_NF_conc_norm,'o-', c='C0',alpha=0.4,label=f'Datos originales\n')
ax.plot(resultados_fit['NF']['H_fit'], resultados_fit['NF']['m_fit'],
        '.-', c='C1',label=f'NF fit\nC = {conc_NF} mg/mL\nm = {masa_NF*1000:.3f} mg\nMs = {resultados_fit["NF"]["Ms"]:.1uS} emu/g\n')


ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend(ncol=1)
ax.grid()
ax.set_title('NF@citrato ')

axins = inset_axes(    ax,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=ax.transAxes,
    borderpad=0) 

# Volver a graficar las curvas en el inset
axins.plot(H_NF, m_NF_norm,'o-', c='C0',alpha=0.4)
axins.plot(resultados_fit['NF']['H_fit'], resultados_fit['NF']['m_fit'],'.-',c='C1')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-20, 20)   # rango eje X
axins.set_ylim(-8, 8)   # rango eje Y
axins.grid()

plt.savefig('NF_fit.png',dpi=300)
plt.show()
#%% Comparo fits NE y NF
fig, ax = plt.subplots( figsize=(8, 6), constrained_layout=True)

ax.plot(resultados_fit['NE']['H_fit'], resultados_fit['NE']['m_fit_sin_lineal'],
        '.-', label=f'NE\nC = {conc_NE} mg/mL\nm = {masa_NE*1000:.3f} mg\nMs = {resultados_fit["NE"]["Ms"]:.1uS} emu/g\n')

ax.plot(resultados_fit['NF']['H_fit'], resultados_fit['NF']['m_fit_sin_lineal'],
        '.-', label=f'NF\nC = {conc_NF} mg/mL\nm = {masa_NF*1000:.3f} mg\nMs = {resultados_fit["NF"]["Ms"]:.1uS} emu/g\n')

ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend(ncol=1)
ax.grid()
ax.set_title('NE@citrato & NF@citrato concentrada')

axins = inset_axes(ax,width="40%",height="40%",
    loc='lower right',bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=ax.transAxes,borderpad=0) 

axins.plot(resultados_fit['NE']['H_fit'], resultados_fit['NE']['m_fit_sin_lineal'],'.-')
axins.plot(resultados_fit['NF']['H_fit'], resultados_fit['NF']['m_fit_sin_lineal'],'.-')

axins.set_xlim(-20, 20)   # rango eje X
axins.set_ylim(-8, 8)   # rango eje Y
axins.grid()

plt.savefig('NE_NF_comparativa_fits.png',dpi=300)
plt.show()
#%%