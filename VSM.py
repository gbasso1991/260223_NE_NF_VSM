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

vol=0.05 #mL
conc_NE_core= 13.9 #mg/mL Magnetita
conc_NE = 9.7 #mg/mL
masa_NE_core= vol*conc_NE_core/1000     #g
masa_NE=vol*conc_NE/1000    # g

data_NE_core = np.loadtxt(os.path.join('data','NE@260203b.txt'), skiprows=12)
H_NE_core = data_NE_core[:, 0]  # Gauss
m_NE_core = data_NE_core[:, 1]*1.00435414  #con correccion de Flavio
m_NE_core_norm = m_NE_core/masa_NE_core

data_NE = np.loadtxt(os.path.join('data','NECitrico@260203a.txt'), skiprows=12)
H_NE = data_NE[:, 0]  # Gauss
m_NE = data_NE[:, 1]  # emu
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

fig2, b = plt.subplots( figsize=(8, 6), constrained_layout=True)
b.plot(H_NE_core, m_NE_core_norm,'.-', label=f'NE core ({masa_NE_core:.1e} g)')
b.plot(H_NE, m_NE_norm,'.-', label=f'NE ({masa_NE:.1e} g)')
b.set_ylabel('m (emu/g)')
b.legend()
b.grid()
b.set_title('NE - Coprecipitacion - Normalizado por masa')
b.set_xlabel('H (G)')
b.set_ylabel('m (emu/g)')


axins = inset_axes(b,width="40%",     # tamaño relativo
                   height="40%",
                   bbox_to_anchor=(-0.01, -0.52, 1, 1),
                    bbox_transform=b.transAxes)  # posición

# Volver a graficar las curvas en el inset
axins.plot(H_NE_core, m_NE_core_norm,'.-')
axins.plot(H_NE, m_NE_norm,'.-')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-60, 60)   # rango eje X
axins.set_ylim(-10, 10)   # rango eje Y

axins.grid()
plt.show()
#%% NF & NF@citrico

vol=0.05 #mL
conc_NF_conc= 13 #mg/mL Magnetita
conc_NF = 1.59 #mg/mL
masa_NF_conc= vol*conc_NF_conc/1000     #g
masa_NF=vol*conc_NF/1000    # g


data_NF = np.loadtxt(os.path.join('data','NFCitrico@260203b.txt'), skiprows=12)
H_NF = data_NF[:, 0]  # Gauss
m_NF = data_NF[:, 1]  # emu
m_NF_norm = m_NF/masa_NF

data_NF_conc = np.loadtxt(os.path.join('data','NFCitrico@conc260203.txt'), skiprows=12)
H_NF_conc = data_NF_conc[:, 0]  # Gauss
m_NF_conc = data_NF_conc[:, 1]  # emu
m_NF_conc_norm = m_NF_conc/masa_NF_conc

fig3, a = plt.subplots( figsize=(8, 6), constrained_layout=True)
a.plot(H_NF, m_NF, '.-', label='NF')
a.plot(H_NF_conc, m_NF_conc, '.-', label='NF conc')
a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title('NF - Solvotermal')
a.set_xlabel('H (G)')
a.set_ylabel('M (emu)')
plt.show()
#%%
fig4, b = plt.subplots(figsize=(8, 6), constrained_layout=True)
b.plot(H_NF, m_NF_norm, '.-', label=f'NF ({masa_NF*1000:.3f} mg)')
b.plot(H_NF_conc, m_NF_conc_norm, '.-', label=f'NF conc ({masa_NF_conc*1000:.3f} mg) ')
b.set_ylabel('m (emu)')
b.legend()
b.grid()
b.set_title('NF - Solvotermal')
b.set_title('NF - Coprecipitacion - Normalizado por masa')
b.set_xlabel('H (G)')
b.set_ylabel('m (emu/g)')


axins = inset_axes(b,width="40%",     # tamaño relativo
                   height="40%",
                   bbox_to_anchor=(-0.01, -0.52, 1, 1),
                    bbox_transform=b.transAxes)  # posición

# Volver a graficar las curvas en el inset
axins.plot(H_NF_conc, m_NF_conc_norm,'.-')
axins.plot(H_NF, m_NF_norm,'.-')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-60, 60)   # rango eje X
axins.set_ylim(-10, 10)   # rango eje Y

axins.grid()
plt.show()


#%% PLOTEO NE y NF concentrada

fig, ax = plt.subplots( figsize=(8, 7), constrained_layout=True)

ax.plot(H_NE, m_NE_norm, '.-', label='NE')
ax.plot(H_NF_conc, m_NF_conc_norm, '.-', label='NF conc')
ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend()
ax.grid()
ax.set_title('NE@citrato - NF@citrato concentrada')
plt.savefig('NE_NF_comparativa_VSM.png',dpi=300)
plt.show()
#%% Normalizo por masa de la muestra y ploteo
# Concentracion_mm    = 10/1000    # g/L de particulas/densidad H20

# Concentracion_mm_F1 = 9.1/1000   # g/L de Fe3O4 / densidad H2O 
# Concentracion_mm_F2 = 6.57/1000   # g/L de Fe3O4 / densidad H2O
# Concentracion_mm_F3 = 3.35/1000  # g/L de Fe3O4 / densidad H2O
# Concentracion_mm_F4 = 3.53/1000  # g/L de Fe3O4 / densidad H2O  

# masa_C1 = (0.1121-0.0613)*Concentracion_mm  # (masa_muestra - masa sachet)*Concentracion_m/m  g
# masa_C2 = (0.1142-0.0618)*Concentracion_mm  # g
# masa_C3 = (0.1061-0.0549)*Concentracion_mm/0.48  # g
# masa_C4 = (0.1005-0.0500)*Concentracion_mm/0.43  # g

# masa_F1 = (0.1082-0.0580)*Concentracion_mm_F1  # g
# masa_F2 = (0.1194-0.0585)*Concentracion_mm_F2  # g
# masa_F3 = (0.1010-0.0489)*Concentracion_mm_F3  # g   
# masa_F4 = (0.1192-0.0691)*Concentracion_mm_F4  # g

# m_C1_norm = m_C1 / masa_C1 
# m_C2_norm = m_C2 / masa_C2 
# m_C3_norm = m_C3 / masa_C3 
# m_C4_norm = m_C4 / masa_C4 

# m_F1_norm = m_F1 / masa_F1 
# m_F2_norm = m_F2 / masa_F2 
# m_F3_norm = m_F3 / masa_F3 
# m_F4_norm = m_F4 / masa_F4 
#%% PLOTEO ALL normalizado
# fig, (a,b) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True, constrained_layout=True)

# # Arriba: muestras C
# a.plot(H_C1, m_C1_norm, '.-', label='C1')
# a.plot(H_C2, m_C2_norm, '.-', label='C2')
# a.plot(H_C3, m_C3_norm, '.-', label='C3')
# a.plot(H_C4, m_C4_norm, '.-', label='C4')
# a.set_ylabel('m (emu/g)')
# a.legend()
# a.grid()
# a.set_title('Muestras C')

# # Abajo: muestras F
# b.plot(H_F1, m_F1_norm, '.-', label='F1')
# b.plot(H_F2, m_F2_norm, '.-', label='F2')
# b.plot(H_F3, m_F3_norm, '.-', label='F3')
# b.plot(H_F4, m_F4_norm, '.-', label='F4')
# b.set_ylabel('m (emu/g)')
# b.set_xlabel('H (G)')
# b.legend()
# b.grid()
# b.set_title('Muestras F')
# plt.savefig('VSM_muestras_C_F.png', dpi=300)
# plt.show()
#%% Curvas Anhistéricas y fit para todas las muestras C  
resultados_fit = {}
H_fit_arrays = {}
m_fit_arrays = {}

for nombre, H, m  in [('NE', H_NE, m_NE_norm)]:
    H_anhist, m_anhist = mt.anhysteretic(H, m)
    fit = fit3.session(H_anhist, m_anhist, fname=nombre, divbymass=False)
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
#%% Curvas Anhistéricas y fit para todas las muestras F
for nombre, H, m ,mass in [
    ('F1', H_F1, m_F1, masa_F1), ('F2', H_F2, m_F2, masa_F2),
    ('F3', H_F3, m_F3, masa_F3), ('F4', H_F4, m_F4, masa_F4)]:
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
    resultados_fit_C[nombre]={'H_anhist': H_anhist,
                            'm_anhist': m_anhist,
                            'H_fit': fit.X,
                            'm_fit': fit.Y,
                            'm_fit_sin_lineal': m_fit_sin_lineal,
                            'linear_contrib': linear_contrib,
                            'Ms':fit.derived_parameters()['m_s'],
                            'fit': fit}
    
    H_fit_arrays_F[nombre] = fit.X
    m_fit_arrays_F[nombre] = fit.Y


#%% Ploteo VSM normalizado y fitting para cada muestra individualmente
muestras_C = [
    ('C1', H_C1, m_C1,masa_C1),
    ('C2', H_C2, m_C2,masa_C2),
    ('C3', H_C3, m_C3,masa_C3),

    ('C4', H_C4, m_C4,masa_C4),]

for idx, (nombre, H, m ,mass) in enumerate(muestras_C):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    
    ax.plot(H, m/mass, '.-', label='VSM normalizado')
    ax.plot(H_fit_arrays_C[nombre], m_fit_arrays_C[nombre], '-', label='Fitting')
    
    # Ms con error a 2 cifras significativas
    Ms = resultados_fit_C[nombre]['Ms']
    # Formateo con 2 cifras significativas en el error
    Ms_str = f"Ms = {Ms:.1uS} emu/g"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.75, 0.5, Ms_str, transform=ax.transAxes, fontsize=12,
            va='center',ha='center', bbox=props)
    
    ax.grid()
    ax.legend()
    ax.set_xlabel('H (G)')
    ax.set_ylabel('m (emu/g)')
    plt.suptitle('VSM normalizado y fitting '+nombre)
    plt.savefig(f'VSM_normalizado_vs_fit_{nombre}.png', dpi=300)
    plt.show()

#plt.savefig('VSM_normalizado_vs_fit_por_muestra.png', dpi=300)
#%% Ploteo fits 
muestras_C = [
    ('C1', H_C1, m_C1_norm),
    ('C2', H_C2, m_C2_norm),
    ('C3', H_C3, m_C3_norm),
    ('C4', H_C4, m_C4_norm),]
muestras_F = [
    ('F1', H_F1, m_F1_norm),
    ('F2', H_F2, m_F2_norm),
    ('F3', H_F3, m_F3_norm),
    ('F4', H_F4, m_F4_norm),]

fig, (a, b) = plt.subplots( 2,1, figsize=(8, 10), sharex=True, sharey=True, constrained_layout=True)

# Izquierda: muestras C
for idx, (nombre, H, m_norm) in enumerate(muestras_C):
    a.plot(H, m_norm, '.', label=nombre )
    a.plot(H_fit_arrays_C[nombre], m_fit_arrays_C[nombre], '-', label=nombre+' Fitting')

a.set_ylabel('m (emu/g)')
a.legend(ncol=2)
a.grid()
a.set_title('Muestras C')
#a.set_xlabel('H (G)')

# # Derecha: muestras F
for idx, (nombre, H, m_norm) in enumerate(muestras_F):
    b.plot(H, m_norm, '.', label=nombre )
    b.plot(H_fit_arrays_F[nombre], m_fit_arrays_F[nombre], '-', label=nombre+' Fitting')

b.set_ylabel('m (emu/g)')
b.set_xlabel('H (G)')
b.legend(ncol=2)
b.grid()
b.set_title('Muestras F')

plt.savefig('VSM_fits_C_F.png', dpi=300)
plt.show()
#%% PLOTEO ALL normalizado
fig, (a,b) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True, constrained_layout=True)

# Arriba: muestras C
a.plot(H_C1, m_C1_norm, '.-', label='C1')
a.plot(H_C2, m_C2_norm, '.-', label='C2')
a.plot(H_C3, m_C3_norm, '.-', label='C3')
a.plot(H_C4, m_C4_norm, '.-', label='C4')
a.set_ylabel('m (emu/g)')
a.legend()
a.grid()
a.set_title('Muestras C')

# Abajo: muestras F
b.plot(H_F1, m_F1_norm, '.-', label='F1')
b.plot(H_F2, m_F2_norm, '.-', label='F2')
b.plot(H_F3, m_F3_norm, '.-', label='F3')
b.plot(H_F4, m_F4_norm, '.-', label='F4')
b.set_ylabel('m (emu/g)')
b.set_xlabel('H (G)')
b.legend()
b.grid()
b.set_title('Muestras F')
plt.savefig('VSM_muestras_C_F.png', dpi=300)
plt.show()
# %%
