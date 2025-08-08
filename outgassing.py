import numpy as np

# Exercise 7.1
def calc_gas_spec(T,p,Xmelt_H2O,Xmelt_CO2,DIW):
  # T is temperature in K
  # p is pressure in bar
  # Xmelt_H2O is melt molar fraction of H2O (i.e. of H2)
  # Xmelt_CO2 is melt molar fraction of CO2 (i.e. of C)
  # DIW is Delta IW: log10 variation of oxygen fugacity
  R = 8.3145 # J/(mol K) gas constant

  # Exercise 7.2
  DfG_H2O=-241548 + 12.6844*T*np.log10(T) +  10.9782*T
  DfG_CO2=-392647 +  4.5855*T*np.log10(T) -  16.9762*T
  DfG_CO =-107052 + 12.6092*T*np.log10(T) - 131.0773*T

  log_fO2 = -27714/T + 6.899 + (0.05*(p-1))/T + DIW
  fO2 = 10**log_fO2

  ratio_XH2O_XH2 = np.sqrt(fO2 * np.exp(-2*DfG_H2O / (R*T)))
  ratio_XCO2_XCO = np.sqrt(fO2 * np.exp(-2*(DfG_CO2-DfG_CO) / (R*T)))

  # Exercise 7.3
  # XH2+XH2O=Xmelt_H2O
  # -> XH2(1+ratio)=Xmelt_H2O
  # XCO+XCO2=Xmelt_CO2
  # -> XCO(1+ratio)=Xmelt_CO2
  XH2 = Xmelt_H2O / (1+ratio_XH2O_XH2)
  XH2O = Xmelt_H2O - XH2
  XCO = Xmelt_CO2 / (1+ratio_XCO2_XCO)
  XCO2 = Xmelt_CO2 - XCO
  return np.array([XH2,XH2O,XCO,XCO2])