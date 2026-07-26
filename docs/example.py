### libraries ###

import polars as pl
import numpy as np
import lifeflow as lf

### Import data and hypotheses ###

df = pl.read_parquet("data/example_lifeflow.parquet")
df_m = df.filter(pl.col("sexo") == "H") #shared hipotheses male
df_f = df.filter(pl.col("sexo") == "M") #shared hipoteses female

qx_m = pl.read_csv("data/hypotheses/qx_TEST_M.csv")["qx"].to_numpy()
qx_m = np.repeat(1 - (1-qx_m)**(1/12), 12)

qx_f = pl.read_csv("data/hypotheses/qx_TEST_F.csv")["qx"].to_numpy()
qx_f = np.repeat(1-(1-qx_f)**(1/12), 12)

##### LifeFlow - df_m #####

### Objects ###

pfl = lf.Portfolio(df_m, id_col="poliza_id") 
tl = lf.Timeline("duration_end")
qx = lf.Decrement(qx_m, tl, index_var="age")

prob = lf.Probabilites(pfl, [qx])
tpx = prob.inforce
exit_death = prob.exit_by(qx)

tpx.shape
exit_death.shape

### Flows ###

@lf.grid(pfl, tl, n_jobs=1, jit=True)
def discount(t, tipo_interes_tecnico):
    r = (1 + tipo_interes_tecnico)**(1/12) - 1
    return (1 + r)**(-t)    

discount_grid = discount()
discount_grid

@lf.grid(pfl, tl, n_jobs=1, jit=True)
def capital_sup(t, capital_supervivencia_inicial, tasa_revalorizacion_supervivencia, duration_if, duration_end):
    if t != duration_end:
        return 0.0
    year = (duration_if + t - 1) // 12
    return capital_supervivencia_inicial * (1 + tasa_revalorizacion_supervivencia) ** year

cap_sup = capital_sup()
cap_sup.shape

@lf.grid(pfl, tl, n_jobs=1, jit=True)
def capital_death(t, capital_fallecimiento_inicial, tasa_revalorizacion_fallecimiento, duration_if):
    year = (duration_if + t - 1) // 12
    return capital_fallecimiento_inicial * (1 + tasa_revalorizacion_fallecimiento) ** year

cap_death = capital_death()

@lf.grid(pfl, tl, payable="pre", n_jobs=1, jit=True)
def prima(t, prima_anual_inicial, tasa_revalorizacion_prima, frecuencia_pago, duration_if):
    periodo = 12 // frecuencia_pago
    if (duration_if + t) % periodo != 0:
        return 0.0
    year = (duration_if + t - 1) // 12
    return prima_anual_inicial * (1 + tasa_revalorizacion_prima) ** year / frecuencia_pago

premium = prima()
premium

@lf.grid(pfl, tl, n_jobs=1, jit=True)
def gasto(t, capital_supervivencia_inicial, tasa_revalorizacion_supervivencia, tasa_gasto_sobre_capital, duration_if):
    year = (duration_if + t - 1) // 12
    capital = capital_supervivencia_inicial * (1 + tasa_revalorizacion_supervivencia) ** year
    return capital * ((1 + tasa_gasto_sobre_capital) ** (1/12) - 1)

expenses = gasto()
expenses

cap_sup.shape
cap_death.shape
premium.shape
expenses.shape

sup_flow = cap_sup * tpx; sup_flow
death_flow = cap_death * exit_death; death_flow
premium_flow = premium * tpx; premium_flow
expenses_flow = expenses * tpx; expenses_flow

provision_flow = sup_flow + death_flow + expenses_flow - premium_flow; provision_flow
vaa_provision = provision_flow * discount_grid; vaa_provision

# VAA provision per policy
print(vaa_provision.sum(axis = 1))

# VAA portfolio total flow
print(vaa_provision.sum(axis = 0))

# VAA total portfolio present value actuarial provision
print(vaa_provision.sum())
