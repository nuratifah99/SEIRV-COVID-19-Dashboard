import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

st.title("SEIRV Model for COVID-19")
st.sidebar.markdown("Choose the parameters values")
subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


# Slider for total population

N=st.sidebar.slider("Total Population, N: ", min_value=100000, max_value=10000000, step=1)

# Slider for initial conditions

E0=st.sidebar.slider("Initial Exposed Population,  E0: ".translate(subscript), min_value=0, max_value=int(0.5*N), step=1)

I0=0
R0=0
V0=0
S0=N-E0-I0-R0-V0


# Slider for parameters

# lamda=st.sidebar.slider("Recruitment rate of susceptible, λ: ", min_value=0, max_value=int(0.5*S0), step=1)

lamda=0

mu=st.sidebar.slider("Natural mortality rate, μ: ", min_value=0.0, max_value=0.2, step=0.001)

delta=st.sidebar.slider("Mortality rate due to COVID-19, δ: ", min_value=0.0, max_value=0.5, step=0.001)

sigma=st.sidebar.slider("Progression rate from Exposed (E) to Infected (I), σ: ", min_value=0.0, max_value=0.3, step=0.001)

R0=st.sidebar.slider("R-Naught, R0: ".translate(subscript), min_value=0.5, max_value=4.0, step=0.001)

eta=st.sidebar.slider("Recovery rate from COVID-19, η: ", min_value=0.1, max_value=0.9, step=0.001)

beta=R0*eta

p=st.sidebar.slider("Vaccination success rate, p: ", min_value=0.1, max_value=0.95, step=0.001)

alpha=st.sidebar.slider("Vaccination rate, α: ", min_value=0.0, max_value=0.8, step=0.001)


# SEIRV Model

def f(t,y):
    
    S=y[0]
    E=y[1]
    I=y[2]
    R=y[3]
    V=y[4]
    
    dSdt=              lamda - (beta*S*I)/N - alpha*S - mu*S
    dEdt= (beta*S*I)/N + ((1-p)*beta*V*I)/N - sigma*E - mu*E
    dIdt=                   sigma*E - eta*I - delta*I - mu*I
    dRdt=                                 p*V + eta*I - mu*R
    dVdt=          alpha*S - ((1-p)*beta*V*I)/N - p*V - mu*V
    
    return np.array([dSdt,dEdt,dIdt,dRdt,dVdt])

graph = st.selectbox("Select number of days",('14 days', '100 days', '365 days'))

if graph == '14 days':
    t_span=np.array([0,14])
elif graph == '100 days':
    t_span=np.array([0,100])
elif graph == '365 days':
    t_span=np.array([0,365])

t_eval=np.linspace(t_span[0],t_span[1])
y0=np.array([S0,E0,I0,R0,V0])
sol=solve_ivp(f,t_span,y0,method='RK45',t_eval=t_eval)

# Display plotting

fig=go.Figure()
fig.update_layout(title='COVID-19 prediction based on SEIRV Model',xaxis_title='Days',yaxis_title='Number of population')
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0],mode='lines', name='Susceptible'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1],mode='lines', name='Exposed'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2],mode='lines',name='Infected'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3],mode='lines',name='Removed'))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[4],mode='lines',name='Vaccinated'))
st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("Remarks: The SEIRV Model used for the dashboard was referred from A susceptible-exposed-infected-recovered-vaccinated
(SEIRV) mathematical model of measles in Madagascar by Alhamami, H.(2019). Meanwhile,for the range of parameters, they were estimated based on Singapore’s pandemic preparedness: An overview of the first wave of covid-19 paper by Tan, J., B. (2021)")

