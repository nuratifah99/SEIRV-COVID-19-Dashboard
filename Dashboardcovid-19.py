import numpy as np
import pandas as pd
import streamlit as st
import altair as al
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint,solve_ivp
from plotly.subplots import make_subplots 
from lmfit import minimize, Parameters, report_fit


st.set_page_config(
page_title = 'COVID-19 Dashboard',
page_icon = '✅',
layout = 'wide'
)
 
title="COVID-19 in Malaysia"
subtitle="The information and visualization displayed in this website is for Final Year Project (FYP) purposes only"
st.markdown(f"<h1 style='text-align: center; color: black;'>{title}</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center; color: black;'>{subtitle}</h3>", unsafe_allow_html=True)
st.markdown("<hr/>",unsafe_allow_html=True)
    
# Import raw data
Exposed_hospital_raw_data=pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/hospital.csv")
Exposed_pkrc_raw_data=pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/pkrc.csv")
Infected_new_cases_raw_data=pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_malaysia.csv")
Infected_new_cases_import_raw_data=pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_malaysia.csv")
Removed_recovered_raw_data=pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_malaysia.csv")
Removed_Death_new_raw_data=pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/deaths_malaysia.csv")
Vaccinated_raw_data=pd.read_csv("https://raw.githubusercontent.com/CITF-Malaysia/citf-public/main/vaccination/vax_malaysia.csv")
N_Population_raw_data=pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/static/population.csv")

 
# Filter data to be used

# Exposed
df_Exposed_hospital_data=pd.DataFrame({'Date': pd.to_datetime(Exposed_hospital_raw_data['date']),'person under investigation(pui)':Exposed_hospital_raw_data['admitted_pui']})
Exposed_hospital_data_state = df_Exposed_hospital_data[(df_Exposed_hospital_data['Date'] >= '2021-02-24')]
Exposed_hospital_data=Exposed_hospital_data_state.groupby(['Date'], as_index=True).agg('sum')

df_Exposed_pkrc_data=pd.DataFrame({'Date': pd.to_datetime(Exposed_pkrc_raw_data['date']),'person under investigation(pui)':Exposed_pkrc_raw_data['admitted_pui']})
Exposed_pkrc_data_state=df_Exposed_pkrc_data[(df_Exposed_pkrc_data['Date'] >= '2021-02-24')]
Exposed_pkrc_data=Exposed_pkrc_data_state.groupby(['Date'], as_index=True).agg('sum')

Total_Exposed=Exposed_hospital_data + Exposed_pkrc_data

# Infected
df_Infected_new_cases_data=pd.DataFrame({'Date': pd.to_datetime(Infected_new_cases_raw_data['date']),'New cases':Infected_new_cases_raw_data['cases_new']})
Infected_new_cases_data=df_Infected_new_cases_data[(df_Infected_new_cases_data['Date'] >= '2021-02-24')]
Infected_new_cases=Infected_new_cases_data.groupby(['Date'], as_index=True).agg('sum')

df_Infected_new_cases_import_data=pd.DataFrame({'Date': pd.to_datetime(Infected_new_cases_import_raw_data['date']),'New cases':Infected_new_cases_import_raw_data['cases_import']})
Infected_new_cases_import_data=df_Infected_new_cases_import_data[(df_Infected_new_cases_import_data['Date'] >= '2021-02-24')]
Infected_new_cases_import=Infected_new_cases_import_data.groupby(['Date'], as_index=True).agg('sum')

Total_Infected=Infected_new_cases + Infected_new_cases_import

# Removed
df_Removed_recovered_data=pd.DataFrame({'Date': pd.to_datetime(Removed_recovered_raw_data['date']),'Removed cases':Removed_recovered_raw_data['cases_recovered']})
Removed_recovered_data=df_Removed_recovered_data[(df_Removed_recovered_data['Date'] >= '2021-02-24')]
Removed_recovered=Removed_recovered_data.groupby(['Date'], as_index=True).agg('sum')

df_Removed_Death_new_cases=pd.DataFrame({'Date': pd.to_datetime(Removed_Death_new_raw_data['date']),'Removed cases':Removed_Death_new_raw_data['deaths_new']})
Removed_Death_new_cases=df_Removed_Death_new_cases[(df_Removed_Death_new_cases['Date'] >= '2021-02-24')]
Removed_Death=Removed_Death_new_cases.groupby(['Date'], as_index=True).agg('sum')

Total_Removed=Removed_recovered + Removed_Death
# Total_Removed=Removed_recovered 


# Vaccinated
df_Vaccinated_adult=pd.DataFrame({'Date': pd.to_datetime(Vaccinated_raw_data['date']),'Vaccinated':Vaccinated_raw_data['daily_full']})
Vaccinated_adult_data=df_Vaccinated_adult[(df_Vaccinated_adult['Date'] >= '2021-02-24')]
Vaccinated_adult=Vaccinated_adult_data.groupby(['Date'], as_index=True).agg('sum')

df_Vaccinated_child=pd.DataFrame({'Date': pd.to_datetime(Vaccinated_raw_data['date']),'Vaccinated':Vaccinated_raw_data['daily_full_child']})
Vaccinated_child_data=df_Vaccinated_child[(df_Vaccinated_child['Date'] >= '2021-02-24')]
Vaccinated_child=Vaccinated_child_data.groupby(['Date'], as_index=True).agg('sum')

Total_Vaccinated=Vaccinated_adult + Vaccinated_child

# Total Population
N_Population_data=N_Population_raw_data[N_Population_raw_data['state']=='Malaysia']

def ode_model(z, t, lamda, sigma, beta, alpha, mu):
    """
    Reference Alhamami (2019)
    """

    delta = 0.016 # from Alsayed paper
    p = 0.5     # Sinovac
    eta = 0.1 # from Alsayed paper

    S, E, I, R, V = z
    N = S + E + I + R + V

    dSdt = lamda - (beta*S*I)/N - alpha*S - mu*S
    dEdt = (beta*S*I)/N + ((1-p)*beta*V*I)/N - sigma*E - mu*E
    dIdt = sigma*E - eta*I - delta*I - mu*I
    dRdt =  p*V + eta*I - mu*R
    dVdt = alpha*S - ((1-p)*beta*V*I)/N - p*V - mu*V
    return [dSdt, dEdt, dIdt, dRdt, dVdt]
    
def ode_solver(t, initial_conditions, params):
    initE, initI, initR, initN, initV = initial_conditions
    lamda, sigma, beta, alpha, mu = params['lamda'].value, params['sigma'].value, params['beta'].value, params['alpha'].value, params['mu'].value
    initS = initN - (initE + initI + initR + initV)
    t_eval=np.linspace(t[0],t[1],1)
    res = odeint(ode_model, [initS, initE, initI, initR, initV], t, args=(lamda, sigma, beta, alpha, mu))
    return res

def initial_params():
    lamda = 6.25*10**-3
    beta = 0.62*10**-8
    sigma = 1/7
    alpha = 0.5
    mu=6.25*10**-3

    params = Parameters()
    params.add('lamda', value=lamda, min=0)
    params.add('sigma', value=sigma, min=0,max=10)
    params.add('beta', value=beta, min=0,max=10)
    params.add('alpha', value=alpha, min=0,max=10)
    params.add('mu', value=mu, min=0,max=10)
    return params

def error(params, initial_conditions, tspan, data):
    sol = ode_solver(tspan, initial_conditions, params)
    return (sol[:,1:5] - data).ravel()

def mae(days,fitted_predicted_EIRV,observed_EIRV):
    E_MAE=np.mean(np.abs(fitted_predicted_EIRV[:days, 0] - observed_EIRV[:days, 0]))
    I_MAE=np.mean(np.abs(fitted_predicted_EIRV[:days, 1] - observed_EIRV[:days, 1]))
    R_MAE=np.mean(np.abs(fitted_predicted_EIRV[:days, 2] - observed_EIRV[:days, 2]))
    V_MAE=np.mean(np.abs(fitted_predicted_EIRV[:days, 3] - observed_EIRV[:days, 3]))
    return [E_MAE,I_MAE,R_MAE,V_MAE]
    
def rmse(days,fitted_predicted_EIRV,observed_EIRV):
    E_RMSE= np.sqrt(np.mean((fitted_predicted_EIRV[:days, 0] - observed_EIRV[:days, 0])**2))
    I_RMSE= np.sqrt(np.mean((fitted_predicted_EIRV[:days, 1] - observed_EIRV[:days, 1])**2))
    R_RMSE= np.sqrt(np.mean((fitted_predicted_EIRV[:days, 2] - observed_EIRV[:days, 2])**2))
    V_RMSE= np.sqrt(np.mean((fitted_predicted_EIRV[:days, 3] - observed_EIRV[:days, 3])**2))
    return [E_RMSE,I_RMSE,R_RMSE,V_RMSE] 
    
# Initial Conditions

initN =int(N_Population_data['pop'])
initE =Total_Exposed['person under investigation(pui)'].iloc[0]
initI =Total_Infected['New cases'].iloc[0]
initR =Total_Removed['Removed cases'].iloc[0]
initV =Total_Vaccinated['Vaccinated'].iloc[0]
initS = initN - (initE + initI + initR + initV)

initial_conditions = [initE, initI, initR, initN, initV]
covid_data_full=pd.concat([Total_Exposed,Total_Infected,Total_Removed,Total_Vaccinated],join='inner',axis=1)

def main(params, initial_conditions,tspan, data, covid_data, days):
    # PARAMETER ESTIMATION

    result = minimize(error, params, args=(initial_conditions, tspan, data), method='least_squares')
    params['lamda'].value = result.params['lamda'].value
    params['sigma'].value = result.params['sigma'].value
    params['beta'].value = result.params['beta'].value
    params['alpha'].value = result.params['alpha'].value
    params['mu'].value = result.params['mu'].value
        
    # SOLVE FOR FITTED DATA
        
    observed_EIRV = covid_data.values
    tspan_fit_pred = np.arange(0, observed_EIRV.shape[0], 1)
    fitted_predicted = ode_solver(tspan_fit_pred, initial_conditions, params)
    fitted_predicted_EIRV = fitted_predicted[:, 1:5]

    # CALCULATE MAE AND RMSE

    MAE= mae(days,fitted_predicted_EIRV,observed_EIRV)
    RMSE = rmse(days,fitted_predicted_EIRV,observed_EIRV)
        
    return {'rst':result,'prm':params, 'fit':fitted_predicted,'mae': MAE,'rmse': RMSE}

     
# Dropdown
graph = st.selectbox("Select number of days",('14 Days', '30 Days','100 Days', '300 Days'))

if graph == '14 Days':
    covid_data=covid_data_full.iloc[0:14]
    t=covid_data.index
    days=len(t)
    tspan = np.arange(0, days, 1)
    data=covid_data.values
    params = initial_params()
    result_all = main(params, initial_conditions,tspan, data, covid_data, days)
    result = result_all['rst']
    params = result_all['prm']
    fitted_predicted = result_all['fit']
    MAE = result_all['mae']
    RMSE = result_all['rmse']
    
elif graph == '30 Days':
    covid_data=covid_data_full.iloc[0:30]
    t=covid_data.index
    days=len(t)
    tspan = np.arange(0, days, 1)

elif graph == '100 Days':
    covid_data=covid_data_full.iloc[0:100]
    t=covid_data.index
    days=len(t)
    tspan = np.arange(0, days, 1)
    
elif graph == '300 Days':
    covid_data=covid_data_full.iloc[0:300]
    t=covid_data.index
    days=len(t)
    tspan = np.arange(0, days, 1) 

data=covid_data.values
params = initial_params()
result_all = main(params, initial_conditions,tspan, data, covid_data, days)
result = result_all['rst']
params = result_all['prm']
fitted_predicted = result_all['fit']
MAE = result_all['mae']
RMSE = result_all['rmse']
# Plot Observed vs Fitted data

final = data + result.residual.reshape(data.shape)

fig_data_E = go.Figure()

fig_data_E.add_trace(go.Scatter(x=t, y=data[:, 0], mode='markers', name='Observed', line = dict(dash='dot')))

fig_data_E.add_trace(go.Scatter(x=t, y=final[:, 0], mode='lines',marker = {'color' : 'black'}, name='Fitted', line = dict(dash='dot')))

fig_data_E.update_layout(title='Exposed',xaxis_title='Date',yaxis_title='Population',title_x=0.5,width=400,height=300)
    
fig_data_I = go.Figure()

fig_data_I.add_trace(go.Scatter(x=t, y=data[:, 1], mode='markers', name='Observed', line = dict(dash='dot')))
    
fig_data_I.add_trace(go.Scatter(x=t, y=final[:, 1], mode='lines', marker = {'color' : 'red'}, name='Fitted', line = dict(dash='dot')))

fig_data_I.update_layout(title='Infected',xaxis_title='Date',yaxis_title='Population',title_x=0.5,width=400,height=300)

fig_data_R = go.Figure()

fig_data_R.add_trace(go.Scatter(x=t, y=data[:, 2], mode='markers', name='Observed', line = dict(dash='dot')))

fig_data_R.add_trace(go.Scatter(x=t, y=final[:, 2], mode='lines',marker = {'color' : 'green'}, name='Fitted', line = dict(dash='dot')))

fig_data_R.update_layout(title='Removed',xaxis_title='Date',yaxis_title='Population',title_x=0.5,width=400,height=300)

fig_data_V = go.Figure()

fig_data_V.add_trace(go.Scatter(x=t, y=data[:, 3], mode='markers', name='Observed', line = dict(dash='dot')))

fig_data_V.add_trace(go.Scatter(x=t, y=final[:, 3], mode='lines', marker = {'color' : 'purple'}, name='Fitted', line = dict(dash='dot')))

fig_data_V.update_layout(title='Vaccinated',xaxis_title='Date',yaxis_title='Population',title_x=0.5,width=400,height=300)

# Plot SEIRV

# t=covid_data.index
S =fitted_predicted[:, 0]
E =fitted_predicted[:, 1]
I =fitted_predicted[:, 2]
R =fitted_predicted[:, 3]
V =fitted_predicted[:, 4]

fig_model=go.Figure()
# fig_model.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Susceptible' ))
fig_model.add_trace(go.Scatter(x=t, y=E, mode='lines', name='Exposed'))
fig_model.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infected'))
fig_model.add_trace(go.Scatter(x=t, y=R, mode='lines',name='Removed'))
fig_model.add_trace(go.Scatter(x=t, y=V, mode='lines', name='Vaccinated'))
fig_model.update_layout(title='SEIRV Model',xaxis_title='Day',yaxis_title='Population',title_x=0.5,width=700, height=700)
fig_model.update_layout(legend=dict(x=.77, y=.97,traceorder="normal", font=dict(family="Times New Roman", size=14, color="Black"),))
fig_model.update_layout(legend={'itemsizing': 'constant'})
st.markdown("<hr/>",unsafe_allow_html=True)    
# First Row
st.write('**Daily Statistic**')

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    delta = 0.016 # from Alsayed paper
    p = 0.5     # Sinovac
    eta = 0.1 # from Alsayed paper
    lamda = params['lamda'].value
    sigma = params['sigma'].value
    beta = params['beta'].value
    alpha = params['alpha'].value
    mu = params['mu'].value
    R0 = (sigma*(beta*mu*(p+mu)+(1-p)*beta*alpha*mu))/((mu+sigma)*(eta+mu+delta)*(p+mu)*(alpha+mu))
    Rnaught = "{:.2f}".format(R0)
    st.markdown(f"<h3 style='text-align: center; color: blue;'>{Rnaught}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: blue;'>{'R-Naught'}</h4>", unsafe_allow_html=True)
    
with kpi2:
    Exposed = int(E[-1])
    st.markdown(f"<h3 style='text-align: center; color: black;'>{Exposed}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: black;'>{'Exposed cases'}</h4>", unsafe_allow_html=True)
    
with kpi3:
    Infected = int(I[-1])
    st.markdown(f"<h3 style='text-align: center; color: red;'>{Infected}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: red;'>{'Confirmed cases'}</h4>", unsafe_allow_html=True)

with kpi4:
    Recovered = int(R[-1]) 
    st.markdown(f"<h3 style='text-align: center; color: green;'>{Recovered}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: green;'>{'Recovered cases'}</h4>", unsafe_allow_html=True)
    
with kpi5:
    Vaccinated = int(V[-1]) 
    st.markdown(f"<h3 style='text-align: center; color: purple;'>{Vaccinated}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: purple;'>{'Vaccinated'}</h4>", unsafe_allow_html=True)

st.markdown("<hr/>",unsafe_allow_html=True)

# Second Row
st.write('**Observed data vs Fitted data**')

chart1, chart2 = st.columns(2)

with chart1:
    st.plotly_chart(fig_data_E, use_container_width=True)
    st.plotly_chart(fig_data_I, use_container_width=True)
    
with chart2:
    st.plotly_chart(fig_data_R, use_container_width=True)
    st.plotly_chart(fig_data_V, use_container_width=True)
    
st.markdown("<hr/>",unsafe_allow_html=True)

# Third Row
chart01, chart02, = st.columns(2)

with chart01:
    
    MAE_RMSE = {'Population': ['Exposed','Infected','Removed','Vaccinated']}
    MAE_RMSE =pd.DataFrame(MAE_RMSE)
    MAE_RMSE['Fitted Mean Absolute Error(MAE)']= MAE
    MAE_RMSE['Fitted Root Mean Square Error(RMSE)']= RMSE
    
    st.write('**Error between Observed data vs Fitted data**')
    st.table(MAE_RMSE)
    
    PARAMS={'Parameter':['λ','σ','β','α','μ','δ','p','η']}
    PARAMS=pd.DataFrame(PARAMS)
    desc=['Recruitment rate of susceptible','Progression rate from Exposed (E) to Infected (I)','COVID-19 Infection rate','Vaccination rate','Natural mortality rate','Mortality rate due to COVID-19','Vaccination success rate','Recovery rate from COVID-19']
    value=[params['lamda'].value,params['sigma'].value,params['beta'].value,params['alpha'].value,params['mu'].value,0.016,0.5,0.1]
    PARAMS['Description']=desc
    PARAMS['Value']=value
        
    st.write('**Parameters value**')
    hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """ 
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(PARAMS)

with chart02:
        st.plotly_chart(fig_model, use_container_width=False)

st.markdown("<hr/>",unsafe_allow_html=True)

st.write("Click this [repo](https://github.com/nuratifah99/SEIRV-COVID-19-Dashboard/blob/a6f493019edff27d5aab2088a7c9707c7dec4ce1/Dashboardcovid-19.py) to view full code")

st.markdown("References:  \n\n" 
            "   1. SEIRV Model: Alhamami, H. (2019). ProQuest Dissertations & Theses Global.  \n"
            "   2. Parameter estimation: Alsayed, A., et al. (2020). International Journal of Environmental Research and Public Health.  \n"
            "   3. Parameter estimation: https://github.com/silpara/simulators/tree/master/compartmental_models   \n"
            "   4. COVIDNOW in Malaysia: https://covidnow.moh.gov.my/   \n"
            "   5. Malaysia's Ministry of Health: https://github.com/MoH-Malaysia/covid19-public/tree/main/epidemic   \n"
            "   6. COVID-19 Immunisation Task Force: https://github.com/CITF-Malaysia/citf-public   \n")
 
st.markdown("\n\n"
            "&#169 2021 Nur Atifah Baharuddin and Nurul Farahain Mohammad") 
