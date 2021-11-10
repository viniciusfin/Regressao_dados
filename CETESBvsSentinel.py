# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Importando os pacotes
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from statsmodels.graphics.api import abline_plot
import glob
import os
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score


tabela_global = []
coefs_LM = []
coefs_GLM = []
media_R2LM =[]
media_RMSGLM =[]

# Leitura dos dados - Colocar todos na mesma pasta
poluente = "CO"
folder = 'E:/Meus Documentos/Área de Trabalho/REG_daily/'+poluente+"_diario/"
os.chdir(folder)
extension = 'csv'
files = [i for i in glob.glob('*.{}'.format(extension))]
for file in files:
    
    

    data = pd.read_csv(folder+'/'+file)
    name_title = file
    
    pd.set_option('use_inf_as_na', True) #setado para usar inf como NaN
  
    
    
    #%% General linear model # ------------------------------ ALTERAR AQUI O POLUENTE -------------------------------------------------------------
    poluente2 = poluente+"_diario"
    aplicar_log = "sim"
    
    if aplicar_log == "sim":
        Xlog = "log"
    else:
        Xlog=""
    if poluente2 == poluente+"_mensal":
        freq = "_month"
    if poluente2 == poluente+"_diario":
        freq = ""
    if poluente2 == poluente+"_semanal":
        freq = "_week"
    if poluente == "CO":
        alf = 0.6
    else:
        alf = 0.6
    if poluente == "O3":
        cor = "black"
    if poluente == "NO2":
        cor = "green"
    if poluente == "CO":
        cor = "#386DB4"
    if poluente == "MP":
        cor = "red"
        
        
    
    #Removendo outliuers
    # Abrindo arquivo
    data = pd.read_csv(folder+file)
    iqr5Sentinel = np.nanpercentile(data['Sentinel'],5)
    iqr95Sentinel = np.nanpercentile(data['Sentinel'],95)
    iqr5CETESB = np.nanpercentile(data['CETESB'],5)
    iqr95CETESB = np.nanpercentile(data['CETESB'],95)
    data2=data
    data2['CETESB'][data2['CETESB']<iqr5CETESB] = np.nan
    data2['CETESB'][data2['CETESB']>iqr95CETESB] = np.nan
    data2['Sentinel'][data2['Sentinel']<iqr5Sentinel] = np.nan
    data2['Sentinel'][data2['Sentinel']>iqr95Sentinel] = np.nan
    
    data2 =data2.dropna() #removendo os NaN
    data2['CETESB'] = np.log10(data2['CETESB'])
    data2['Sentinel'] = np.log10(data2['Sentinel'])
    X = data2['Sentinel'].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data2['CETESB'].values.reshape(-1, 1) 
    
    modelX = sm.GLM(Y, X, family=sm.families.Gaussian())
    res = modelX.fit()
    print(res.summary())
    yhat = res.mu
    print('Parametros_GLM:',  res.params)
    rms = sqrt (mean_squared_error(yhat, Y)) #root mean square
    print('RMS_GLM:',  rms)
    r2 = r2_score(Y, yhat) # (coefficient of determination) regression score function (true, predit)
    print('R2_GLM:', r2)
    print('STD_ERROR_GLM:', res.bse)
    resid = res.resid_deviance.copy()
    resid_std = stats.zscore(resid)
    resid2 = resid**2
    rms_log = sqrt(mean_squared_error(10**yhat, 10**Y))
    
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)
    Y_pred = linear_regressor.predict(X)
    linear_regressor.score(X,Y)
    resid_LM = (Y - Y_pred)
    shapiro_stat_LM, shapiro_p_valor_LM = stats.shapiro(resid_LM)
    resid_antlog = ((10**Y)-(10**Y_pred))
    resid_antlog2 = resid_antlog**2

    
    resid_LM2 = resid_LM**2
    divisor_desvio = len(X)-2
    divisao = np.sum(resid_LM2)/divisor_desvio
    erro_std_LM = sqrt(divisao)
    R2_LM = r2_score(Y, Y_pred)
    
    
    
    coefReg=[]
    coefsReg = pd.DataFrame()
    coefs_LM+=[linear_regressor.coef_]
    coefs_GLM+=[res.params]
    media_R2LM+=[R2_LM]
    media_RMSGLM+=[rms]
    media_LM_LOG_RMSE = [round(sqrt(sum(resid_antlog2)/len(resid_antlog2)),2)]
    media_GLM_LOG_RMSE = [round(rms_log,2)]
    media_LM_RMSE = [round(sqrt(sum(resid_LM2)/len(resid_LM2)),2)]
    media_GLM_RMSE = [round(sqrt(sum(resid2)/len(resid2)),2)]
    
    num_estacao=["Americana", "Araraquara", "Bauru", "Campinas-Centro", "Campinas-Taquaral", "Catanduva", "Diadema", "Guarulhos-Pimentas", "Ibirapuera",
                 "Interlagos", "Itaim Paulista", "Itaquera", "Limeira", "Mooca", "Osasco", "Parque D.Pedro II", "Perus", "Pinheiros", "Piracicaba",
                 "Presidente Prudente", "Rio Claro-Jd.Guanabara", "S.Bernardo-Centro", "Santa Gertrudes", "Santana", "Santo Amaro", "Santos", "Santos-Ponta da Praia",
                 "Sorocaba"]
    
    for i in range(0, len(num_estacao)):
        if file.removesuffix(".csv") == num_estacao[i]:
            coefsReg["Num_estacao"] = num_estacao.index(file.removesuffix(".csv"))+1
        else:
            coefsReg["Num_estacao"] = []


    coefsReg["Num_estacao"] = num_estacao.index(file.removesuffix(".csv"))
    coefsReg['Estacao'] = file.removesuffix(".csv")
    coefsReg['Parametros_GLM'] = res.params
    coefsReg['RMS_GLM'] = round(rms,2)
    coefsReg['R2_GLM'] = round(r2,2)
    coefsReg['Std_erro_GLM'] = np.around(res.bse,2)
    coefsReg['Parametros_LM'] = linear_regressor.coef_
    coefsReg['LM_INTERCEPT'] = linear_regressor.intercept_
    coefsReg['R2_LM'] = round(R2_LM,2)
    coefsReg['R2_LM_alternativo'] = linear_regressor.score(X,Y)
    coefsReg['Std_erro_LM'] = np.around(erro_std_LM,2)
    coefsReg['Estacao'] = file.removesuffix(".csv")
    coefsReg["Num_estacao"] = num_estacao.index(file.removesuffix(".csv"))+1
    coefsReg["RMSE_LM"] =round(sqrt(sum(resid_LM2)/len(resid_LM2)),2)
    coefsReg["RMSE_GLM"] =round(sqrt(sum(resid2)/len(resid2)),2)
    coefsReg["RMSE_LM_LOG"] = round(sqrt(sum(resid_antlog2)/len(resid_antlog2)),2)
    coefsReg["RMSE_GLM_LOG"] = round(rms_log,2)
    
    valor_shapiro = 0.05
    if (stats.shapiro(res.resid_pearson).pvalue) > valor_shapiro:
        coefsReg['Normalidade Resid GLM'] = "Sim"
    else:
        coefsReg["Normalidade Resid GLM"] = "Nao"
        coefsReg["N_normal_GLM"] = res.params
    if shapiro_p_valor_LM > valor_shapiro:
         coefsReg['Normalidade Resid LM'] = "Sim"
    else:
        coefsReg["Normalidade Resid LM"] = "Nao"
        coefsReg["N_normal_LM"] = linear_regressor.coef_ 
    
    tabela_global+=[coefsReg]
    
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(4.5, 5.2)
    
                 
    #SCATTER PARA GLM ---------------------------------------------------------------------
    
    t = Xlog+"C = "+"{:.6s}".format(str(res.params[0]))+"*"+Xlog+"Sentinel"
    ax[0].scatter(yhat, Y, label =t, color = cor,alpha = alf)
    line_fit = sm.GLM(Y, sm.add_constant(yhat, prepend=True)).fit()
    abline_plot(model_results=line_fit, ax=ax[0])
    ax[0].set_title(file.removesuffix(".csv"),fontsize = 8, fontweight="bold")
    ax[0].set_ylabel('Observed values',fontsize = 8)
    ax[0].set_xlabel('GLM fitted values',fontsize = 8)
    ax[0].tick_params(axis='both', which='major', labelsize=7)
    ax[0].legend(fontsize =9, markerscale = 0.00001, frameon = False)

    
    
    # SCATTER PARA LM ---------------------------------------------------------------------
    u = Xlog+"C = "+str(round(float(linear_regressor.intercept_[0]),3))+"+("+str(round(float(linear_regressor.coef_[0]),3))+"*"+Xlog+"Sentinel)"
    ax[1].scatter(Y_pred,Y, label = u,color = cor, alpha = alf)
    line_fit = sm.OLS(Y, sm.add_constant(Y_pred, prepend=True)).fit()
    abline_plot(model_results=line_fit, ax=ax[1])
    ax[1].set_ylabel('Observed values',fontsize = 8)
    ax[1].set_xlabel('LM fitted values',fontsize = 8)
    ax[1].tick_params(axis='both', which='major', labelsize=7)
    ax[1].legend(fontsize =9, markerscale = 0.000001, frameon = False)
    
media_LM = sum(coefs_LM)/len(coefs_LM)
media_GLM = sum(coefs_GLM)/len(coefs_GLM)
media_R2_LM = sum(media_R2LM)/len(media_R2LM)
media_RMS_GLM = sum(media_RMSGLM)/len(media_RMSGLM)

coefsReg["media_LM_RMSE_log"] = sum(media_LM_LOG_RMSE)/len(media_LM_LOG_RMSE)
coefsReg["media_GLM_RMSE_log"] = sum(media_GLM_LOG_RMSE)/len(media_GLM_LOG_RMSE)     
coefsReg["media_LM_RMSE"] = sum(media_LM_RMSE)/len(media_LM_LOG_RMSE)
coefsReg["media_GLM_RMSE"] = sum(media_GLM_RMSE)/len(media_GLM_LOG_RMSE)         
coefsReg['Media_coef_GLM'] = media_GLM
coefsReg['Media_coef_LM'] = media_LM
coefsReg['Media_RMS_GLM'] = media_RMS_GLM
coefsReg['Media_R2_LM'] = media_R2_LM


tabela_final = pd.concat(tabela_global) 
tabela_final.to_csv('E:/Meus Documentos/Área de Trabalho/GLM_mensal/PLANILHA_RESUMO/'+poluente2+"_LOG.csv",index=False)
    
