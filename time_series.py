# -*- coding: utf-8 -*-
"""
Created on Sun May 29 21:20:18 2016

@author: AbreuLastra_Work
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api

loansData = pd.read_csv('https://media.githubusercontent.com/media/briansorahan/thinkful-data-science/master/LoanStats3b.csv', header=1, low_memory=False)

loansData['issue_d_format'] = pd.to_datetime(loansData['issue_d'])
lDts = loansData.set_index('issue_d_format')
nts = lDts.resample('M').count().head()

plt.figure()
plt.plot(nts.loan_amnt)

dts = nts.diff(periods=1)

statsmodels.api.graphics.tsa.plot_acf(nts.loan_amnt)
statsmodels.api.graphics.tsa.plot_acf(dts.loan_amnt)

statsmodels.api.graphics.tsa.plot_pacf(nts.loan_amnt)
statsmodels.api.graphics.tsa.plot_pacf(dts.loan_amnt)