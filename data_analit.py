from data_extraction import Extractor
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import GradientBoostingRegressor
from mpldatacursor import datacursor
matplotlib.style.use('ggplot')

E = Extractor('./', file_train='data/train.csv', file_test='data/test.csv')
result_frame = E.df_creation()[0]

# List of object columns
ls = []
for i in result_frame.columns:
    if result_frame[i].dtype == 'object':
        ls.append(i)

# Encoding and fillin data frame
result_frame = E.encoding_for_labels(result_frame)
result_frame = result_frame.fillna(result_frame.mean())

# Feature importance extraction with best regressor
imp, col = E.importance(result_frame, 'SalePrice', {
    'n_estimators': 4840,
    'learning_rate': 0.22,
    'max_features': 'sqrt',
}, method=GradientBoostingRegressor)
dict_of_imp = dict(zip(col, imp))

# Copy of data frame for datacursor
hover = result_frame.copy()
hover = hover.astype(str)

# Plotting each object column
for i in ls:
    fig, axes = plt.subplots(nrows=1, ncols=2)
    rs = result_frame.groupby(i)[i].count()
    cr = result_frame[i].corr(result_frame['SalePrice'])
    ax = rs.plot.pie(ax=axes[0], title='Corr:%s, Importance: %s'%(cr, dict_of_imp[i]))
    result_frame.plot.scatter(y='SalePrice', x=i, s=50, ax=axes[1], title='Corr:%s, Importance: %s'%(cr, dict_of_imp[i]))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    datacursor(hover=True, point_labels=hover[i])
    plt.show()