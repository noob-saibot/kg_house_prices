import pandas
from data_extraction import Extractor
from sklearn.preprocessing import LabelEncoder

def go_deeper():

    drop_list = ['Street', 'Utilities', 'PoolArea', 'PoolQC']


    dict_of_search={
       'LotFrontage': ((313, 242000), (313, 160000)),
       'ExterQual': (),
       'ScreenPorch': ((410, 475000),),
       'GarageArea': ((1248, 81000), (1418, 160000), (1356, 168000), (1390, 253293)),
       'KitchenAbvGr': ((0,127500), (3,113000), (3,106000)),
       'Exterior1st': ((14,105000), (13,262000), (12,287000), (12,230000)),
       'BsmtExposure': ((4,556581),),
       'CentralAir': (),
       '2ndFlrSF': (),
       'Condition1': ((3, 475000), (6,423000)),
       'OverallCond': ((2, 394432), (9, 475000)),
       'LotConfig': ((4, 207000), (4, 183900), (4, 128000), (4, 315000)),
       'Condition2': (),
       'Street': (),
       'GarageYrBlt': ((1995, 625000),),
       'BsmtFullBath': ((3, 179000),),
       'EnclosedPorch': ((552, 235000),),
       'PoolArea': (),
       'WoodDeckSF': ((857, 385000),),
       'MiscFeature': (),
       'BsmtHalfBath': ((2, 127500), (2,194201)),
       'GarageFinish': ((1,582933), (2,475000)),
       'OpenPorchSF': ((523, 34900), (547, 256000), (502, 325000)),
       'Foundation': ((5, 265979),),
       'LotShape': (),
       'HouseStyle': ((7, 475000),),
       'MasVnrType': ((4, 277000), (0, 465000)),
       'BsmtFinSF2': ((1474, 301000),),
       'YearBuilt': ((1892, 475000),),
       'LandContour': ((1, 315000), (3, 538000)),
       'LotArea': ((115149, 302000), (159000, 277000), (164660, 228950), (215245, 375000)),
       'MiscVal': ((3500, 55000), (8300, 190000), (15500, 151500)),
       'MSSubClass': (),
       '3SsnPorch': (),
       'Fence': ((3,475000), (2,381000)),
       'Heating': ((1, 375000),),
       'TotalBsmtSF': ((6110, 160000),),
       'PoolQC': (),
       'HeatingQC': ((4,87000),),
       'BsmtFinType2': ((3,555000),(0,284000),(4,402000), (4,375000)),
       'SaleCondition': ((4, 359100), (4, 274970)),
       'BsmtFinType1': ((2,538000), (2,375000), (4,381000)),
       'GarageCars': (),
       'KitchenQual': ((0,625000), (1,375000), (1,359100)),
       'LowQualFinSF': ((572, 475000),),
       'YrSold': ((2006, 625000), (2006, 556581), (2010, 611657), (2010, 538000)),
       '1stFlrSF': ((4692,160000),),
       'BsmtCond': ((4, 61000),),
       'SaleType': ((4, 451950), (5, 328900)),
       'GarageQual': (),
       'OverallQual': ((4, 256000), (10, 160000), (10, 184750)),
       'ExterCond': ((3, 765000), (4,118000), (4, 161000), (4,325000)),
       'BedroomAbvGr': ((8,200000), (6,200000), (2,611657), (2,555000), (1,501837)),
       'YearRemodAdd': ((1996, 625000), (1965, 375000)),
       'MSZoning': (),
       'GarageType': ((5,359100), (2,475000)),
       'RoofMatl': ((7, 160000), (6, 137000)),
       'Neighborhood': (),
       'PavedDrive': (),
       'BsmtFinSF1': ((5644, 160000),),
       'GarageCond': ((3, 274970), (3, 302000)),
       'Fireplaces': (),
       'TotRmsAbvGrd': ((14,200000), (2, 39300), (10,555000), (10,625000)),
       'FireplaceQu': ((1,625000), (1,475000), (4, 130500)),
       'Exterior2nd': ((15,105000), (3, 625000)),
       'Alley': (),
       'BldgType': (),
       'Electrical': ((0,167500), (5,67000)),
       'MasVnrArea': ((1600, 239000),),
       'BsmtQual': ((1, 538000), (2,475000)),
       'MoSold': ((1, 582933), (3,611657), (4,555000), (6,538000)),
       'LandSlope': ((1, 538000),),
       'Functional': ((4,538000), (6,129000), (3,316600), (2,315000), (2,61000)),
       'Utilities': (),
       'BsmtUnfSF': (),
       'HalfBath': (),
       'RoofStyle': ((5, 260000), (5,190000)),
       'SalePrice': (),
       'FullBath': (),
       'GrLivArea': ((5642,160000), (4676, 184750))
    }

    E = Extractor(work_dir='./',
                  file_train='data/train.csv',
                  file_test='data/test.csv')
    frame, frame2 = E.df_creation()

    for i in frame.columns:
        if frame[i].dtype == 'object' and dict_of_search.get(i, False):
            enc = LabelEncoder()
            frame[i] = frame[i].factorize()[0]
            enc = enc.fit(frame[i])
            frame[i] = enc.transform(frame[i])
            for j in dict_of_search[i]:
                v = set(frame[i][frame[i] == j[0]].index.tolist()).intersection(frame[i][frame['SalePrice'] == j[1]].index.tolist())
                if v:
                    print(frame[i][tuple(v)[0]], end=' ')
                    frame[i][tuple(v)[0]] = frame[i].mean()
                    print(frame[i][tuple(v)[0]])
            frame[i] = enc.inverse_transform(frame[i])
        elif frame[i].dtype != 'object' and dict_of_search.get(i, False):
            for j in dict_of_search[i]:
                v = set(frame[i][frame[i] == j[0]].index.tolist()).intersection(
                    frame[i][frame['SalePrice'] == j[1]].index.tolist())
                if v:
                    frame[i][tuple(v)[0]] = frame[i].mean()

    return pandas.concat([frame, frame2], axis=0).drop(drop_list, axis=1)