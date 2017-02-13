import pandas
from numpy import sqrt, log1p, expm1
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import make_scorer


class Extractor:
    def __init__(self, work_dir, file_train, file_test = None):
        self.work_dir = work_dir
        self.file_tr = file_train
        self.file_ts = file_test
        self.frame = None

    def __str__(self):
        return str(self.frame)

    def frame_corr(self, *data_frames, delta_lvl=None):
        """
        Correlation of elements of frame
        Parameters
        ----------
        :param data_frames: list
            List of frames.

        :param delta_lvl: float
            Correlation level.

        Returns
        -------
        :return _all_frames_corr: list
            List of correlation for each frame.

        :return self.frame: pandas.DataFrame
            Correlation for each element if MAIN frame
        """
        if data_frames:
            _all_frames_corr = []
            for index, data_frame in enumerate(data_frames):
                if isinstance(data_frame, pandas.DataFrame):
                    if delta_lvl:
                        _all_frames_corr.append(self.delta_corr(data_frame.corr(), delta_lvl))
                    else:
                        _all_frames_corr.append(data_frame.corr())
                else:
                    print(index, "It's not a data frame", sep=' --- ')
                    _all_frames_corr.append(None)
            return _all_frames_corr

        elif isinstance(self.frame, pandas.DataFrame):
            if delta_lvl:
                return self.delta_corr(self.frame.corr(), delta_lvl)
            else:
                return self.frame.corr()
        else:
            return 'Frame is empty.'

    # Frame creation
    def df_creation(self):
        self.frame = pandas.read_csv(self.work_dir+self.file_tr, index_col=0)
        if self.file_ts:
            frame_test = pandas.read_csv(self.work_dir + self.file_ts, index_col=0)
            return self.frame, frame_test
        return self.frame

    @staticmethod
    def delta_corr(data_frame, delta):
        """
        Correlation of elements of frame
        Parameters
        ----------
        :param data_frame: pandas.DataFrame
            Data frame.

        :param delta: list
            Correlation level.

        Returns
        -------
        :return data_frame: pandas.DataFrame
            Data frame with Nan values on delta places.
        """
        if isinstance(data_frame, pandas.DataFrame):
            for a in data_frame.columns:
                data_frame.ix[abs(data_frame[a]) < delta, a] = None
                data_frame.ix[abs(data_frame[a]) == 1.0, a] = None
            return data_frame

    @staticmethod
    def importance(data_frame, y_field, m_param, method=ExtraTreesRegressor):
        """
            Feature importance extraction
            Parameters
            ----------
            :param data_frame: pandas.DataFrame
                Data frame.

            :param y_field: str
                Name of column with respect to which we will extract feature importance.

            :param method: Class regressor of sklearn
                Regressor for feature extraction.

            :param m_param: tuple
                Tuple of parameters for Regressor.

            Returns
            -------
            :return feature_importances_: numpy.array
                Importance of features.

            :return columns: pandas.indexes.base.Index
                Names of features.
            """
        data_frame = data_frame.fillna(data_frame.mean())
        clf = method(**m_param).fit(data_frame.drop([y_field], axis=1), data_frame[y_field])
        return clf.feature_importances_, data_frame.drop([y_field], axis=1).columns

    @staticmethod
    def encoding_for_labels(data_frame):
        enc = LabelEncoder()
        for column in data_frame.select_dtypes(include=['object']).columns:
            data_frame[column] = data_frame[column].factorize()[0]
            data_frame[column] = enc.fit_transform(data_frame[column])
        return data_frame

    @staticmethod
    def normalize_it(n_frame, n_method=log1p):
        for col in n_frame:
            if n_frame[col].dtype != 'object':
                n_frame[col] = n_method(n_frame[col])
        return n_frame

    def saver(self, frame_save, name):
        frame_save.to_csv(self.work_dir+name)


class Viewer:
    import matplotlib
    matplotlib.style.use('ggplot')

    def __init__(self, data_frame):
        self.data_frame = data_frame.dropna(axis=0)

    def bar(self):
        self.data_frame.plot(kind='bar')
        plt.show()

    def line(self):
        self.data_frame.plot()
        plt.show()

    def site_chart(self):
        template = """{{labels: {labels}, datasets:
        [{{label: "Dataset", backgroundColor: window.chartColors.red,
        borderColor: window.chartColors.red, data: {data},
        fill: false,}}, {{}}]}},"""

        result_dict = {'labels': [x for x in self.data_frame.index.values],
                       'data': [x[0] for x in self.data_frame.values]}

        return template.format(**result_dict).replace('\'', '"')


class Learning:
    def __init__(self, data_frame, cross_params=5, y_col=''):
        self.data_frame = data_frame
        self.cross = cross_params
        if not y_col:
            self.slice = data_frame.columns[-1]
        else:
            self.slice = y_col

    def __str__(self):
        return str(self.data_frame)

    def folding(self):
        folds = KFold(n_splits=self.cross, shuffle=False)
        folds = folds.split(self.data_frame.drop(['SalePrice'], axis=1), self.data_frame['SalePrice'])
        # for train_index, test_index in folds:
        #     print(self.data_frame.shape)
        #     print("TRAIN:", len(train_index), "TEST:", len(test_index))
        return folds

    @staticmethod
    def root_mse_score(predictions, targets):
        return sqrt(((predictions - targets) ** 2).mean())

    def trees(self, m_params, cross=True):
        frame_l = self.data_frame.fillna(self.data_frame.mean())
        reg = ExtraTreesRegressor(**m_params)
        print(frame_l[self.slice].values)

        if cross:
            results = cross_val_score(reg, frame_l.drop([self.slice], axis=1),
                                      frame_l[self.slice], cv=5, n_jobs=-1,
                                      scoring=make_scorer(self.root_mse_score))
            results.mean()
        else:
            reg.fit(frame_l.drop([self.slice], axis=1), frame_l[self.slice])
            return reg


if __name__ == "__main__":
    E = Extractor(work_dir='./',
                  file_train='data/train.csv',
                  file_test='data/test.csv')
    frame, frame2 = E.df_creation()

    result_frame = pandas.concat([frame, frame2], axis=0)

    # df = E.frame_corr(delta_lvl=0.2)[['SalePrice']]
    # frame = E.encoding_for_labels(frame)

    dict_of_params = {
        'n_estimators': 300,
        #'n_jobs': 4,
        "verbose": True,
        "criterion": 'mse'
    }

    # imp, col = E.importance(frame, 'SalePrice', m_param=dict_of_params)
    # frame = pandas.DataFrame(imp, dtype='float64', index=col, columns=['Importance'])
    # V = Viewer(frame.sort_values(by='Importance', ascending=False).head(10))
    # V.bar()
    # print(V.site_chart())

    result_frame = pandas.get_dummies(result_frame)
    result_frame = E.normalize_it(result_frame)
    frame_train = result_frame.dropna(subset=['SalePrice'])
    frame_test = result_frame[result_frame['SalePrice'].isnull()]

    frame_train = frame_train.fillna(frame_train.mean())
    X_train, X_test, y_train, y_test = train_test_split(frame_train.drop(['SalePrice'], axis=1),
                                                        frame_train['SalePrice'],
                                                        train_size=0.7)
    model = ExtraTreesRegressor(**dict_of_params)
    model = GradientBoostingRegressor(**dict_of_params)


    model.fit(X_train, y_train)

    print(Learning.root_mse_score(model.predict(X_test), y_test))

    submission = frame_test['SalePrice']
    frame_test = frame_test.drop(['SalePrice'], axis=1)
    frame_test = frame_test.fillna(frame_test.mean())
    rs = expm1(model.predict(frame_test))

    submission = pandas.DataFrame(data=rs, index=submission.index, columns=['SalePrice'])
    E.saver(submission, 'sub.csv')




