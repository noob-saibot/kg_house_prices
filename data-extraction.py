import pandas
import matplotlib.pyplot as plt


class Extractor:
    def __init__(self, work_dir, file_tr):
        self.work_dir = work_dir
        self.file_tr = file_tr
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
                        _all_frames_corr.append(self._delta_corr(data_frame.corr(), delta_lvl))
                    else:
                        _all_frames_corr.append(data_frame.corr())
                else:
                    print(index, "It's not a data frame", sep=' --- ')
                    _all_frames_corr.append(None)
            return _all_frames_corr

        elif isinstance(self.frame, pandas.DataFrame):
            if delta_lvl:
                return self._delta_corr(self.frame.corr(), delta_lvl)
            else:
                return self.frame.corr()
        else:
            return 'Frame is empty.'

    # Frame creation
    def df_creation(self):
        self.frame = pandas.read_csv(self.work_dir+self.file_tr, index_col=0)
        return self.frame

    @staticmethod
    def _delta_corr(data_frame, delta):
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

if __name__ == "__main__":
    E = Extractor(work_dir='C:/work/houses/kg_house_prices/', file_tr='data/train.csv')
    frame = E.df_creation()
    df = E.frame_corr(delta_lvl=0.2)[['SalePrice']]
    Viewer(df).bar()
    print(Viewer(df).site_chart())
