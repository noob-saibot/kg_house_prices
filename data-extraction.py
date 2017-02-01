import pandas


class Extractor:
    def __init__(self, work_dir, file_tr):
        self.work_dir = work_dir
        self.file_tr = file_tr
        self.frame = None

    def __str__(self):
        return str(self.frame)

    def frame_corr(self, *data_frames):
        """
        Correlation of elements of frame
        Parameters
        ----------
        :param data_frames: list
            List of frames.

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
                    _all_frames_corr.append(data_frame.corr())
                else:
                    print(index, "It's not a data frame", sep=' --- ')
                    _all_frames_corr.append(None)
            return _all_frames_corr

        elif isinstance(self.frame, pandas.DataFrame):
            return self.frame.corr()
        else:
            return 'Frame is empty.'

    # Frame creation
    def df_creation(self):
        self.frame = pandas.read_csv(self.work_dir+self.file_tr, index_col=0)
        return self.frame


class Viewer:
    pass

if __name__ == "__main__":
    E = Extractor(work_dir='C:/work/houses/kg_house_prices/', file_tr='data/train.csv')
    frame = E.df_creation()
    print(E.frame_corr())
