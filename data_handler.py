import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.io


class DataHandler:
    def __init__(self, data_path, name, selected_features):
        """
        데이터 핸들러 초기화
        Args:
            data_path (str): .mat 파일이 있는 경로
            name (str): 데이터셋 이름
            selected_features (list): 선택할 특성들의 리스트
        """
        self.data_path = data_path
        self.name = name
        self.selected_features = selected_features
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_data(self):
        """전체 데이터셋 로드"""
        data = scipy.io.loadmat(f'{self.data_path}/Target_extreme_data_MS_{self.name}.mat')
        Target_extreme_data = data['Target_extreme_data']

        feature_names = ['T', 'ζ', 'V', 'B', 'D', 'i/N', 'H', 'asp']
        X = pd.DataFrame(Target_extreme_data[:, :-1], columns=feature_names)
        y = Target_extreme_data[:, -1]

        self.X = X[self.selected_features].to_numpy()
        self.y = np.array(y)
        return self.X, self.y

    def load_sample_data(self, sample_path):
        """최적화를 위한 샘플 데이터셋 로드"""
        data = scipy.io.loadmat(f'{sample_path}/Target_extreme_data_sample.mat')
        Sample_Target_extreme_data = data['Target_extreme_data_sample']

        feature_names = ['T', 'ζ', 'V', 'B', 'D', 'i/N', 'H', 'asp']
        X = pd.DataFrame(Sample_Target_extreme_data[:, :-1], columns=feature_names)
        y = Sample_Target_extreme_data[:, -1]

        return X[self.selected_features].to_numpy(), np.array(y)

    def split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        데이터를 train, validation, test 세트로 분할
        """
        # 먼저 test set 분리
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # 남은 데이터에서 validation set 분리
        val_ratio = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )

        return (
            self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test
        )

    def get_data_stats(self):
        """데이터셋의 기본 통계 정보 반환"""
        stats = {
            'n_samples': len(self.y),
            'n_features': self.X.shape[1],
            'feature_names': self.selected_features,
            'y_mean': np.mean(self.y),
            'y_std': np.std(self.y),
            'y_min': np.min(self.y),
            'y_max': np.max(self.y)
        }
        return stats