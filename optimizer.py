import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from pysr import PySRRegressor
import optuna
import logging
import matplotlib.pyplot as plt
from scipy import stats


class ParameterOptimizer:
    def __init__(self, X, y, n_trials=50, cv_splits=5):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.scaler = RobustScaler()
        self.best_params = None
        self.study = None
        self.equations_history = []  # 방정식 히스토리 추가

        # 로깅 설정
        self.logger = logging.getLogger(__name__)

        # 이상치 제거 및 전처리
        self._preprocess_data()

    def _preprocess_data(self):
        """데이터 전처리 및 이상치 제거"""
        # IQR을 사용한 이상치 제거
        z_scores = stats.zscore(self.y)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = abs_z_scores < 3

        self.X = self.X[filtered_entries]
        self.y = self.y[filtered_entries]

        # 특성 스케일링
        self.X = self.scaler.fit_transform(self.X)

        # 타겟 변수 로그 변환 (음수나 0이 없는 경우)
        if np.all(self.y > 0):
            self.y = np.log1p(self.y)

        self.logger.info(f"데이터 전처리 완료: {len(self.X)} 샘플 남음")

    def _create_model(self, trial):
        """단일 trial을 위한 PySR 모델 생성"""
        alpha = trial.suggest_float('alpha', 0.1, 0.9)
        beta = 1 - alpha

        elementwise_loss_str = f"using Statistics; myloss(x, y) = {alpha}*((x-y)/y)^2 + {beta}* Statistics.var(x-y)"

        model = PySRRegressor(
            elementwise_loss=elementwise_loss_str,
            procs=4,
            populations=500,
            population_size=50,
            ncycles_per_iteration=1000,
            maxsize=30,
            model_selection="best",
            niterations=1,
            binary_operators=["pow", "/", "*"],
            constraints={"pow": (-1, 2)},
            nested_constraints={"pow": {"pow": 0}},
            tournament_selection_p=0.8,
            tournament_selection_n=10,
            complexity_of_constants=2,
            weight_randomize=0.1,
            precision=32,
            turbo=True,
            warm_start=True,
        )
        return model, alpha, beta

    def _objective(self, trial):
        """최적화 목적 함수"""
        model, alpha, beta = self._create_model(trial)

        # K-fold 교차 검증
        kf = KFold(n_splits=self.cv_splits, shuffle=True, random_state=42)
        scores = []
        equations = []

        for train_idx, val_idx in kf.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                # 현재 방정식 저장
                current_equation = str(model.get_best())
                equations.append(current_equation)

                mse = np.mean((y_val - y_pred) ** 2)
                var = np.var(y_pred - y_val)
                score = alpha * mse + beta * var

                if np.isfinite(score):
                    scores.append(score)

            except Exception as e:
                self.logger.warning(f"Fold 학습 실패: {str(e)}")
                continue

        # 방정식 히스토리에 현재 trial의 최고 방정식 추가
        if equations:
            self.equations_history.append({
                'trial': trial.number,
                'equation': min(equations, key=len),  # 가장 간단한 방정식 선택
                'score': np.mean(scores) if scores else float('inf')
            })

        if not scores:
            return float('inf')

        return np.mean(scores)

    def optimize(self):
        """하이퍼파라미터 최적화 수행"""
        study = optuna.create_study(direction='minimize')

        try:
            study.optimize(self._objective, n_trials=self.n_trials,
                           callbacks=[self._callback])
            self.study = study
            self.best_params = study.best_params
        except Exception as e:
            self.logger.error(f"최적화 실패: {str(e)}")
            raise

        return self.best_params

    def _callback(self, study, trial):
        """최적화 진행 상황 모니터링"""
        if trial.number % 5 == 0:
            self.logger.info(f"\nTrial {trial.number}:")
            self.logger.info(f"Current value: {trial.value:.6f}")
            if self.equations_history:
                latest_eq = self.equations_history[-1]
                self.logger.info(f"Current equation: {latest_eq['equation']}")

    def plot_optimization_results(self):
        """최적화 과정 시각화"""
        if self.study is None:
            self.logger.warning("시각화를 위한 최적화 결과가 없습니다.")
            return

        plt.figure(figsize=(20, 10))

        # 1. 최적화 히스토리
        plt.subplot(2, 2, 1)
        trials = self.study.trials
        values = [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        plt.plot(values, 'b-', linewidth=2, label='Score')
        plt.xlabel('Trial number', fontsize=12)
        plt.ylabel('Objective value', fontsize=12)
        plt.title('Optimization History', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        # 2. Alpha vs Score
        plt.subplot(2, 2, 2)
        alphas = [t.params['alpha'] for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        plt.scatter(alphas, values, alpha=0.6, c='blue', s=100)
        plt.xlabel('Alpha', fontsize=12)
        plt.ylabel('Objective value', fontsize=12)
        plt.title('Alpha vs Score', fontsize=14)
        plt.grid(True, alpha=0.3)

        # 3. 방정식 복잡도 변화
        plt.subplot(2, 2, 3)
        eq_lengths = [len(eq['equation']) for eq in self.equations_history]
        plt.plot(eq_lengths, 'g-', linewidth=2)
        plt.xlabel('Trial number', fontsize=12)
        plt.ylabel('Equation length', fontsize=12)
        plt.title('Equation Complexity Over Trials', fontsize=14)
        plt.grid(True, alpha=0.3)

        # 4. 상위 5개 방정식 표시
        plt.subplot(2, 2, 4)
        best_equations = sorted(self.equations_history, key=lambda x: x['score'])[:5]
        plt.axis('off')
        plt.title('Top 5 Best Equations', fontsize=14)

        for i, eq in enumerate(best_equations, 1):
            plt.text(0.1, 0.9 - i * 0.15,
                     f"{i}. Score: {eq['score']:.6f}\n   {eq['equation']}",
                     fontsize=10,
                     wrap=True)

        plt.tight_layout()
        plt.show()

    def get_optimization_summary(self):
        """최적화 결과 요약"""
        if self.study is None:
            return {"status": "최적화가 아직 실행되지 않음"}

        best_trial = self.study.best_trial

        # 상위 5개 방정식 추가
        best_equations = sorted(self.equations_history,
                                key=lambda x: x['score'])[:5]

        return {
            "best_alpha": self.best_params['alpha'],
            "best_beta": 1 - self.best_params['alpha'],
            "best_value": best_trial.value,
            "n_completed_trials": len(self.study.trials),
            "n_failed_trials": len([t for t in self.study.trials
                                    if t.state != optuna.trial.TrialState.COMPLETE]),
            "top_5_equations": [
                {
                    "equation": eq['equation'],
                    "score": eq['score'],
                    "trial": eq['trial']
                } for eq in best_equations
            ]
        }