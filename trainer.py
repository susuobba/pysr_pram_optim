import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from pysr import PySRRegressor
from datetime import datetime


class ModelTrainer:
    def __init__(self, alpha, beta, output_dir, model_name):
        """
        Args:
            alpha (float): MSE 가중치
            beta (float): 분산 가중치
            output_dir (str): 모델과 결과를 저장할 디렉토리
            model_name (str): 모델 이름
        """
        self.alpha = alpha
        self.beta = beta
        self.output_dir = output_dir
        self.model_name = model_name
        self.model = None
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'equations': [],
            'complexities': []
        }

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

    def initialize_model(self):
        """PySR 모델 초기화"""
        elementwise_loss_str = f"using Statistics; myloss(x, y) = {self.alpha}*((x-y)/y)^2 + {self.beta}* Statistics.var(x-y)"

        self.model = PySRRegressor(
            elementwise_loss=elementwise_loss_str,
            procs=4,
            populations=1000,
            population_size=50,
            ncycles_per_iteration=500,
            maxsize=35,
            model_selection="best",
            niterations=1,
            binary_operators=["pow", "/", "*"],
            constraints={"pow": (-1, 2)},
            nested_constraints={"pow": {"pow": 0}},
            tournament_selection_n=15,
            adaptive_parsimony_scaling=25,
            weight_randomize=0.1,
            precision=32,
            complexity_of_variables=2,
            turbo=True,
            warm_start=True,
        )

    def calculate_loss(self, y_true, y_pred):
        """사용자 정의 손실 함수"""
        mse = mean_squared_error(y_true, y_pred)
        variance = np.var(y_pred - y_true)
        return self.alpha * mse + self.beta * variance

    def evaluate(self, X, y_true):
        """모델 평가"""
        y_pred = self.model.predict(X)
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'variance': np.var(y_pred - y_true),
            'combined_loss': self.calculate_loss(y_true, y_pred)
        }
        return metrics, y_pred

    def train(self, X_train, y_train, X_val, y_val,
              max_iterations=100, patience=10, eval_interval=1):
        """
        모델 학습
        Args:
            patience (int): 조기 종료를 위한 patience
            eval_interval (int): 평가 간격
        """
        if self.model is None:
            self.initialize_model()

        best_val_loss = float('inf')
        patience_counter = 0
        best_equation = None
        best_iteration = 0

        for iteration in range(max_iterations):
            # 모델 학습
            self.model.fit(X_train, y_train)

            # 현재 식 저장
            current_equation = str(self.model.get_best())
            self.training_history['equations'].append(current_equation)

            # 현재 복잡도 저장
            complexity = len(current_equation.split('+'))
            self.training_history['complexities'].append(complexity)

            if iteration % eval_interval == 0:
                # 훈련 및 검증 세트 평가
                train_metrics, _ = self.evaluate(X_train, y_train)
                val_metrics, _ = self.evaluate(X_val, y_val)

                self.training_history['train_losses'].append(train_metrics['combined_loss'])
                self.training_history['val_losses'].append(val_metrics['combined_loss'])

                print(f"\nIteration {iteration}:")
                print(f"Train Loss: {train_metrics['combined_loss']:.6f}")
                print(f"Val Loss: {val_metrics['combined_loss']:.6f}")
                print(f"Current Equation: {current_equation}")
                print(f"Complexity: {complexity}")

                # 조기 종료 확인
                if val_metrics['combined_loss'] < best_val_loss:
                    best_val_loss = val_metrics['combined_loss']
                    best_equation = current_equation
                    best_iteration = iteration
                    patience_counter = 0
                    # 최고 성능 모델 저장
                    self.save_model(f"best_model_{self.model_name}")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\nEarly stopping at iteration {iteration}")
                    print(f"Best iteration was {best_iteration}")
                    print(f"Best equation: {best_equation}")
                    break

            # 학습 과정 시각화
            if iteration % (eval_interval * 5) == 0:
                self.plot_training_progress()

        return best_equation, best_iteration

    def save_model(self, filename):
        """모델 저장"""
        filepath = os.path.join(self.output_dir, f"{filename}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'alpha': self.alpha,
                'beta': self.beta,
                'training_history': self.training_history
            }, f)

    def load_model(self, filename):
        """모델 로드"""
        filepath = os.path.join(self.output_dir, f"{filename}.pkl")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.alpha = data['alpha']
            self.beta = data['beta']
            self.training_history = data['training_history']

    def plot_training_progress(self):
        """학습 진행 상황 시각화"""
        plt.figure(figsize=(15, 10))

        # 1. Loss curves
        plt.subplot(2, 2, 1)
        plt.plot(self.training_history['train_losses'], label='Train Loss')
        plt.plot(self.training_history['val_losses'], label='Validation Loss')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # 2. Complexity over time
        plt.subplot(2, 2, 2)
        plt.plot(self.training_history['complexities'])
        plt.xlabel('Iteration')
        plt.ylabel('Equation Complexity')
        plt.title('Equation Complexity Over Time')

        plt.tight_layout()

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.output_dir,
                                 f'training_progress_{timestamp}.png'))
        plt.close()

    def final_evaluation(self, X_test, y_test):
        """최종 모델 평가"""
        metrics, y_pred = self.evaluate(X_test, y_test)

        # 결과 시각화
        plt.figure(figsize=(15, 5))

        # 1. Actual vs Predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted')

        # 2. Residuals
        plt.subplot(1, 2, 2)
        residuals = y_pred - y_test
        plt.hist(residuals, bins=50)
        plt.xlabel('Residual')
        plt.ylabel('Count')
        plt.title('Residual Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir,
                                 f'final_evaluation_{self.model_name}.png'))
        plt.close()

        return metrics