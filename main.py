import os
from datetime import datetime
import json
import logging
from data_handler import DataHandler
from optimizer import ParameterOptimizer
from trainer import ModelTrainer


def setup_logging(output_dir):
    """로깅 설정"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_config(config, output_dir):
    """설정 저장"""
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def main():
    # 기본 설정
    config = {
        'data_path': r'G:\내 드라이브\대학원\연구\내풍\ESWL\research\result',  # 데이터 디렉토리
        'sample_path': r'G:\내 드라이브\대학원\연구\내풍\ESWL\research\result',  # 샘플 데이터 디렉토리
        'output_dir': './output',  # 출력 디렉토리
        'name': 'AC_P_NLC',  # 데이터셋 이름
        'n_trials': 50,  # 최적화 시도 횟수
        'max_iterations': 100,  # 최대 학습 반복 횟수
        'patience': 10,  # 조기 종료 patience
        'eval_interval': 1  # 평가 간격
    }

    # 출력 디렉토리 생성
    os.makedirs(config['output_dir'], exist_ok=True)

    # 로깅 설정
    logger = setup_logging(config['output_dir'])

    # 설정 저장
    save_config(config, config['output_dir'])

    try:
        # 1. 데이터 준비
        logger.info("Initializing data handler...")
        selected_features = ['V', 'T', 'asp', 'ζ', 'i/N']
        data_handler = DataHandler(config['data_path'], config['name'], selected_features)

        # 전체 데이터셋 로드
        logger.info("Loading full dataset...")
        X, y = data_handler.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = data_handler.split_data()

        # 데이터 통계 로깅
        logger.info("Dataset statistics:")
        stats = data_handler.get_data_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

        # 2. 파라미터 최적화
        logger.info("Loading sample dataset for parameter optimization...")
        X_sample, y_sample = data_handler.load_sample_data(config['sample_path'])

        logger.info("Starting parameter optimization...")
        optimizer = ParameterOptimizer(X_sample, y_sample, n_trials=config['n_trials'])
        best_params = optimizer.optimize()

        # 최적화 결과 로깅
        logger.info("Optimization results:")
        summary = optimizer.get_optimization_summary()
        for key, value in summary.items():
            logger.info(f"{key}: {value}")

        # 최적화 결과 시각화
        optimizer.plot_optimization_results()

        # 3. 모델 학습
        logger.info("Starting model training with optimized parameters...")
        model_trainer = ModelTrainer(
            alpha=best_params['alpha'],
            beta=best_params['beta'],
            output_dir=config['output_dir'],
            model_name=config['name']
        )

        best_equation, best_iteration = model_trainer.train(
            X_train, y_train,
            X_val, y_val,
            max_iterations=config['max_iterations'],
            patience=config['patience'],
            eval_interval=config['eval_interval']
        )

        # 4. 최종 평가
        logger.info("Performing final evaluation...")
        final_metrics = model_trainer.final_evaluation(X_test, y_test)

        logger.info("Final evaluation metrics:")
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value}")

        logger.info(f"Best equation found: {best_equation}")
        logger.info(f"Best iteration: {best_iteration}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()