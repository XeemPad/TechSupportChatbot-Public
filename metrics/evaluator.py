import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class QAEvaluator:
    """Класс для комплексной оценки QA-системы"""
    
    def __init__(self, qa_system, test_sets=None, metrics_config=None, work_dir=Path(".")):
        self.qa_system = qa_system
        self.test_sets = test_sets or {}
        self.metrics_config = metrics_config or self._default_config()
        self.work_dir = Path(work_dir)
        self.history_dir = self.work_dir / "metrics_history"
        self.history_dir.mkdir(exist_ok=True, parents=True)
        self.results = {}
    
    def _default_config(self):
        """Конфигурация метрик по умолчанию"""
        return {
            "retrieval": ["precision@k", "recall@k", "mrr", "ndcg@k"], 
            "generation": ["bleu", "rouge-l", "exact_match"],
            "efficiency": ["latency", "memory_usage"]
        }
    
    def evaluate_all(self):
        """Запуск всех оценок"""
        results = {}
        
        for test_name, test_data in self.test_sets.items():
            print(f"Оцениваем на наборе: {test_name}")
            results[test_name] = self.evaluate_test_set(test_data)
            
        self.results = results
        return results
    
    def evaluate_test_set(self, test_data):
        """Оценка на конкретном тестовом наборе"""
        results = {}
        
        # Оценка поиска
        if "retrieval" in self.metrics_config:
            from .retrieval import evaluate_retrieval
            results["retrieval"] = evaluate_retrieval(
                self.qa_system, test_data, k_values=[1, 3, 5]
            )
        
        # Оценка генерации
        if "generation" in self.metrics_config:
            from .generation import evaluate_generation
            results["generation"] = evaluate_generation(
                self.qa_system, test_data
            )
        
        # Оценка эффективности
        if "efficiency" in self.metrics_config:
            from .efficiency import evaluate_efficiency
            results["efficiency"] = evaluate_efficiency(
                self.qa_system, test_data
            )
            
        return results
    
    def save_results(self, name=None):
        """Сохранение результатов оценки"""
        if not self.results:
            print("Нет результатов для сохранения")
            return None
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = name or f"qa_metrics_{timestamp}"
        file_path = self.history_dir / f"{name}.json"
        
        # Конвертируем numpy типы для JSON сериализации
        def convert_types(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):  # type: ignore
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):  # type: ignore
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
                
        # Подготавливаем данные для сохранения
        save_data = {
            "timestamp": timestamp,
            "model_info": self._get_model_info(),
            "results": {}
        }
        
        for test_name, test_results in self.results.items():
            save_data["results"][test_name] = {}
            for category, metrics in test_results.items():
                save_data["results"][test_name][category] = {
                    k: convert_types(v) for k, v in metrics.items()
                }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
            
        print(f"Метрики сохранены в {file_path}")
        return file_path
    
    def _get_model_info(self):
        """Получение информации о модели"""
        info = {}
        
        # Извлечение параметров модели
        for attr in ['qa_threshold', 'rag_threshold', 'qa_weight', 'rag_weight', 'model_name', 'use_sbert']:
            if hasattr(self.qa_system, attr):
                info[attr] = getattr(self.qa_system, attr)
                
        # Для более сложных объектов используем дополнительную обработку
        if hasattr(self.qa_system, 'baseline_searcher') and hasattr(self.qa_system.baseline_searcher, 'model_name'):
            info['embedding_model'] = self.qa_system.baseline_searcher.model_name
            
        return info
    
    def plot_comparison(self, other_results=None, metric_filter=None):
        """Построение сравнительного графика"""
        # Получаем данные текущей оценки
        all_results = {"current": self._flatten_results(self.results)}
        
        # Добавляем другие результаты для сравнения
        if other_results:
            for name, res in other_results.items():
                all_results[name] = self._flatten_results(res)
        
        # Создаем DataFrame для графика
        df = pd.DataFrame(all_results)
        
        # Исключаем метрики эффективности, чтобы не портить масштаб
        excluded_metrics = ['latency', 'memory_usage']
        df = df.loc[[idx for idx in df.index if not any(metric in idx for metric in excluded_metrics)]]
        
        # Фильтруем метрики если нужно
        if metric_filter:
            df = df.loc[[col for col in df.index if any(m in col for m in metric_filter)]]
        
        # Создаем график с улучшенной визуализацией
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Настраиваем цвета и прозрачность
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Строим график с прозрачностью и улучшенной легендой
        df.plot(
            kind='bar', 
            ax=ax,
            alpha=0.7,
            width=0.8,
            edgecolor='black',
            linewidth=0.5,
            color=colors[:len(df.columns)]
        )
        
        plt.title("Сравнение метрик качества", fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_for_gradio(self, other_results=None, metric_filter=None):
        """Создает график в формате, подходящем для Gradio
        
        Returns:
            fig: Matplotlib фигура для Gradio
        """
        # Создаем фигуру с соотношением сторон для веб-интерфейса
        fig = self.plot_comparison(other_results, metric_filter)
        
        # Настраиваем для веб-отображения
        plt.tight_layout()
        
        return fig
    
    def load_historical_results(self, filename=None):
        """Загрузка исторических результатов для сравнения"""
        if filename:
            file_path = self.history_dir / filename
        else:
            # Берем самый свежий файл
            files = list(self.history_dir.glob("*.json"))
            if not files:
                print("Исторические данные не найдены")
                return None
            file_path = max(files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"Загружены исторические данные из {file_path}")
                return data.get("results", {})
        except Exception as e:
            print(f"Ошибка загрузки исторических данных: {e}")
            return None
    
    def _flatten_results(self, nested_results, prefix=""):
        """Преобразование вложенных результатов в плоскую структуру"""
        flat = {}
        
        for test_name, test_results in nested_results.items():
            test_prefix = f"{prefix}_{test_name}" if prefix else test_name
            
            # Проверяем, является ли test_results словарем
            if isinstance(test_results, dict):
                for category, metrics in test_results.items():
                    category_prefix = f"{test_prefix}_{category}"
                    
                    # Проверяем, является ли metrics словарем
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            flat_key = f"{category_prefix}_{metric_name}"
                            flat[flat_key] = value
                    else:
                        # Если metrics не словарь, а скаляр
                        flat_key = f"{test_prefix}_{category}"
                        flat[flat_key] = metrics
            else:
                # Если test_results не словарь, а скаляр
                flat[test_prefix] = test_results
                
        return flat