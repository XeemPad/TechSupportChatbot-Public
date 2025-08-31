import time
import os
import psutil
from typing import List, Dict, Any
import numpy as np


def get_memory_usage():
    """Получение текущего использования памяти процессом"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # В МБ


def evaluate_efficiency(qa_system, test_data: List[Dict]) -> Dict:
    """Оценка метрик эффективности"""
    metrics = {
        "latency_mean": 0.0,
        "latency_p90": 0.0,
        "latency_p95": 0.0,
        "memory_usage_mb": 0.0,
    }
    
    latencies = []
    memory_before = get_memory_usage()
    
    n_samples = 0
    
    # Прогреваем систему перед измерением
    try:
        warmup_query = "Как открыть ПВЗ?"
        qa_system.answer_question(warmup_query)
    except Exception as e:
        print(f"Ошибка при прогреве: {e}")
    
    # Измеряем время и память
    for query_data in test_data:
        query = query_data.get("query", "")
        if not query:
            continue
            
        n_samples += 1
        
        try:
            # Измеряем время ответа
            start_time = time.time()
            _ = qa_system.answer_question(query)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # в миллисекундах
            latencies.append(latency)
            
        except Exception as e:
            print(f"Ошибка при оценке эффективности: {e}")
    
    memory_after = get_memory_usage()
    memory_growth = memory_after - memory_before
    
    # Вычисляем метрики
    if latencies:
        metrics["latency_mean"] = np.mean(latencies)
        metrics["latency_p90"] = np.percentile(latencies, 90)
        metrics["latency_p95"] = np.percentile(latencies, 95)
        
    metrics["memory_usage_mb"] = memory_growth if memory_growth > 0 else 0.0
    
    return metrics
