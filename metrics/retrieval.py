import numpy as np
from typing import List, Dict, Any, Tuple, Union


def precision_at_k(retrieved_ids: List, relevant_ids: List, k: int) -> float:
    """Вычисление Precision@k"""
    if not retrieved_ids or not relevant_ids:
        return 0.0
        
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = set(retrieved_at_k) & set(relevant_ids)
    
    return len(relevant_retrieved) / min(k, len(retrieved_at_k))

def recall_at_k(retrieved_ids: List, relevant_ids: List, k: int) -> float:
    """Вычисление Recall@k"""
    if not retrieved_ids or not relevant_ids:
        return 0.0
        
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = set(retrieved_at_k) & set(relevant_ids)
    
    return len(relevant_retrieved) / len(relevant_ids)

def mean_reciprocal_rank(retrieved_ids: List, relevant_ids: List) -> float:
    """Вычисление Mean Reciprocal Rank"""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
            
    return 0.0

def ndcg_at_k(retrieved_ids: List, relevant_ids: List, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k"""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        # Для простоты считаем релевантность бинарной (1 если документ релевантен)
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / np.log2(i + 2)  # +2 т.к. i начинается с 0
    
    # Идеальная DCG (все релевантные документы впереди)
    ideal_dcg = 0.0
    for i in range(min(k, len(relevant_ids))):
        ideal_dcg += 1.0 / np.log2(i + 2)
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate_retrieval(qa_system, test_data, k_values=[1, 3, 5]):
    """Комплексная оценка метрик поиска с искусственной релевантностью"""
    metrics = {f"precision@{k}": 0.0 for k in k_values}
    metrics.update({f"recall@{k}": 0.0 for k in k_values})
    metrics["mrr"] = 0.0
    
    n_valid_samples = 0
    
    for query_data in test_data:
        query = query_data.get("query", "")
        expected_answer = query_data.get("expected_answer", "")
        
        if not query or not expected_answer:
            continue
            
        n_valid_samples += 1
        
        # Получаем результаты поиска
        results = []
        if hasattr(qa_system, 'baseline_searcher'):
            results = qa_system.baseline_searcher.search(query, top_k=max(k_values))
        
        # Оцениваем семантическую близость ответов для определения релевантности
        relevant_indices = []
        for i, result in enumerate(results):
            answer = result.get('answer', '') if isinstance(result, dict) else str(result)
            
            # Проверяем семантическую близость или текстовое совпадение
            # Простое решение - считать релевантными документы с большим пересечением слов
            expected_words = set(expected_answer.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(expected_words & answer_words) / max(len(expected_words), 1) if expected_words else 0
            
            if overlap > 0.3:  # Порог релевантности
                relevant_indices.append(i)
        
        # Вычисляем метрики на основе индексов
        for k in k_values:
            metrics[f"precision@{k}"] += len([i for i in relevant_indices if i < k]) / k
            metrics[f"recall@{k}"] += len([i for i in relevant_indices if i < k]) / max(len(relevant_indices), 1)
        
        # MRR
        if relevant_indices:
            metrics["mrr"] += 1.0 / (min(relevant_indices) + 1)
    
    # Усредняем результаты
    if n_valid_samples > 0:
        for key in metrics:
            metrics[key] /= n_valid_samples
    
    return metrics
