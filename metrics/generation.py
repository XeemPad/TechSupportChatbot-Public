import re
from typing import List, Dict, Any, Tuple, Union
import numpy as np


def calculate_bleu(candidate, reference):
    """Вычисление BLEU score с добавлением сглаживания"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        
        # Добавляем сглаживание
        smoothie = SmoothingFunction().method1
        
        candidate_tokens = word_tokenize(candidate.lower())
        reference_tokens = [word_tokenize(reference.lower())]
        
        # Защита от пустых текстов
        if not candidate_tokens or not reference_tokens[0]:
            return 0.0
            
        # Используем более короткие n-граммы и сглаживание
        weights = [0.5, 0.3, 0.2, 0]  # снижаем вес 3-грамм и убираем 4-граммы
        return sentence_bleu(
            reference_tokens, 
            candidate_tokens, 
            weights=weights,
            smoothing_function=smoothie
        )
    except Exception as e:
        print(f"Ошибка BLEU: {e}")
        return 0.0
    

def calculate_rouge(candidate: str, reference: str) -> float:
    """Вычисление ROUGE score"""
    from rouge import Rouge
    
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores[0]['rouge-l']['f']  # используем F1 меру для ROUGE-L


def evaluate_generation(qa_system, test_data: List[Dict]) -> Dict:
    """Комплексная оценка метрик генерации"""
    metrics = {
        "bleu": 0.0,
        "rouge-l": 0.0,
        "exact_match": 0.0,
        "token_overlap": 0.0
    }
    
    n_samples = 0
    
    for query_data in test_data:
        query = query_data.get("query", "")
        expected = query_data.get("expected_answer", "")
        
        if not query or not expected:
            continue
            
        n_samples += 1
        
        # Получаем ответ системы
        try:
            response = qa_system.answer_question(query)
            
            if isinstance(response, dict):
                generated = response.get("answer", "")
            else:
                generated = str(response)
                
            # Метрики качества текста
            metrics["bleu"] += calculate_bleu(generated, expected)
            metrics["rouge-l"] += calculate_rouge(generated, expected)
            
            # Точное совпадение (1 если ответы идентичны)
            metrics["exact_match"] += 1.0 if generated.strip().lower() == expected.strip().lower() else 0.0
            
            # Overlap токенов - более гибкая мера
            gen_words = set(re.findall(r'\b\w+\b', generated.lower()))
            exp_words = set(re.findall(r'\b\w+\b', expected.lower()))
            overlap = len(gen_words & exp_words) / max(len(exp_words), 1) if exp_words else 0
            metrics["token_overlap"] += overlap
            
        except Exception as e:
            print(f"Ошибка при генерации ответа: {e}")
            # Пропускаем этот пример, но не уменьшаем n_samples
    
    # Усредняем результаты
    if n_samples > 0:
        for key in metrics:
            metrics[key] /= n_samples
            
    return metrics
