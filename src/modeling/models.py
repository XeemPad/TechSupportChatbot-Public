import numpy as np

from src.modeling.base_models import (
    BaseQASearcher, BaseDocSearcher,
    SbertQASearcher, TfidfQASearcher,
    SbertDocSearcher, TfidfDocSearcher,
    SaigaRAG
)

from typing import Optional
from pathlib import Path
from abc import ABC, abstractmethod


# Базовый класс для гибридной системы
class HybridQASystem(ABC):
    def __init__(self, 
                 qa_threshold=0.7,
                 doc_threshold=0.5,
                 models_dir="./saved_models"):
        
        self.qa_threshold = qa_threshold
        self.doc_threshold = doc_threshold
        self.models_dir = Path(models_dir)
        
        # Эти поля должны быть инициализированы в дочерних классах
        self.qa_searcher: Optional[BaseQASearcher] = None
        self.doc_searcher: Optional[BaseDocSearcher] = None
    
    @abstractmethod
    def _init_searchers(self):
        """Инициализация поисковых компонентов"""
        pass
    
    def fit(self, questions, answers, documents):
        """Обучение всех компонентов системы"""
        # Инициализируем поисковые компоненты если еще не созданы
        if self.qa_searcher is None or self.doc_searcher is None:
            self._init_searchers()
            
        # Обучаем QA компонент
        print(f"Обучение QA поисковика...")
        self.qa_searcher.fit(questions, answers)  # type: ignore
        
        # Обучаем DOC компонент
        print(f"Обучение DOC модели...")
        self.doc_searcher.fit(documents)  # type: ignore
    
    def answer_question(self, query):
        """Основной метод для получения ответа с умным выбором и весами"""
        # Получаем результаты от обеих систем
        qa_results = self.qa_searcher.search(query, top_k=3)  # type: ignore
        context, doc_scores = self.doc_searcher.retrieve(query, top_k=3)  # type: ignore
        
        best_qa_score = qa_results[0]['score'] if qa_results else 0.0
        best_doc_score = doc_scores.max()  if len(doc_scores) > 0 else 0.0
        
        # Логика выбора с весами:
        if best_qa_score >= self.qa_threshold:  # QA база дает хороший результат
            return {
                'answer': qa_results[0]['answer'],  # type: ignore
                'source': 'qa_database',
                'confidence': best_qa_score,
                'similar_question': qa_results[0]['question'],  # type: ignore
                'reason': f'QA база: {best_qa_score:.3f} vs DOC: {best_doc_score:.3f}'
            }
        elif best_doc_score >= self.doc_threshold:  # DOC дает хороший результат
            generated_answer = self.generate_answer(context, query)
            return {
                'answer': generated_answer,
                'source': 'generated_from_docs',
                'context': context,
                'confidence': best_doc_score,
                'reason': f'DOC: {best_doc_score:.3f} vs QA: {best_qa_score:.3f}'
            }
        else:  # Оба результата плохие - возвращаем лучший из плохих
            if best_qa_score >= best_doc_score:
                return {
                    'answer': qa_results[0]['answer'] if qa_results else "Ответ не найден",
                    'source': 'qa_database_low_confidence',
                    'confidence': best_qa_score,
                    'similar_question': qa_results[0]['question'] if qa_results else "",
                    'reason': f'Низкая уверенность. QA: {best_qa_score:.3f} vs DOC: {best_doc_score:.3f}'
                }
            else:
                generated_answer = self.generate_answer(context, query)
                return {
                    'answer': generated_answer,
                    'source': 'generated_from_docs_low_confidence',
                    'context': context,
                    'confidence': best_doc_score,
                    'reason': f'Низкая уверенность. DOC: {best_doc_score:.3f} vs QA: {best_qa_score:.3f}'
                }

    def generate_answer(self, context, question):
        """Простая генерация ответа на основе контекста"""
        # Для демонстрационных целей используем простую шаблонную генерацию
        if not context or len(context) < 20:
            return "Недостаточно информации для ответа на ваш вопрос."
            
        context_parts = context.split("\n\n")
        most_relevant = context_parts[0] if context_parts else context
        
        # Поиск ключевых фраз в контексте на основе слов из вопроса
        question_words = set(question.lower().split())
        key_sentences = []
        
        for sentence in most_relevant.split(". "):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = set(sentence.lower().split())
            # Если есть пересечение с вопросом
            if sentence_words.intersection(question_words):
                key_sentences.append(sentence)
        
        if key_sentences:
            return ". ".join(key_sentences) + "."
        else:
            # Если нет хороших совпадений, берём первый абзац
            return most_relevant

    @staticmethod
    def get_short_name() -> str:
        return "AbstractHybridQASystem"


# Конкретная реализация для TF-IDF
class TfidfHybridQASystem(HybridQASystem):
    def __init__(self, 
                 qa_threshold=0.6,
                 doc_threshold=0.4,
                 **kwargs):
        super().__init__(
            qa_threshold=qa_threshold,
            doc_threshold=doc_threshold,
            **kwargs
        )
        self._init_searchers()
        
    def _init_searchers(self):
        """Инициализация TF-IDF поисковиков"""
        self.qa_searcher = TfidfQASearcher(models_dir=self.models_dir)
        self.doc_searcher = TfidfDocSearcher(models_dir=self.models_dir)
    
    @staticmethod
    def get_short_name() -> str:
        return "Hybrid_TFIDF_system"

# Конкретная реализация для SBERT
class SbertHybridQASystem(HybridQASystem):
    def __init__(self, 
                 qa_threshold=0.65,
                 doc_threshold=0.55,
                 model_name="ai-forever/sbert_large_nlu_ru",
                 **kwargs):
        super().__init__(
            qa_threshold=qa_threshold,
            doc_threshold=doc_threshold,
            **kwargs
        )
        self.model_name = model_name
        self._init_searchers()
        
    def _init_searchers(self):
        """Инициализация SBERT поисковиков"""
        self.qa_searcher = SbertQASearcher(
            models_dir=self.models_dir,
            model_name=self.model_name
        )
        self.doc_searcher = SbertDocSearcher(
            models_dir=self.models_dir,
            model_name=self.model_name
        )

    @staticmethod
    def get_short_name() -> str:
        return "Hybrid_SBERT_system"

class SaigaRagQASystem(HybridQASystem):
    """RAG система на основе Saiga LLM"""
    
    def __init__(self, 
                 qa_searcher=None,  # Опционально: существующий QA searcher
                 doc_searcher=None, # Опционально: существующий Doc searcher
                 model_params=dict(),
                 **kwargs):
        super().__init__(**kwargs)
        self.model_params = model_params
        
        # Сохраняем ссылки на внешние поисковики, если есть
        self.external_qa_searcher = qa_searcher
        self.external_doc_searcher = doc_searcher
        
        self._init_searchers()

    def _init_searchers(self):
        """Инициализация поисковиков с возможностью переиспользования"""
        # Используем внешние поисковики если они переданы, иначе создаем новые
        if self.external_qa_searcher is not None:
            self.qa_searcher = self.external_qa_searcher
            print("Используем внешний QA searcher для Saiga")
        else:
            self.qa_searcher = SbertQASearcher(models_dir=self.models_dir)
            
        if self.external_doc_searcher is not None:
            self.doc_searcher = self.external_doc_searcher
            print("Используем внешний Doc searcher для Saiga")
        else:
            self.doc_searcher = SbertDocSearcher(models_dir=self.models_dir)
            
        # Инициализируем Saiga
        self.saiga = SaigaRAG(models_dir=self.models_dir, **self.model_params)
    
    def fit(self, questions, answers, documents):
        """Обучение модели с учетом переиспользования поисковиков"""
        # Обучаем только если используем собственные поисковики
        if self.external_qa_searcher is None:
            print("Обучаем QA searcher для Saiga...")
            self.qa_searcher.fit(questions, answers)  # type: ignore
        else:
            print("Используем предобученный QA searcher для Saiga")
            
        if self.external_doc_searcher is None:
            print("Обучаем Doc searcher для Saiga...")
            self.doc_searcher.fit(documents)  # type: ignore
        else:
            print("Используем предобученный Doc searcher для Saiga")
        
    def answer_question(self, query):
        """Генерация ответа на основе QA + документов"""
        # Все остается как было в вашей реализации
        qa_results = self.qa_searcher.search(query, top_k=3)  # type: ignore
        qa_score = qa_results[0]['score'] if qa_results else 0
        
        context, doc_scores = self.doc_searcher.retrieve(query, top_k=3)  # type: ignore
        doc_score = doc_scores[0] if len(doc_scores) > 0 else 0
        
        if qa_score >= self.qa_threshold and qa_score >= doc_score:
            # Используем готовый ответ из QA
            answer = qa_results[0]['answer']  # type: ignore
            source = "qa_exact" if qa_score > 0.9 else "qa_similar"
            confidence = qa_score
            
            return {
                "answer": answer,
                "source": source,
                "confidence": confidence,
                "similar_question": qa_results[0]['question'],  # type: ignore
                "reason": f"Найден похожий вопрос с уверенностью {qa_score:.2f}"
            }
            
        elif doc_score >= self.doc_threshold:
            # Генерируем ответ с помощью Saiga

            if qa_score > 0.3 and qa_results:
                similar_qa = f"\n\nПохожий вопрос: {qa_results[0]['question']}\nОтвет: {qa_results[0]['answer']}"
                if isinstance(context, list):
                    context.extend(similar_qa)
                else:
                    context += similar_qa

            answer, prompt, params = self.saiga.generate(context, query)
            
            
            return {
                "answer": answer,
                "source": "rag_generated",
                "confidence": doc_score,
                "context": context,
                "prompt": prompt,
                "generation_params": params,
                "reason": f"Сгенерировано на основе документов с уверенностью {doc_score:.2f}"
            }
            
        else:
            # Недостаточно информации
            return {
                "answer": "К сожалению, я не нашел достаточно информации для ответа на этот вопрос.",
                "source": "fallback",
                "confidence": max(qa_score, doc_score),
                "reason": "Недостаточная уверенность для ответа"
            }

    def __getstate__(self):
        """Управление сериализацией для pickle"""
        state = self.__dict__.copy()

        # Если saiga имеет свой __getstate__, используем его для корректной сериализации
        if hasattr(self.saiga, '__getstate__'):
            state['saiga'] = self.saiga.__getstate__()
        return state
    
    def __setstate__(self, state):
        """Восстановление после десериализации"""
        self.__dict__.update(state)
        
        # Проверяем, есть ли saiga и нужно ли восстановить только модель и токенизатор
        if hasattr(self, 'saiga') and self.saiga:
            if not hasattr(self.saiga, 'model') or not hasattr(self.saiga, 'tokenizer'):
                try:
                    print("Восстанавливаем модель и токенизатор...")
                    
                    self.saiga._init_model()
                    print("Модель успешно восстановлена")
                except Exception as e:
                    raise RuntimeError(f"Ошибка восстановления модели: {e}")

    @staticmethod
    def get_short_name() -> str:
        return "Hybrid_Sbert-Saiga_system"