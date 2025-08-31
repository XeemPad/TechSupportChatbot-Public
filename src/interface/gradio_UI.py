import gradio as gr
import pandas as pd
import numpy as np

import pathlib
import time

import json
import pickle

# Импорты моделей
from src.modeling.models import (
    HybridQASystem, 
    TfidfHybridQASystem, SbertHybridQASystem, SaigaRagQASystem
)
from src.modeling.data_preprocessing import DataPreprocessor
from metrics.evaluator import QAEvaluator


CACHE_EXTENSION: str = ".pkl"

#TODO: create class ProccessedData consisting of what now is self.qa_df, kb_db etc.


class QASystemInterface:
    """
    Класс, предоставляющий интерфейс для фронтенда (на Gradio),
    инициализирующий все модели
    """
    
    def __init__(self, work_dir: pathlib.Path = pathlib.Path("./"), save_to_cache: bool = True, offload: bool = False):
        self.work_dir = work_dir
        self.data_dir = work_dir / "data"
        self.cache_dir = work_dir / "qa_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.save_to_cache = save_to_cache
        self.models: dict[str, HybridQASystem] = {}
        self.offload = offload
        self.load_data()
        self.initialize_models()

    def load_data(self):
        """Загрузка предобработанных данных"""
        print("\tЗагружаем предобработанные данные...")
        
        processed_dir = self.work_dir / "processed_data"
        
        # Проверяем наличие обработанных файлов
        if not processed_dir.exists():
            print("Предобработанные данные не найдены. Запускаем предобработку...")
            
            try:
                preprocessor = DataPreprocessor(work_dir=self.work_dir)
                preprocessor.process_all()
            except Exception as e:
                raise Exception("Error during DataPreproccessing in QASystemInterface.load_data():\n", str(e))
        
        # Загружаем обработанные данные
        self.qa_df = pd.read_csv(processed_dir / 'qa_pairs_processed.csv')
        self.kb_df = pd.read_csv(processed_dir / 'knowledge_chunks_processed.csv') 
        self.docs_info_df = pd.read_csv(processed_dir / 'documents_info.csv')
        self.unified_df = pd.read_csv(processed_dir / 'unified_dataset.csv')
        
        print(f"Загружено QA пар: {len(self.qa_df)}")
        print(f"Загружено информационных чанков: {len(self.kb_df)}")
        print(f"Загружено документов: {len(self.docs_info_df)}")
        print(f"Единый датасет: {len(self.unified_df)} записей")

    def initialize_models(self):
        """Инициализация моделей с кэшированием"""
        print("\tИнициализируем модели...")
        
        self._create_tfidf_model()

        # Кэш файлы
        sbert_cache = self.cache_dir / "hybrid_sbert_system.pkl"
        saiga_cache = self.cache_dir / "saiga_rag_system.pkl"
        
        
        # Hybrid SBERT
        if sbert_cache.exists():
            print("Загружаем Hybrid SBERT из кэша...")
            try:
                with open(sbert_cache, 'rb') as f:
                    self.models['hybrid_sbert'] = pickle.load(f)
                print("Hybrid SBERT загружен из кэша!")
            except Exception as e:
                print(f"Ошибка загрузки кэша: {e}")
                print("Создаем новую модель...")
                self._create_sbert_model(sbert_cache)
        else:
            print("Создаем новую Hybrid SBERT модель...")
            self._create_sbert_model(sbert_cache)
        
        # Saiga RAG
        try:            
            if saiga_cache.exists():
                print("Загружаем Saiga RAG из кэша...")
                try:
                    with open(saiga_cache, 'rb') as f:
                        self.models['saiga_rag'] = pickle.load(f)
                    print("Saiga RAG загружен из кэша!")
                except Exception as e:
                    print(f"Ошибка загрузки кэша Saiga: {e}")
                    print("Создаем новую модель Saiga...")
                    self._create_saiga_model(saiga_cache)
            else:
                print("Создаем новую модель Saiga RAG...")
                self._create_saiga_model(saiga_cache)
        except Exception as e:
            print(f"Не удалось инициализировать Saiga: {e}")
            print("Продолжаем без модели Saiga")
        
        print("\tВсе модели готовы!")

    def _create_tfidf_model(self):
        cache_name: str = TfidfHybridQASystem.get_short_name() + CACHE_EXTENSION
        cache_file: pathlib.Path = self.cache_dir / cache_name

        if cache_file.exists():
            print("Загружаем Hybrid TF-IDF из кэша...")
            try:
                with open(tfidf_cache, 'rb') as f:
                    self.models['hybrid_tfidf'] = pickle.load(f)
                print("Hybrid TF-IDF загружен из кэша!")
            except Exception as e:
                print(f"Ошибка загрузки кэша: {e}")
                print("Создаем новую модель...")
                self._create_tfidf_model(tfidf_cache)
        else:
            print("Создаем новую Hybrid TF-IDF модель...")
            self._create_tfidf_model(tfidf_cache)
        
        self.models['hybrid_tfidf'] = TfidfHybridQASystem(
            qa_threshold=0.7,   
            doc_threshold=0.6,  
            models_dir=str(self.cache_dir / "tfidf_models")
        )
        self.models['hybrid_tfidf'].fit(
            self.qa_df['question'].tolist(),
            self.qa_df['answer'].tolist(),
            self.kb_df['text'].tolist()
        )

        if self.save_to_cache:
            self.models['hybrid_tfidf']
            # Сохраняем в кэш
            print(f"Сохранение TF-IDF в кэш...")

            with open(cache_file, 'wb') as f:
                pickle.dump(hybrid_tfidf, f)
            print(f"TF-IDF модель сохранена в кэш: {cache_file}")

    def _create_sbert_model(self, cache_file):
        """Создание SBERT модели с оптимизированными параметрами"""
        hybrid_sbert = SbertHybridQASystem(
            qa_threshold=0.75,
            doc_threshold=0.55, 
            models_dir=str(self.cache_dir / "sbert_models")
        )
        hybrid_sbert.fit(
            self.qa_df['question'].tolist(),
            self.qa_df['answer'].tolist(),
            self.kb_df['text'].tolist()
        )
        self.models['hybrid_sbert'] = hybrid_sbert
        
        if self.save_to_cache:
            print(f"Сохранение SBERT в кэш...")
            # Сохраняем в кэш
            with open(cache_file, 'wb') as f:
                pickle.dump(hybrid_sbert, f)
            print(f"SBERT модель сохранена в кэш: {cache_file}")
    
    def _create_saiga_model(self, cache_file):
        """Создание Saiga RAG модели с переиспользованием поисковиков SBERT"""
        
        # Проверяем, есть ли уже SBERT модель
        sbert_model = self.models.get('hybrid_sbert')
        
        if sbert_model is not None:
            print("Переиспользуем поисковики SBERT для Saiga")
            qa_searcher = sbert_model.qa_searcher
            doc_searcher = sbert_model.doc_searcher
        else:
            print("SBERT модель не найдена, создаем новые поисковики для Saiga")
            qa_searcher = None
            doc_searcher = None
        
        # Создаем модель Saiga с оптимизированными параметрами
        saiga_params = {
            'temperature': 0.6,
            'top_p': 0.85,
            'repetition_penalty': 1.05,
            'no_rep_ngram_size': 8,
            'max_tokens': 256,
            'do_sample': False,

            'offload': self.offload  
        }
        saiga_model = SaigaRagQASystem(
            qa_threshold=0.8,
            doc_threshold=0.3,
            models_dir=str(self.cache_dir / "saiga_models"),
            qa_searcher=qa_searcher,
            doc_searcher=doc_searcher,
            model_params=saiga_params
        )
        
        # Обучаем модель
        saiga_model.fit(
            self.qa_df['question'].tolist(),
            self.qa_df['answer'].tolist(),
            self.kb_df['text'].tolist()
        )
        
        self.models['saiga_rag'] = saiga_model
        if self.save_to_cache:
            print(f"Сохранение Saiga в кэш...")
            
            # Сохраняем в кэш
            with open(cache_file, 'wb') as f:
                pickle.dump(saiga_model, f)
            print(f"Saiga модель сохранена в кэш: {cache_file}")

    def _get_cache_name(self, qa_system: HybridQASystem):
        return qa_system.get_short_name()

    def _load_system_from_cache()

    def init_metrics(self):
        """Инициализация системы метрик"""
        from metrics.test_sets import TestSetManager
        
        # Создаем менеджер тестовых наборов
        self.test_manager = TestSetManager(work_dir=self.work_dir)
        
        # Если тестовых наборов нет, создаем
        if not self.test_manager.list_test_sets():
            print("Создаем стратифицированную выборку...")
            self.test_manager.create_stratified_sample(self.qa_df, self.kb_df)

    def import_test_set(self, csv_path, name="imported_test"):
        """Импорт тестового набора из CSV-файла и его оценка"""
        if not hasattr(self, 'test_manager'):
            self.init_metrics()
            
        # Импортируем тестовый набор
        file_path = self.test_manager.import_test_set_from_csv(csv_path)
        if not file_path:
            return "Ошибка при импорте файла", None
            
        print(f"Тестовый набор успешно импортирован как '{name}'")
        return f"Импортирован тестовый набор '{name}' из {csv_path}", None

    def load_test_set(self):
        """Загрузка тестового набора в каталоге проекта"""
        if not hasattr(self, 'test_manager'):
            self.init_metrics()
            
        # Импортируем тестовый набор из файла проекта
        file_path = self.test_manager.import_test_set_from_csv()
        
        if not file_path:
            print("Ошибка при загрузке тестовой выборки из проекта")
            return None
            
        print(f"Тестовый набор из проекта успешно загружен как 'project_test_data'")
        
        # Обновляем список доступных наборов в интерфейсе (если используется Gradio)
        return self.test_manager.test_sets['project_test_data']

    def evaluate_model(self, model_name, test_set_name):
        """Оценка одной модели на конкретном тестовом наборе"""
        from metrics.evaluator import QAEvaluator
        
        if model_name not in self.models:
            return f"Модель {model_name} не найдена", None
        
        model = self.models[model_name]
        test_set = self.load_test_set()
        
        if not test_set:
            return f"Тестовый набор {test_set_name} не найден", None
        
        # Создаем оценщика
        evaluator = QAEvaluator(model, {test_set_name: test_set}, work_dir=self.work_dir)
        
        # Запускаем оценку
        results = evaluator.evaluate_test_set(test_set)
        
        # Форматируем вывод
        output = f"## Результаты оценки модели {model_name} на наборе {test_set_name}:\n\n"
        
        for category, metrics in results.items():
            output += f"### {category.upper()}:\n"
            for name, value in metrics.items():
                output += f"- {name}: {value:.4f}\n"
            output += "\n"
        
        # Создаем и сохраняем график
        fig = evaluator.plot_for_gradio()
        evaluator.save_results(f"{model_name}_{test_set_name}")
        
        return output, fig

    def compare_models(self, model_names, test_set_name):
        """Сравнение моделей с улучшенной визуализацией"""
        
        test_set = self.load_test_set()
        if not test_set:
            return f"Тестовый набор {test_set_name} не найден", None
        
        results = {}
        
        for model_name in model_names:
            if model_name not in self.models:
                continue
                
            model = self.models[model_name]
            
            # Создаем оценщика для каждой модели
            evaluator = QAEvaluator(model, {test_set_name: test_set}, work_dir=self.work_dir)
            model_results = evaluator.evaluate_test_set(test_set)
            
            # Сохраняем результаты
            results[model_name] = {test_set_name: model_results}
        
        if not results:
            return "Ни одна из указанных моделей не найдена", None
        
        # Для графика используем первую модель как основную
        primary_model = list(results.keys())[0]
        primary_evaluator = QAEvaluator(self.models[primary_model], work_dir=self.work_dir)
        primary_evaluator.results = {test_set_name: results[primary_model][test_set_name]}
        
        # Создаем сравнительный график
        other_results = {name: res for name, res in results.items() if name != primary_model}
        fig = primary_evaluator.plot_for_gradio(other_results=other_results)
        
        # Форматируем текстовый вывод
        output = f"## Сравнение моделей на наборе {test_set_name}\n\n"
        
        for model_name, model_data in results.items():
            output += f"### {model_name}\n"
            
            for test_name, test_results in model_data.items():
                for category, metrics in test_results.items():
                    output += f"**{category}**: "
                    output += ", ".join([f"{name}: {value:.4f}" for name, value in metrics.items() 
                                        if "mean" in name or "@1" in name or name in ["bleu", "rouge-l"]])
                    output += "\n"
        
        return output, fig

    def clear_cache(self):
        """Очистка кэша для пересоздания моделей"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print("Кэш очищен")
    
    def answer_question_hybrid(self, question: str, model_type: str = "sbert"):
        """Ответ через гибридную систему"""
        if not question.strip():
            return "Пожалуйста, введите вопрос", ""
        
        model_key = f"hybrid_{model_type}"
        if model_key not in self.models:
            return "Модель не найдена", ""
            
        start_time = time.time()
        result = self.models[model_key].answer_question(question)
        inference_time = time.time() - start_time
        
        # Детали
        details = f"**Время ответа:** {inference_time*1000:.1f}ms\n\n"
        details += f"**Источник:** {result['source']}\n"
        details += f"**Уверенность:** {result['confidence']:.3f}\n\n"
        
        if 'similar_question' in result:
            details += f"**Похожий вопрос:** {result['similar_question']}\n\n"
        
        if 'context' in result:
            details += f"**Использованный контекст:**\n{result['context'][:300]}...\n\n"
        
        if 'reason' in result:
            details += f"**Логика выбора:** {result['reason']}\n\n"
        
        return result['answer'], details
    
    def answer_question_saiga(self, question: str):
        """Ответ через Saiga RAG"""
        if not question.strip():
            return "Пожалуйста, введите вопрос", ""
        
        if 'saiga_rag' not in self.models:
            return "Модель Saiga не загружена", ""
            
        start_time = time.time()
        result = self.models['saiga_rag'].answer_question(question)
        inference_time = time.time() - start_time
        
        # Детали с информацией о скорости генерации
        details = f"**Время ответа:** {inference_time*1000:.1f}ms\n\n"
        details += f"**Источник:** {result['source']}\n"
        details += f"**Уверенность:** {result['confidence']:.3f}\n\n"
        
        if 'context' in result:
            details += f"**Использованный контекст:**\n{result['context'][:300]}...\n\n"
        
        if 'prompt' in result:
            details += f"**Использованный промпт:**\n```\n{result['prompt']}\n```\n\n"
            
        if 'generation_params' in result:
            details += f"**Параметры генерации:**\n"
            for k, v in result['generation_params'].items():
                details += f"- {k}: {v}\n"
        
        return result['answer'], details

    def compare_all_models(self, question: str):
        """Сравнение всех моделей"""
        if not question.strip():
            return "Введите вопрос для сравнения", "", "", "", ""
        
        # TF-IDF гибрид
        tfidf_answer, tfidf_details = self.answer_question_hybrid(question, "tfidf")
        
        # SBERT гибрид
        sbert_answer, sbert_details = self.answer_question_hybrid(question, "sbert")
        
        # Saiga RAG
        saiga_available = 'saiga_rag' in self.models
        if saiga_available:
            saiga_answer, saiga_details = self.answer_question_saiga(question)
        else:
            saiga_answer = "Модель Saiga не загружена"
            saiga_details = "Недоступно"
        
        # Сводка
        summary = f"## Сравнение всех моделей для вопроса:\n**{question}**\n\n"
        summary += f"### TF-IDF (базовый подход)\n"
        summary += f"**Ответ:** {tfidf_answer[:150]}...\n\n"
        summary += f"### SBERT (векторные эмбеддинги)\n"
        summary += f"**Ответ:** {sbert_answer[:150]}...\n\n"
        summary += f"### Saiga RAG (нейросетевая генерация)\n"
        summary += f"**Ответ:** {saiga_answer[:150]}...\n\n"
        
        return summary, tfidf_answer, sbert_answer, saiga_answer, \
            f"TF-IDF: {tfidf_details}\n\nSBERT: {sbert_details}\n\nSaiga: {saiga_details}"
    
    def get_random_questions(self):
        """Получить случайные вопросы из базы"""
        sample_questions = self.qa_df['question'].sample(5).tolist()
        return "\n".join([f"• {q}" for q in sample_questions])
    
    def get_stats(self):
        """Статистика по обработанным данным"""
        stats = f"""
## Статистика данных

**QA Пары:**
- Всего пар: {len(self.qa_df):,}
- Средняя длина вопроса: {self.qa_df['length_q'].mean():.0f} символов
- Средняя длина ответа: {self.qa_df['length_a'].mean():.0f} символов

**База знаний:**
- Всего информационных чанков: {len(self.kb_df):,}
- Уникальных документов: {len(self.docs_info_df)}
- Средняя длина чанка: {self.kb_df['length'].mean():.0f} символов

**Источники QA пар:**
"""
        
        # Статистика по источникам QA
        source_counts = self.qa_df['source'].value_counts()
        for source, count in source_counts.items():
            stats += f"- {source}: {count} пар\n"
        
        stats += "\n**Типы документов в базе знаний:**\n"
        
        # Статистика по типам документов
        type_counts = self.kb_df['source_type'].value_counts()
        for doc_type, count in type_counts.items():
            stats += f"- {doc_type}: {count} чанков\n"
        
        stats += "\n**Документы:**\n"
        for _, row in self.docs_info_df.iterrows():
            stats += f"- {row['title']}: {row['chunks_count']} чанков ({row['source_type']})\n"
        
        return stats


# Создаем интерфейс
def create_interface(**kwargs):
    print("\tЗапускаем QA систему...")
    qa_system: QASystemInterface = QASystemInterface(**kwargs)
    cache_dir = qa_system.cache_dir

    # CSS стили
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 20px;
    }
    .model-tab {
        border-radius: 10px;
        padding: 10px;
    }
    .vs-header {
        text-align: center;
        color: #E63946; 
        font-size: 1.2em;
        margin: 15px 0;
    }
    """
    print("\tИнициализируем веб-приложение...")
    
    with gr.Blocks(css=css, title="WB Tech Support AI") as demo:
        
        gr.HTML("""
        <div class="main-header">
            <h1>WB Tech Support QA System</h1>
            <p>Сравнение разных подходов</p>
        </div>
        """)
        
        with gr.Tabs():
            
            # Вкладка Hybrid TF-IDF
            with gr.TabItem("Baseline TF-IDF", elem_classes="model-tab"):
                gr.Markdown("### Классический подход: TF-IDF")
                gr.Markdown("**Особенности:** Быстро, надежно, работает с ключевыми словами")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        tfidf_input = gr.Textbox(
                            label="Ваш вопрос",
                            placeholder="Например: Как оформить возврат товара?",
                            lines=2
                        )
                        tfidf_btn = gr.Button("Найти ответ (TF-IDF)", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("**Примеры вопросов:**")
                        sample_questions_tfidf = gr.Textbox(
                            value=qa_system.get_random_questions(),
                            label="Случайные вопросы из базы",
                            lines=6,
                            interactive=False
                        )
                        refresh_tfidf_btn = gr.Button("Обновить примеры")
                
                tfidf_answer = gr.Textbox(label="Ответ", lines=6)
                tfidf_details = gr.Markdown(label="Детали работы системы")
            
            # Вкладка Hybrid SBERT
            with gr.TabItem("SBERT-large", elem_classes="model-tab"):
                gr.Markdown("### SBERT эмбеддинги от SberAI")
                gr.Markdown("**Особенности:** Понимает семантику, контекст, синонимы")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        sbert_input = gr.Textbox(
                            label="Ваш вопрос",
                            placeholder="Например: Что делать если товар не пришел?",
                            lines=2
                        )
                        sbert_btn = gr.Button("Найти ответ (SBERT)", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("**Примеры вопросов:**")
                        sample_questions_sbert = gr.Textbox(
                            value=qa_system.get_random_questions(),
                            label="Случайные вопросы из базы",
                            lines=6,
                            interactive=False
                        )
                        refresh_sbert_btn = gr.Button("Обновить примеры")
                
                sbert_answer = gr.Textbox(label="Ответ", lines=6)
                sbert_details = gr.Markdown(label="Детали работы системы")
            
            # Вкладка Saiga RAG
            with gr.TabItem("Saiga RAG (LLM)", elem_classes="model-tab"):
                gr.Markdown("### Генеративная модель: Saiga на базе Mistral-7B")
                gr.Markdown("**Особенности:** Генерирует собственные ответы на основе релевантного контекста")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        saiga_input = gr.Textbox(
                            label="Ваш вопрос",
                            placeholder="Например: Что нужно знать о сдаче невыкупа?",
                            lines=2
                        )
                        saiga_btn = gr.Button("Найти ответ (Saiga)", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("**Примеры вопросов:**")
                        sample_questions_saiga = gr.Textbox(
                            value=qa_system.get_random_questions(),
                            label="Случайные вопросы из базы",
                            lines=6,
                            interactive=False
                        )
                        refresh_saiga_btn = gr.Button("Обновить примеры")
                
                saiga_answer = gr.Textbox(label="Ответ", lines=8)
                saiga_details = gr.Markdown(label="Детали работы системы")
            
            # Вкладка Прямое сравнение
            with gr.TabItem("Сравнение моделей"):
                gr.Markdown("### Сравнение всех подходов")
                gr.Markdown("**Задайте один вопрос и получите ответы от всех систем**")
                
                compare_input = gr.Textbox(
                    label="Вопрос для сравнения",
                    placeholder="Введите вопрос чтобы увидеть разницу подходов",
                    lines=2
                )
                compare_btn = gr.Button("Сравнить подходы", variant="primary", size="lg")
                
                comparison_summary = gr.Markdown(label="Сводка сравнения")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### TF-IDF")
                        tfidf_comp = gr.Textbox(label="Ответ TF-IDF", lines=6)
                    
                    with gr.Column():
                        gr.Markdown("#### SBERT")
                        sbert_comp = gr.Textbox(label="Ответ SBERT", lines=6)
                        
                    with gr.Column():
                        gr.Markdown("#### Saiga")
                        saiga_comp = gr.Textbox(label="Ответ Saiga", lines=6)
                
                comparison_details = gr.Markdown(label="Подробная информация")

            with gr.TabItem("Метрики"):
                    gr.Markdown("### Оценка качества QA системы")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Загрузка тестовой выборки из проекта")
                            load_test_button = gr.Button("Загрузить тестовый набор из data/test.csv")
                        
                        with gr.Column():
                            load_status = gr.Markdown()

                    # Добавьте обработчик
                    load_test_button.click(
                        lambda: "Тестовый набор успешно загружен из data/test.csv" 
                                if qa_system.load_test_set() is not None 
                                else "Ошибка при загрузке тестового набора",
                        inputs=[],
                        outputs=[load_status]
                    )

                    with gr.Row():
                        with gr.Column():
                            model_dropdown = gr.Dropdown(
                                choices=["hybrid_tfidf", "hybrid_sbert", "saiga_rag"],
                                value="saiga_rag",
                                label="Модель для оценки"
                            )
                            
                            test_set_dropdown = gr.Radio(
                                choices=["standard", "paraphrased", "misspelled", "special_cases", "full"],
                                value="full",
                                label="Тестовый набор"
                            )
                            
                            evaluate_button = gr.Button("Запустить оценку")
                            
                        with gr.Column():
                            metrics_output = gr.Markdown(label="Результаты оценки")
                            metrics_plot = gr.Plot(label="Визуализация метрик")
                    
                    gr.Markdown("### Сравнение моделей")
                    with gr.Row():
                        with gr.Column():
                            compare_test_set = gr.Radio(
                                choices=["standard", "paraphrased", "misspelled", "special_cases", "full"],
                                value="full",
                                label="Тестовый набор для сравнения"
                            )
                            compare_models_button = gr.Button("Сравнить модели")
                            
                        with gr.Column():
                            compare_output = gr.Markdown(label="Результаты сравнения")
                            compare_plot = gr.Plot(label="График сравнения")
                    
                    # Обработчик оценки одной модели
                    evaluate_button.click(
                        qa_system.evaluate_model,
                        inputs=[model_dropdown, test_set_dropdown],
                        outputs=[metrics_output, metrics_plot]
                    )
                    
                    # Обработчик сравнения моделей
                    compare_models_button.click(
                        lambda test_set: qa_system.compare_models(["hybrid_tfidf", "hybrid_sbert", "saiga_rag"], test_set),
                        inputs=[compare_test_set],
                        outputs=[compare_output, compare_plot]
                    )

            with gr.TabItem("Настройки моделей"):
                gr.Markdown("### Настройка параметров моделей")
                
                with gr.Tabs():
                    with gr.TabItem("SBERT"):
                        # Существующие настройки SBERT
                        qa_threshold_slider = gr.Slider(
                            minimum=0.3, maximum=0.9, value=0.55, step=0.05,
                            label="Порог для QA ответов (выше = строже)"
                        )
                
                        doc_threshold_slider = gr.Slider(
                            minimum=0.2, maximum=0.8, value=0.45, step=0.05,
                            label="Порог для DOC ответов"
                        )

                        test_question = gr.Textbox(
                            label="Тестовый вопрос",
                            placeholder="Как вернуть БАДы?"
                        )
                        
                        test_btn = gr.Button("Протестировать SBERT с новыми настройками")
                        
                        test_output = gr.Textbox(label="Результат", lines=8)

                    with gr.TabItem("Saiga"):
                        gr.Markdown("### Параметры генерации Saiga")
                        
                        temperature_slider = gr.Slider(
                            minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                            label="Temperature (разнообразие ответов)"
                        )
                        
                        top_p_slider = gr.Slider(
                            minimum=0.5, maximum=1.0, value=0.9, step=0.05,
                            label="Top-P (ядро вероятности)"
                        )
                        
                        max_tokens_slider = gr.Slider(
                            minimum=64, maximum=512, value=256, step=32,
                            label="Максимальное количество токенов"
                        )

                        qa_threshold_slider_saiga = gr.Slider(
                            minimum=0.3, maximum=0.9, value=0.55, step=0.05,
                            label="Порог для QA ответов (выше = строже)"
                        )
                
                        doc_threshold_slider_saiga = gr.Slider(
                            minimum=0.2, maximum=0.8, value=0.45, step=0.05,
                            label="Порог для DOC ответов"
                        )
                        
                        saiga_test_question = gr.Textbox(
                            label="Тестовый вопрос",
                            placeholder="Что делать, если клиент просит кэшбэк?"
                        )
                        
                        saiga_test_btn = gr.Button("Протестировать настройки Saiga")
                        
                        saiga_test_output = gr.Textbox(label="Результат", lines=8)
            
            # Вкладка Статистика
            with gr.TabItem("Статистика данных"):
                gr.Markdown(qa_system.get_stats())

        # Обработчик для смены порогов и весов
        def test_with_params(question, qa_thresh, doc_thresh):
            # Создаем временную модель с новыми параметрами
            temp_model = SbertHybridQASystem(
                qa_threshold=qa_thresh,
                doc_threshold=doc_thresh,
                models_dir=str(cache_dir / "sbert_models")
            )
            # Используем уже обученные компоненты
            temp_model.qa_searcher = qa_system.models['hybrid_sbert'].qa_searcher
            temp_model.doc_searcher = qa_system.models['hybrid_sbert'].doc_searcher
            
            result = temp_model.answer_question(question)
            return f"Ответ: {result['answer']}\n\nУверенность: {result['confidence']:.3f}\nИсточник: {result['source']}"

        test_btn.click(
            test_with_params,
            inputs=[test_question, qa_threshold_slider, doc_threshold_slider],
            outputs=test_output
        )

        def test_saiga_params(question, temp, top_p, max_tokens, qa_thresh, doc_thresh):
            if 'saiga_rag' not in qa_system.models:
                return "Модель Saiga не загружена"
                
            # Создаем копию модели с новыми параметрами
            saiga_model = qa_system.models['saiga_rag']
            saiga_model.temperature = temp
            saiga_model.top_p = top_p
            saiga_model.max_tokens = max_tokens
            saiga_model.qa_threshold = qa_thresh
            saiga_model.doc_threshold = doc_thresh
            
            result = saiga_model.answer_question(question)
            return f"Ответ: {result['answer']}\n\nИсточник: {result['source']}"
        
        saiga_test_btn.click(
            test_saiga_params,
            inputs=[saiga_test_question, temperature_slider, top_p_slider, max_tokens_slider, 
                    qa_threshold_slider_saiga, doc_threshold_slider_saiga],
            outputs=saiga_test_output
        )

        saiga_btn.click(
            qa_system.answer_question_saiga,
            inputs=saiga_input,
            outputs=[saiga_answer, saiga_details]
        )

        refresh_saiga_btn.click(
            qa_system.get_random_questions,
            outputs=sample_questions_saiga
        )
        
        # Обработчики событий
        tfidf_btn.click(
            lambda q: qa_system.answer_question_hybrid(q, "tfidf"),
            inputs=tfidf_input,
            outputs=[tfidf_answer, tfidf_details]
        )
        
        sbert_btn.click(
            lambda q: qa_system.answer_question_hybrid(q, "sbert"),
            inputs=sbert_input,
            outputs=[sbert_answer, sbert_details]
        )
        
        refresh_tfidf_btn.click(
            qa_system.get_random_questions,
            outputs=sample_questions_tfidf
        )
        
        refresh_sbert_btn.click(
            qa_system.get_random_questions,
            outputs=sample_questions_sbert
        )
        
        compare_btn.click(
            qa_system.compare_all_models,
            inputs=compare_input,
            outputs=[comparison_summary, tfidf_comp, sbert_comp, saiga_comp, comparison_details]
        )
    
    return demo, qa_system