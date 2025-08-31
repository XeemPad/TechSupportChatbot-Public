import json
import random
import re
from pathlib import Path

from typing import Optional

import pandas as pd


class TestSetManager:
    """Управление тестовыми наборами данных"""
    
    def __init__(self, work_dir=Path("./"), test_sets_dir: Optional[str]=None):
        self.work_dir = Path(work_dir)
        self.base_dir = self.work_dir / "data" if test_sets_dir is None else Path(test_sets_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.test_sets = self._load_all_sets()
    
    def _load_all_sets(self):
        """Загрузка всех доступных тестовых наборов"""
        test_sets = {}
        
        for file_path in self.base_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    test_sets[file_path.stem] = json.load(f)
                    print(f"Загружен тестовый набор: {file_path.stem} ({len(test_sets[file_path.stem])} примеров)")
            except Exception as e:
                print(f"Ошибка при загрузке {file_path}: {e}")
                
        return test_sets
    
    def import_test_set_from_csv(self, file_path: Optional[str|Path]=None):
        """Импорт тестового набора из файла test.csv в каталоге data проекта"""
        
        # Определяем путь к файлу
        csv_path = self.base_dir / "test.csv" if file_path is None else Path(file_path)
        
        print(f"Импортируем тестовую выборку из: {csv_path}")
        
        # Загружаем данные из CSV
        try:
            df = pd.read_csv(csv_path)
            print(f"Загружено {len(df)} строк из {csv_path}")
            
            # Получаем имена колонок и проверяем их наличие
            columns = df.columns.tolist()
            print(f"Колонки в файле: {columns}")
            
            # Обрабатываем разные форматы файла
            q_column = next((col for col in columns if 'quest' in col.lower()), columns[0])
            a_column = next((col for col in columns if 'ans' in col.lower()), columns[1])
            
            # Преобразуем в формат тестового набора
            test_set = []
            for _, row in df.iterrows():
                test_set.append({
                    "query": row[q_column],
                    "expected_answer": row[a_column],
                    "relevant_docs": [], 
                    "category": "project_test_data"
                })
                
            # Сохраняем тестовый набор
            return self.add_test_set("project_test_data", test_set)
            
        except Exception as e:
            print(f"Ошибка при импорте CSV: {e}")
            return None
    
    def get_test_set(self, name):
        """Получение тестового набора по имени"""
        test_set = self.test_sets.get(name)
        if not test_set:
            print(f"Тестовый набор '{name}' не найден.")
            return None
        return test_set
    
    def add_test_set(self, name, data):
        """Добавление нового тестового набора"""
        self.test_sets[name] = data
        
        # Сохранение в файл
        file_path = self.base_dir / f"{name}.json"
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Сохранен тестовый набор: {name} ({len(data)} примеров)")
        return file_path
    
    def list_test_sets(self):
        """Список доступных тестовых наборов"""
        return list(self.test_sets.keys())
    
    def create_stratified_sample(self, qa_df, kb_df=None, size=100):
        """Создание стратифицированной выборки из имеющихся данных"""
        # Создаем тестовый набор разных типов
        standard = []
        misspelled = []
        
        # Выбираем случайные вопросы из базы
        questions = qa_df['question'].tolist()
        answers = qa_df['answer'].tolist()
        
        # Лимит на выборку
        sample_size = min(size, len(questions))
        selected_indices = random.sample(range(len(questions)), sample_size)
        
        for idx in selected_indices:
            q = questions[idx]
            a = answers[idx]
            
            # Добавляем стандартный вопрос
            standard.append({
                "query": q,
                "expected_answer": a,
                "relevant_docs": [],
                "category": "standard"
            })
            
            # Добавляем версию с опечатками
            misspelled.append({
                "query": self._add_typos(q),
                "expected_answer": a,
                "relevant_docs": [],
                "category": "misspelled"
            })
        
        # Создаем и сохраняем наборы
        self.add_test_set("standard", standard[:min(25, len(standard))])
        self.add_test_set("misspelled", misspelled[:min(25, len(misspelled))])
        
        # Создаем полный набор для общей оценки
        full_set = []
        full_set.extend(standard[:min(15, len(standard))])
        full_set.extend(misspelled[:min(10, len(misspelled))])
        random.shuffle(full_set)  # Перемешиваем для более объективной оценки
        
        self.add_test_set("full", full_set)
        
        return self.test_sets
    
    def _add_typos(self, text):
        """Добавление случайных опечаток в текст"""
        chars = list(text)
        num_typos = max(1, len(text) // 10)  # ~10% опечаток
        
        for _ in range(num_typos):
            typo_idx = random.randint(0, len(chars) - 1)
            
            # Если символ не пробел
            if chars[typo_idx] != ' ':
                typo_type = random.choice(["swap", "miss", "double", "replace"])
                
                if typo_type == "swap" and typo_idx < len(chars) - 1:
                    chars[typo_idx], chars[typo_idx+1] = chars[typo_idx+1], chars[typo_idx]
                elif typo_type == "miss":
                    chars[typo_idx] = ""
                elif typo_type == "double" and chars[typo_idx] != ' ':
                    chars[typo_idx] = chars[typo_idx] * 2
                elif typo_type == "replace":
                    replacements = {
                        'о': 'а', 'а': 'о', 'е': 'и', 'и': 'е',
                        'т': 'д', 'д': 'т', 'з': 'с', 'с': 'з',
                        '?': '7'
                    }
                    if chars[typo_idx].lower() in replacements:
                        chars[typo_idx] = replacements[chars[typo_idx].lower()]
                
        return "".join(chars)
    
    def create_manual_test_set(self, name, queries, expected_answers):
        """Создание тестового набора вручную"""
        if len(queries) != len(expected_answers):
            raise ValueError("Количество запросов и ожидаемых ответов должно совпадать")
        
        test_data = []
        for query, answer in zip(queries, expected_answers):
            test_data.append({
                "query": query,
                "expected_answer": answer,
                "relevant_docs": [],
                "category": "manual"
            })
            
        return self.add_test_set(name, test_data)
