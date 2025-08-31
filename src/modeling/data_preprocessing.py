import pandas as pd
import numpy as np
from pathlib import Path
import re
import json
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm



class DataPreprocessor:
    """Предобработчик данных для QA системы"""
    
    def __init__(self, work_dir=Path("./"), output_dir=None, data_dir=None):
        self.work_dir = work_dir
        self.data_dir = work_dir / 'data' if data_dir is None else data_dir
        self.output_dir = self.work_dir / 'processed_data' if output_dir is None else Path(output_dir)

        self.output_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        self.processed_data = {
            'direct_qa_pairs': [],   # Прямые пары вопрос-ответ
            'knowledge_chunks': [],  # Чанки знаний
            'documents_info': []     # Информация о документах
        }
    
    def process_all(self, qa_file='QA_pairs.xlsx', kb_file='knowledge_base.xlsx', kb_delimiter: str='<nt>',
                    analyze: bool=True):
        """Выполнение всей предобработки"""
        print("Начинаем полную предобработку данных...")
        
        self._load_raw_data(qa_file, kb_file)
        self._process_qa_pairs()
        self._process_knowledge_base(kb_delimiter=kb_delimiter)
        self._remove_duplicates()
        if analyze: 
            self._analyze_processed_data()
        
        # Сохраняем результаты
        output_dir = self.save_to_csv()
        self.save_to_json()
        
        # Создаем единый датасет
        unified_df = self._create_unified_dataset() 
        unified_df.to_csv(output_dir / 'unified_dataset.csv', index=False, encoding='utf-8')
        
        print(f"\nПредобработка завершена!")
        print(f"Итоговый единый датасет: {len(unified_df)} записей")
        
        return unified_df
    
    def _load_raw_data(self, qa_file='QA_pairs.xlsx', kb_file='knowledge_base.xlsx'):
        """Загрузка сырых данных"""
        print("\tЗагружаем сырые данные...")
        
        # QA пары
        self.qa_df = pd.read_excel(self.data_dir / qa_file)
        
        # База знаний
        kb_dict = pd.read_excel(self.data_dir / kb_file, 
                               sheet_name=['Knowledge_base', 'Sources'])
        self.kb_df = kb_dict['Knowledge_base']
        self.kb_info_df = kb_dict['Sources']
        
        print(f"QA пар: {len(self.qa_df)}")
        print(f"KB чанков: {len(self.kb_df)}")
        print(f"Источников: {len(self.kb_info_df)}")
    
    def _process_qa_pairs(self):
        """Обработка готовых QA пар"""
        print("Обрабатываем QA пары...")
        
        for idx, row in tqdm(self.qa_df.iterrows(), total=len(self.qa_df)):
            question = self.clean_text(str(row['question']))
            answer = self.clean_text(str(row['answer']))
            
            if len(question) > 5 and len(answer) > 5:
                self.processed_data['direct_qa_pairs'].append({
                    'id': f"qa_{idx}",
                    'question': question,
                    'answer': answer,
                    'source': 'direct_qa',
                    'length_q': len(question),
                    'length_a': len(answer)
                })
    
    def _process_knowledge_base(self, kb_delimiter: str='<nt>'):
        """Обработка базы знаний"""
        print("Обрабатываем базу знаний...")
        
        # Группируем по document_id
        for doc_id in tqdm(self.kb_info_df['document_id'].unique()):
            doc_info = self.kb_info_df[self.kb_info_df['document_id'] == doc_id].iloc[0]
            doc_chunks = self.kb_df[self.kb_df['document_id'] == doc_id]
            
            # Информация о документе
            doc_title = str(doc_info['title']) if 'title' in doc_info else f"Document_{doc_id}"
            
            self.processed_data['documents_info'].append({
                'document_id': doc_id,
                'title': doc_title,
                'chunks_count': len(doc_chunks),
                'source_type': self.classify_document_type(doc_title)
            })
            
            # Обрабатываем чанки документа
            for chunk_idx, chunk_row in doc_chunks.iterrows():
                chunk_text = str(chunk_row['chunk'])
                
                # Проверяем, есть ли в чанке пары вопрос-ответ
                if kb_delimiter in chunk_text:
                    # Извлекаем QA пары
                    qa_pairs = self.extract_qa_from_chunk(chunk_text)
                    for q, a in qa_pairs:
                        self.processed_data['direct_qa_pairs'].append({
                            'id': f"kb_qa_{doc_id}_{chunk_idx}",
                            'question': q,
                            'answer': a,
                            'source': f'knowledge_base_doc_{doc_id}',
                            'document_title': doc_title,
                            'length_q': len(q),
                            'length_a': len(a)
                        })
                else:
                    # Обычный информационный чанк
                    cleaned_chunk = self.clean_text(chunk_text)
                    if len(cleaned_chunk) > 20:  # Минимальная длина чанка
                        self.processed_data['knowledge_chunks'].append({
                            'id': f"chunk_{doc_id}_{chunk_idx}",
                            'text': cleaned_chunk,
                            'document_id': doc_id,
                            'document_title': doc_title,
                            'length': len(cleaned_chunk),
                            'source_type': self.classify_document_type(doc_title)
                        })
    
    def _remove_duplicates(self):
        """Удаление дубликатов"""
        print("Удаляем дубликаты...")
        
        # Дубликаты в QA парах
        qa_texts = set()
        unique_qa_pairs = []
        
        for qa_pair in self.processed_data['direct_qa_pairs']:
            text_key = (qa_pair['question'].lower(), qa_pair['answer'].lower())
            if text_key not in qa_texts:
                qa_texts.add(text_key)
                unique_qa_pairs.append(qa_pair)
        
        removed_qa = len(self.processed_data['direct_qa_pairs']) - len(unique_qa_pairs)
        self.processed_data['direct_qa_pairs'] = unique_qa_pairs
        
        # Дубликаты в чанках знаний
        chunk_texts = set()
        unique_chunks = []
        
        for chunk in self.processed_data['knowledge_chunks']:
            text_key = chunk['text'].lower()
            if text_key not in chunk_texts and len(text_key) > 20:
                chunk_texts.add(text_key)
                unique_chunks.append(chunk)
        
        removed_chunks = len(self.processed_data['knowledge_chunks']) - len(unique_chunks)
        self.processed_data['knowledge_chunks'] = unique_chunks
        
        print(f"Удалено дубликатов QA: {removed_qa}")
        print(f"Удалено дубликатов чанков: {removed_chunks}")

    def _analyze_processed_data(self):
        """Анализ обработанных данных"""
        print("\n=== АНАЛИЗ ОБРАБОТАННЫХ ДАННЫХ ===")
        
        qa_pairs = self.processed_data['direct_qa_pairs']
        chunks = self.processed_data['knowledge_chunks']
        docs = self.processed_data['documents_info']
        
        print(f"Всего QA пар: {len(qa_pairs)}")
        print(f"Всего информационных чанков: {len(chunks)}")
        print(f"Всего документов: {len(docs)}")
        
        # Анализ источников QA пар
        if qa_pairs:
            sources = {}
            for qa in qa_pairs:
                source = qa['source']
                sources[source] = sources.get(source, 0) + 1
            
            print("\nРаспределение QA пар по источникам:")
            for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                print(f"  {source}: {count}")
        
        # Анализ типов документов
        if chunks:
            doc_types = {}
            for chunk in chunks:
                doc_type = chunk['source_type']
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            print("\nРаспределение чанков по типам документов:")
            for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {doc_type}: {count}")
        
        # Статистика длин
        if qa_pairs:
            q_lengths = [qa['length_q'] for qa in qa_pairs]
            a_lengths = [qa['length_a'] for qa in qa_pairs]
            
            print(f"\nСтатистика длин вопросов:")
            print(f"  Средняя: {np.mean(q_lengths):.1f}")
            print(f"  Медиана: {np.median(q_lengths):.1f}")
            print(f"  Мин-Макс: {min(q_lengths)}-{max(q_lengths)}")
            
            print(f"\nСтатистика длин ответов:")
            print(f"  Средняя: {np.mean(a_lengths):.1f}")
            print(f"  Медиана: {np.median(a_lengths):.1f}")
            print(f"  Мин-Макс: {min(a_lengths)}-{max(a_lengths)}")
        
        if chunks:
            chunk_lengths = [chunk['length'] for chunk in chunks]
            print(f"\nСтатистика длин чанков:")
            print(f"  Средняя: {np.mean(chunk_lengths):.1f}")
            print(f"  Медиана: {np.median(chunk_lengths):.1f}")
            print(f"  Мин-Макс: {min(chunk_lengths)}-{max(chunk_lengths)}")
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Улучшенная очистка текстовых чанков"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Сначала нормализуем пробелы, но сохраняем структуру
        text = re.sub(r'\t', ' ', text)  # Табы в пробелы
        
        # Исправляем пробелы вокруг кавычек и скобок
        text = re.sub(r'\s+(["\'()])', r' \1', text)  # пробел перед кавычкой
        text = re.sub(r'(["\'()])\s+', r'\1 ', text)   # пробел после кавычки
        
        # Убираем множественные пробелы
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Убираем пробелы в начале и конце
        text = text.strip()
        
        return text
    
    def extract_qa_from_chunk(self, chunk: str, kb_delimiter: str='<nt>') -> List[Tuple[str, str]]:
        """Извлечение пар вопрос-ответ из чанка с разделителем <nt>"""
        qa_pairs = []
        
        if kb_delimiter in chunk:
            parts = chunk.split(kb_delimiter)
            if len(parts) == 2:
                question = self.clean_text(parts[0])
                answer = self.clean_text(parts[1])
                
                if len(question) >= 2 and len(answer) > 10:  # Минимальная длина (Может быть "Hi")
                    qa_pairs.append((question, answer))
        
        return qa_pairs
    
    def classify_document_type(self, title: str) -> str:
        """Классификация типа документа по названию"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['договор', 'соглашение', 'оферта']):
            return 'contract'
        elif any(word in title_lower for word in ['инструкция', 'руководство', 'гайд', 'рекомендации']):
            return 'instruction'
        elif any(word in title_lower for word in ['faq', 'вопрос', 'ответ']):
            return 'faq'
        elif any(word in title_lower for word in ['политика', 'правила', 'условия']):
            return 'policy'
        else:
            return 'general'
    
    def save_to_csv(self, output_dir=None):
        """Сохранение в CSV файлы"""
        if output_dir:
            self.output_dir = output_dir
        
        print(f"Сохраняем данные в {self.output_dir}...")
        
        # QA пары
        qa_df = pd.DataFrame(self.processed_data['direct_qa_pairs'])
        qa_df.to_csv(self.output_dir / 'qa_pairs_processed.csv', index=False, encoding='utf-8')
        
        # Информационные чанки
        chunks_df = pd.DataFrame(self.processed_data['knowledge_chunks'])
        chunks_df.to_csv(self.output_dir / 'knowledge_chunks_processed.csv', index=False, encoding='utf-8')
        
        # Информация о документах
        docs_df = pd.DataFrame(self.processed_data['documents_info'])
        docs_df.to_csv(self.output_dir / 'documents_info.csv', index=False, encoding='utf-8')
        
        print(f"Сохранено файлов: 3")
        return self.output_dir
    
    def save_to_json(self, filename: str = "all_data.json"):
        """Сохранение в JSON"""
        save_path = self.output_dir / filename
        save_path.parent.mkdir(exist_ok=True)
        
        # Конвертируем int64 в обычные int для JSON сериализации
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Применяем конвертацию ко всем данным
        json_data = {}
        for key, items in self.processed_data.items():
            json_data[key] = []
            for item in items:
                converted_item = {}
                for k, v in item.items():
                    converted_item[k] = convert_types(v)
                json_data[key].append(converted_item)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"Данные сохранены в {save_path}")
    
    def _create_unified_dataset(self) -> pd.DataFrame:
        """Создание единого датасета для обучения моделей"""
        unified_data = []
        
        # Добавляем QA пары
        for qa in self.processed_data['direct_qa_pairs']:
            unified_data.append({
                'id': qa['id'],
                'type': 'qa_pair',
                'question': qa['question'],
                'answer': qa['answer'],
                'text': f"{qa['question']} {qa['answer']}",  # Для поиска
                'source': qa['source'],
                'document_title': qa.get('document_title', ''),
                'source_type': 'qa'
            })
        
        # Добавляем информационные чанки
        for chunk in self.processed_data['knowledge_chunks']:
            unified_data.append({
                'id': chunk['id'],
                'type': 'knowledge_chunk',
                'question': '',
                'answer': chunk['text'],
                'text': chunk['text'],
                'source': f"doc_{chunk['document_id']}",
                'document_title': chunk['document_title'],
                'source_type': chunk['source_type']
            })
        
        return pd.DataFrame(unified_data)


def main():
    """Основная функция для запуска предобработки"""
    
    preprocessor = DataPreprocessor()
    
    # Выполняем всю предобработку
    unified_df = preprocessor.process_all()
    
    # Показываем примеры
    print("\nПРИМЕРЫ ОБРАБОТАННЫХ ДАННЫХ")
    
    # QA пары
    qa_examples = unified_df[unified_df['type'] == 'qa_pair'].head(3)
    print("\nПримеры QA пар:")
    for _, row in qa_examples.iterrows():
        print(f"Q: {row['question'][:100]}...")
        print(f"A: {row['answer'][:100]}...")
        print(f"Source: {row['source']}")
        print("-" * 50)
    
    # Информационные чанки
    chunk_examples = unified_df[unified_df['type'] == 'knowledge_chunk'].head(2)
    print("\nПримеры информационных чанков:")
    for _, row in chunk_examples.iterrows():
        print(f"Text: {row['text'][:150]}...")
        print(f"Document: {row['document_title']}")
        print(f"Type: {row['source_type']}")
        print("-" * 50)


if __name__ == "__main__":
    main()