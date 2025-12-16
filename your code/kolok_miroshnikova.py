import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from typing import Tuple, List

class FuzzyStomachDiagnosisSystem:
    def __init__(self):
        # Инициализация нечеткой системы
        self.setup_fuzzy_system()
        
    def setup_fuzzy_system(self):
        """Настройка нечеткой системы вывода"""
        # 1. Определение пространств
        self.atypia = ctrl.Antecedent(np.arange(0, 11, 0.1), 'atypia')
        self.polymorphism = ctrl.Antecedent(np.arange(0, 11, 0.1), 'polymorphism')
        self.mitosis = ctrl.Antecedent(np.arange(0, 11, 0.1), 'mitosis')
        
        # Выходная переменная - риск от 0 до 100
        self.diagnosis_score = ctrl.Consequent(np.arange(0, 101, 1), 'diagnosis_score')
        
        # 2. Функции принадлежности для входных переменных
        # Атипия ядер
        self.atypia['low'] = fuzz.trimf(self.atypia.universe, [0, 0, 4])
        self.atypia['medium'] = fuzz.trimf(self.atypia.universe, [2, 5, 8])
        self.atypia['high'] = fuzz.trimf(self.atypia.universe, [6, 10, 10])
        
        # Полиморфизм
        self.polymorphism['low'] = fuzz.trimf(self.polymorphism.universe, [0, 0, 4])
        self.polymorphism['medium'] = fuzz.trimf(self.polymorphism.universe, [2, 5, 8])
        self.polymorphism['high'] = fuzz.trimf(self.polymorphism.universe, [6, 10, 10])
        
        # Активность митозов
        self.mitosis['low'] = fuzz.trimf(self.mitosis.universe, [0, 0, 4])
        self.mitosis['medium'] = fuzz.trimf(self.mitosis.universe, [2, 5, 8])
        self.mitosis['high'] = fuzz.trimf(self.mitosis.universe, [6, 10, 10])
        
        # 3. Функции принадлежности для выходной переменной
        self.diagnosis_score['normal'] = fuzz.trimf(self.diagnosis_score.universe, [0, 0, 40])
        self.diagnosis_score['dysplasia'] = fuzz.trimf(self.diagnosis_score.universe, [20, 50, 80])
        self.diagnosis_score['cancer'] = fuzz.trimf(self.diagnosis_score.universe, [60, 100, 100])
        
        # 4. База нечетких правил (расширенная)
        rules = [
            # Правила для НОРМАЛЬНОГО состояния
            ctrl.Rule(self.atypia['low'] & self.polymorphism['low'] & self.mitosis['low'], 
                     self.diagnosis_score['normal']),
            ctrl.Rule(self.atypia['low'] & self.polymorphism['low'] & self.mitosis['medium'], 
                     self.diagnosis_score['normal']),
            
            # Правила для ДИСПЛАЗИИ
            ctrl.Rule(self.atypia['medium'] & self.polymorphism['low'] & self.mitosis['low'], 
                     self.diagnosis_score['dysplasia']),
            ctrl.Rule(self.atypia['low'] & self.polymorphism['medium'] & self.mitosis['low'], 
                     self.diagnosis_score['dysplasia']),
            ctrl.Rule(self.atypia['low'] & self.polymorphism['low'] & self.mitosis['high'], 
                     self.diagnosis_score['dysplasia']),
            ctrl.Rule(self.atypia['medium'] & self.polymorphism['medium'] & self.mitosis['low'], 
                     self.diagnosis_score['dysplasia']),
            ctrl.Rule(self.atypia['medium'] & self.polymorphism['low'] & self.mitosis['medium'], 
                     self.diagnosis_score['dysplasia']),
            
            # Правила для РАКА
            ctrl.Rule(self.atypia['high'] & self.mitosis['high'], 
                     self.diagnosis_score['cancer']),
            ctrl.Rule(self.atypia['high'] & self.polymorphism['high'], 
                     self.diagnosis_score['cancer']),
            ctrl.Rule(self.polymorphism['high'] & self.mitosis['high'], 
                     self.diagnosis_score['cancer']),
            ctrl.Rule(self.atypia['medium'] & self.polymorphism['medium'] & self.mitosis['high'], 
                     self.diagnosis_score['cancer']),
            ctrl.Rule(self.atypia['high'] & self.polymorphism['medium'] & self.mitosis['medium'], 
                     self.diagnosis_score['cancer'])
        ]
        
        # 5. Создание системы управления
        self.control_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
    
    def predict(self, atypia_score: float, polymorphism_score: float, mitosis_score: float) -> float:
        """Предсказание диагностического балла"""
        self.simulation.input['atypia'] = atypia_score
        self.simulation.input['polymorphism'] = polymorphism_score
        self.simulation.input['mitosis'] = mitosis_score
        
        try:
            self.simulation.compute()
            return self.simulation.output['diagnosis_score']
        except:
            return 0.0
    
    def interpret_score(self, score: float) -> str:
        """Интерпретация числового балла в диагноз"""
        if score < 35:
            return "normal"
        elif score < 65:
            return "dysplasia"
        else:
            return "cancer"

def generate_test_data(n_samples: int = 200) -> pd.DataFrame:
    """Генерация тестовых данных с пограничными случаями"""
    np.random.seed(42)
    
    data = []
    
    # 1. НОРМАЛЬНЫЕ случаи (60 образцов)
    for _ in range(n_samples // 3 - 10):
        atypia = np.random.uniform(0, 3.5)
        polymorphism = np.random.uniform(0, 3.5)
        mitosis = np.random.uniform(0, 3.5)
        true_diagnosis = "normal"
        data.append([atypia, polymorphism, mitosis, true_diagnosis])
    
    # 2. ДИСПЛАЗИЯ (70 образцов с пограничными случаями)
    for _ in range(n_samples // 3 + 10):
        # Создаем пограничные случаи
        if np.random.random() > 0.7:
            # Пограничный случай: смешанные признаки
            atypia = np.random.uniform(3, 6)
            polymorphism = np.random.uniform(3, 6)
            mitosis = np.random.uniform(2, 5)
        else:
            # Типичная дисплазия
            atypia = np.random.uniform(4, 7)
            polymorphism = np.random.uniform(4, 7)
            mitosis = np.random.uniform(3, 6)
        true_diagnosis = "dysplasia"
        data.append([atypia, polymorphism, mitosis, true_diagnosis])
    
    # 3. РАК (70 образцов с пограничными случаями)
    for _ in range(n_samples // 3 + 10):
        if np.random.random() > 0.8:
            # Пограничный случай рак/дисплазия
            atypia = np.random.uniform(6, 8)
            polymorphism = np.random.uniform(6, 8)
            mitosis = np.random.uniform(5, 7)
        else:
            # Явный рак
            atypia = np.random.uniform(7, 10)
            polymorphism = np.random.uniform(7, 10)
            mitosis = np.random.uniform(6, 10)
        true_diagnosis = "cancer"
        data.append([atypia, polymorphism, mitosis, true_diagnosis])
    
    # Создаем DataFrame
    df = pd.DataFrame(data, columns=['atypia', 'polymorphism', 'mitosis', 'true_diagnosis'])
    
    # Перемешиваем данные
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def visualize_membership_functions(fuzzy_system: FuzzyStomachDiagnosisSystem):
    """Визуализация функций принадлежности"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Входные переменные
    variables = [
        (fuzzy_system.atypia, 'Атипия ядер', axes[0, 0]),
        (fuzzy_system.polymorphism, 'Полиморфизм клеток', axes[0, 1]),
        (fuzzy_system.mitosis, 'Активность митозов', axes[1, 0])
    ]
    
    for var, title, ax in variables:
        for term in ['low', 'medium', 'high']:
            ax.plot(var.universe, var[term].mf, label=term, linewidth=2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Балл (0-10)')
        ax.set_ylabel('Степень принадлежности')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Выходная переменная
    ax = axes[1, 1]
    for term in ['normal', 'dysplasia', 'cancer']:
        ax.plot(fuzzy_system.diagnosis_score.universe, 
                fuzzy_system.diagnosis_score[term].mf, 
                label=term, linewidth=2)
    ax.set_title('Диагностический балл (риск)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Балл риска (0-100)')
    ax.set_ylabel('Степень принадлежности')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fuzzy_membership_functions.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_evaluation(fuzzy_system: FuzzyStomachDiagnosisSystem, test_df: pd.DataFrame):
    """Запуск оценки модели на тестовых данных"""
    predictions = []
    scores = []
    
    print("Оценка нечеткой системы диагностики...")
    print("=" * 60)
    
    # Прогнозирование для каждого образца
    for idx, row in test_df.iterrows():
        score = fuzzy_system.predict(row['atypia'], row['polymorphism'], row['mitosis'])
        pred = fuzzy_system.interpret_score(score)
        
        predictions.append(pred)
        scores.append(score)
    
    # Добавляем предсказания в DataFrame
    test_df['predicted_score'] = scores
    test_df['predicted_diagnosis'] = predictions
    
    # Оценка точности
    accuracy = accuracy_score(test_df['true_diagnosis'], test_df['predicted_diagnosis'])
    
    print(f"Точность модели: {accuracy:.3f}")
    print("\nПодробный отчет по классификации:")
    print("-" * 60)
    print(classification_report(test_df['true_diagnosis'], test_df['predicted_diagnosis']))
    
    return test_df, accuracy

def visualize_results(test_df: pd.DataFrame):
    """Визуализация результатов работы системы"""
    
    # 1. Матрица ошибок
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Матрица путаницы
    cm = confusion_matrix(test_df['true_diagnosis'], test_df['predicted_diagnosis'],
                         labels=["normal", "dysplasia", "cancer"])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["normal", "dysplasia", "cancer"],
                yticklabels=["normal", "dysplasia", "cancer"],
                ax=axes[0, 0])
    axes[0, 0].set_title('Матрица ошибок', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Предсказанный диагноз')
    axes[0, 0].set_ylabel('Истинный диагноз')
    
    # 2. Распределение диагностических баллов по классам
    for i, diagnosis in enumerate(["normal", "dysplasia", "cancer"], 1):
        ax = axes[(i-1)//2, 1] if i <= 2 else axes[1, 0]
        subset = test_df[test_df['true_diagnosis'] == diagnosis]
        ax.hist(subset['predicted_score'], bins=20, alpha=0.7, 
                label=f'{diagnosis} (n={len(subset)})')
        ax.set_title(f'Распределение баллов: {diagnosis}', fontsize=12)
        ax.set_xlabel('Диагностический балл')
        ax.set_ylabel('Частота')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. 3D визуализация пространства признаков
    from mpl_toolkits.mplot3d import Axes3D
    ax = axes[1, 1]
    ax.remove()
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    
    colors = {'normal': 'green', 'dysplasia': 'orange', 'cancer': 'red'}
    
    for diagnosis, color in colors.items():
        subset = test_df[test_df['true_diagnosis'] == diagnosis]
        ax.scatter(subset['atypia'], subset['polymorphism'], subset['mitosis'],
                  c=color, label=diagnosis, alpha=0.6, s=50)
    
    ax.set_xlabel('Атипия ядер')
    ax.set_ylabel('Полиморфизм')
    ax.set_zlabel('Активность митозов')
    ax.set_title('3D пространство признаков', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('diagnosis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Анализ пограничных случаев
    print("\nАнализ пограничных случаев:")
    print("=" * 60)
    
    # Находим случаи, где балл близок к границам
    borderline_cases = test_df[
        ((test_df['predicted_score'] > 30) & (test_df['predicted_score'] < 40)) |  # норма/дисплазия
        ((test_df['predicted_score'] > 60) & (test_df['predicted_score'] < 70))   # дисплазия/рак
    ].copy()
    
    print(f"Найдено пограничных случаев: {len(borderline_cases)}")
    
    if len(borderline_cases) > 0:
        print("\nПримеры пограничных случаев:")
        for idx, row in borderline_cases.head(5).iterrows():
            print(f"Образец {idx}: A={row['atypia']:.1f}, P={row['polymorphism']:.1f}, "
                  f"M={row['mitosis']:.1f} | "
                  f"Истинный: {row['true_diagnosis']}, "
                  f"Предсказанный: {row['predicted_diagnosis']} "
                  f"(балл: {row['predicted_score']:.1f})")

def create_simple_threshold_model(test_df: pd.DataFrame) -> float:
    """Простая пороговая модель для сравнения"""
    # Средний балл по трем признакам, масштабированный до 0-100
    test_df['simple_score'] = (test_df[['atypia', 'polymorphism', 'mitosis']].mean(axis=1) / 10) * 100
    
    # Простая классификация
    def simple_classify(score):
        if score < 40:
            return "normal"
        elif score < 70:
            return "dysplasia"
        else:
            return "cancer"
    
    test_df['simple_prediction'] = test_df['simple_score'].apply(simple_classify)
    
    accuracy = accuracy_score(test_df['true_diagnosis'], test_df['simple_prediction'])
    
    return accuracy, test_df

def compare_models(fuzzy_accuracy: float, threshold_accuracy: float, test_df: pd.DataFrame):
    """Сравнение нечеткой модели с пороговой"""
    print("\nСравнение моделей:")
    print("=" * 60)
    print(f"Нечеткая модель Мамдани: {fuzzy_accuracy:.3f}")
    print(f"Простая пороговая модель: {threshold_accuracy:.3f}")
    print(f"Разница: {(fuzzy_accuracy - threshold_accuracy):.3f}")
    
    # Визуализация сравнения
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Сравнение точности
    models = ['Нечеткая модель', 'Пороговая модель']
    accuracies = [fuzzy_accuracy, threshold_accuracy]
    
    bars = axes[0].bar(models, accuracies, color=['steelblue', 'lightcoral'])
    axes[0].set_ylabel('Точность')
    axes[0].set_title('Сравнение точности моделей', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1)
    
    # Добавление значений на столбцы
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
    
    # Сравнение предсказаний для пограничных случаев
    borderline = test_df[
        (test_df['predicted_score'] > 30) & (test_df['predicted_score'] < 70)
    ]
    
    if len(borderline) > 0:
        correct_fuzzy = (borderline['true_diagnosis'] == borderline['predicted_diagnosis']).sum()
        correct_threshold = (borderline['true_diagnosis'] == borderline['simple_prediction']).sum()
        
        border_accuracies = [correct_fuzzy/len(borderline), correct_threshold/len(borderline)]
        
        bars = axes[1].bar(models, border_accuracies, color=['steelblue', 'lightcoral'])
        axes[1].set_ylabel('Точность')
        axes[1].set_title('Точность на пограничных случаях', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, 1)
        
        for bar, acc in zip(bars, border_accuracies):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Основная функция выполнения"""
    print("Программная система диагностики заболеваний желудка")
    print("Метод: Нечеткие вычисления (Fuzzy Logic)")
    print("=" * 60)
    
    # 1. Инициализация нечеткой системы
    fuzzy_system = FuzzyStomachDiagnosisSystem()
    
    # 2. Визуализация функций принадлежности
    print("\n1. Визуализация функций принадлежности...")
    visualize_membership_functions(fuzzy_system)
    
    # 3. Генерация тестовых данных
    print("\n2. Генерация тестовых данных с пограничными случаями...")
    test_data = generate_test_data(200)
    print(f"Сгенерировано {len(test_data)} образцов:")
    print(test_data['true_diagnosis'].value_counts())
    
    # 4. Оценка нечеткой модели
    print("\n3. Оценка нечеткой модели...")
    results_df, fuzzy_accuracy = run_evaluation(fuzzy_system, test_data)
    
    # 5. Визуализация результатов
    print("\n4. Визуализация результатов...")
    visualize_results(results_df)
    
    # 6. Сравнение с простой моделью
    print("\n5. Сравнение с пороговой моделью...")
    threshold_accuracy, results_df = create_simple_threshold_model(results_df)
    compare_models(fuzzy_accuracy, threshold_accuracy, results_df)
    
    # 7. Демонстрация работы на примерах
    print("\n6. Демонстрационные примеры:")
    print("=" * 60)
    
    examples = [
        (1.5, 2.0, 0.8, "Очевидная норма"),
        (4.5, 5.0, 3.5, "Пограничный случай: дисплазия"),
        (6.5, 7.0, 6.5, "Пограничный случай: дисплазия/рак"),
        (8.5, 9.0, 8.0, "Очевидный рак")
    ]
    
    for atypia, polym, mitos, desc in examples:
        score = fuzzy_system.predict(atypia, polym, mitos)
        diagnosis = fuzzy_system.interpret_score(score)
        print(f"{desc}: A={atypia}, P={polym}, M={mitos}")
        print(f"  → Балл: {score:.1f}, Диагноз: {diagnosis}")
        print("-" * 40)

if __name__ == "__main__":
    main()