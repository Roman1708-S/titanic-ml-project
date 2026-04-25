import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# === Читаємо дані ===
train = pd.read_csv(r'c:\Users\HP\mebli-clients\Nowy folder\titanic_project\train.csv')
test = pd.read_csv(r'c:\Users\HP\mebli-clients\Nowy folder\titanic_project\test.csv')

# Зберігаємо PassengerId для фінального файлу
test_ids = test['PassengerId']

# === Очистка TRAIN ===
train = train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Embarked'] = train['Embarked'].fillna('S')
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)

# === Очистка TEST (те саме!) ===
test = test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
test['Embarked'] = test['Embarked'].fillna('S')
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)

# === Навчаємо модель на ВСІХ тренувальних даних ===
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === Передбачаємо для тестових даних ===
predictions = model.predict(test)

# === Створюємо файл для Kaggle ===
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': predictions
})

submission.to_csv(r'c:\Users\HP\mebli-clients\Nowy folder\titanic_project\submission.csv', index=False)

print('=== ГОТОВО! ===')
print(f'Передбачено для {len(predictions)} пасажирів')
print()
print('Перші 10 передбачень:')
print(submission.head(10))
print()
print('Файл submission.csv створено!')
print('Тепер можеш завантажити його на Kaggle!')