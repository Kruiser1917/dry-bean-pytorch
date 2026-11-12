# PyTorch: тензоры, Autograd и `nn.Module` — классификация сортов фасоли (`Dry Bean Dataset`)

Практическая работа по **Модулю 06 (Deep Learning), Тема 02**: полный воспроизводимый пайплайн обучения многослойного перцептрона на `PyTorch` (`mps` / `cuda` / `cpu`) на датасете **UCI Dry Bean Dataset** (Koklu & Ozkan, 2020) — от аудита дубликатов и масштабов признаков до кастомного `Dataset`, архитектуры `nn.Module`, сохранения лучшего чекпоинта, бит-в-бит верификации `load_state_dict` и честной однократной оценки на `test`.

## Схема пайплайна и архитектура `DryBeanMLP`

```text
data/dry_bean.csv (13 611 x 17)
  ──► Дедупликация ДО сплита (-68 строк HOROZ) ──► 13 543 уникальных объекта (7 сортов)
  ──► Stratified Split 70 / 15 / 15            ──► Train: 9 480 | Val: 2 031 | Test: 2 032
  ──► StandardScaler (fit строго на Train)     ──► Tensor X (float32), y (int64)
  ──► DryBeanDataset + DataLoader              ──► Train (bs=64, shuffle=True), Val/Test (bs=256, shuffle=False)
  ──► DryBeanMLP (16 -> 64 -> 32 -> 7)         ──► Linear + BatchNorm1d + ReLU + Dropout(0.10) -> сырые logits
  ──► AdamW + CrossEntropyLoss (60 эпох)       ──► Лучший чекпоинт на эпохе 37 (checkpoints/best_dry_bean_mlp.pt)
  ──► Чистый экземпляр + load_state_dict       ──► max |logits_before - logits_after| = 0.0 -> оценка на Test
```

### Послойное число обучаемых параметров (`3 591`)

| Слой | Компоненты | Форма параметров | Обучаемых параметров |
| :--- | :--- | :---: | :---: |
| **`fc1 + bn1`** | `Linear(16, 64)` + `BatchNorm1d(64)` | `(64, 16) + (64,) + (64,) + (64,)` | `1 088 + 128 = 1 216` |
| **`fc2 + bn2`** | `Linear(64, 32)` + `BatchNorm1d(32)` | `(32, 64) + (32,) + (32,) + (32,)` | `2 080 + 64 = 2 144` |
| **`fc_out`** | `Linear(32, 7)` (сырые `logits`, без `Softmax`) | `(7, 32) + (7,)` | `224 + 7 = 231` |
| **Итого** | `requires_grad=True` (+ `194` элемента буферов `BN`) | — | **`3 591`** |

## Содержание исследования (`dry_bean_pytorch_pipeline.ipynb`)

1. **Блок 1. Среда, `seed = 42`, `device` (`mps`) и базовые механики PyTorch:** демонстрация `RuntimeError` при смешивании тензоров `cpu` и `mps` (решение через `.to(device)`) и демонстрация удвоения градиентов в `Autograd` при пропуске `optimizer.zero_grad()`.
2. **Блок 2. EDA, очистка данных, `Tensor`, `DryBeanDataset` и `DataLoader`:** удаление `68` полных дубликатов до сплита (`13 611 -> 13 543`), анализ дисбаланса классов (`DERMASON / BOMBAY = 6.79x`) и разброса масштабов признаков (`max(std) / min(std) = 5.01e7`), стратифицированный сплит `70/15/15`, `StandardScaler` строго по `train`.
3. **Блок 3. Архитектура `DryBeanMLP(nn.Module)`:** `Linear + BatchNorm1d + ReLU + Dropout(0.10)`, выход в виде сырых `logits`, проверка различия прямого прохода между `model.train()` (`max |diff| = 1.3366`) и `model.eval()` (`max |diff| = 0.0`).
4. **Блок 4. Обучение и чекпоинтинг:** `CrossEntropyLoss` + `AdamW(lr=1e-3, weight_decay=1e-4)` + `ReduceLROnPlateau` (`60` эпох), модульные `train_one_epoch` и `evaluate`, сохранение лучшего состояния на **эпохе 37** в `checkpoints/best_dry_bean_mlp.pt` (`57.0 КБ`).
5. **Блок 5. Верификация `load_state_dict` и финальная оценка на `Test`:** загрузка через `torch.load(..., weights_only=True)` в чистый экземпляр модели, 100%-ное совпадение логитов и предсказаний на `Validation` (`2031/2031`), матрица ошибок и разбор вероятностей `Softmax` на `Test`.

## Итоговые метрики модели (эпоха чекпоинта: `37`)

| Выборка | Объектов | `Cross-Entropy Loss` | `Accuracy` | `Macro F1` | `Weighted F1` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation (15%)** | `2 031` | `0.2034` | `93.35%` | `0.9415` | `0.9336` |
| **Test (15%)** | `2 032` | `0.2034` | `92.08%` | `0.9347` | `0.9209` |

### Поклассовые метрики на отложенной выборке `Test` (`N = 2 032`)

| Сорт фасоли (`Class`) | `Precision` | `Recall` | `F1-score` | `Support` |
| :--- | :---: | :---: | :---: | :---: |
| **BARBUNYA** | `0.9250` | `0.9343` | `0.9296` | `198` |
| **BOMBAY** | `1.0000` | `1.0000` | `1.0000` | `78` |
| **CALI** | `0.9426` | `0.9388` | `0.9407` | `245` |
| **DERMASON** | `0.9237` | `0.8872` | `0.9051` | `532` |
| **HOROZ** | `0.9464` | `0.9498` | `0.9481` | `279` |
| **SEKER** | `0.9508` | `0.9539` | `0.9524` | `304` |
| **SIRA** | `0.8478` | `0.8864` | `0.8667` | `396` |

## Структура репозитория

```text
dry-bean-pytorch/
├── dry_bean_pytorch_pipeline.ipynb   # Выполненный ноутбук со всеми расчётами, графиками и выводами
├── checkpoints/
│   └── best_dry_bean_mlp.pt          # Лучший сохранённый чекпоинт модели (эпоха 37, weights_only-совместимый)
├── data/
│   └── dry_bean.csv                  # Официальный UCI Dry Bean Dataset (13 611 строк, 16 признаков + Class)
├── requirements.txt                  # Зависимости проекта
├── .gitignore
└── README.md
```

## Как запустить

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook dry_bean_pytorch_pipeline.ipynb
```
