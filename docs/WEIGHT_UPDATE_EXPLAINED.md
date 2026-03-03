# Flujo de Actualización de Pesos en PyTorch

## Resumen Ejecutivo

**Los pesos se actualizan en `optimizer.step()`**, NO en `scheduler.step()`.

- **`optimizer.step()`**: Actualiza los pesos del modelo
- **`scheduler.step()`**: Actualiza el learning rate del optimizador

---

## Flujo Completo de Entrenamiento

### 1️⃣ **Inicialización (una sola vez)**

```python
# En scripts/train.py, función setup_model_and_optimizer()

model = UNet(...)  # Modelo con pesos aleatorios
optimizer = AdamW(model.parameters(), lr=0.001)  # Optimizador
scheduler = get_scheduler(optimizer, config)  # Scheduler
```

**Estado inicial:**
- Modelo: pesos aleatorios (W₀)
- Optimizer: lr = 0.001
- Scheduler: configurado pero no ha hecho nada

---

### 2️⃣ **Loop de Entrenamiento (cada época)**

```python
# En scripts/train.py, función train()

for epoch in range(epochs):
    # 2.1: Entrenar una época
    train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
    
    # 2.2: Validar
    val_metrics = validate(model, val_loader, loss_fn, device)
    
    # 2.3: Actualizar learning rate
    scheduler.step()  # ← AQUÍ se actualiza el LR
    
    # 2.4: Guardar modelo si mejoró
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'best_model.pth')
```

---

### 3️⃣ **Dentro de `train_epoch()` (cada batch)**

Aquí es donde **realmente se actualizan los pesos**:

```python
# En src/training/train.py, función train_epoch()

for batch_idx, batch in enumerate(train_loader):
    input_img = batch['input'].to(device)      # [B, 3, 256, 256]
    target_img = batch['gt'].to(device)        # [B, 3, 256, 256]
    
    # ═══════════════════════════════════════════════════════════
    # PASO 1: LIMPIAR GRADIENTES
    # ═══════════════════════════════════════════════════════════
    optimizer.zero_grad()
    # Pone todos los gradientes a 0
    # Sin esto, los gradientes se acumularían entre batches
    
    # ═══════════════════════════════════════════════════════════
    # PASO 2: FORWARD PASS (Predicción)
    # ═══════════════════════════════════════════════════════════
    output = model(input_img)  # [B, 3, 256, 256]
    # El modelo hace la predicción
    # PyTorch construye el "computation graph" automáticamente
    
    # ═══════════════════════════════════════════════════════════
    # PASO 3: CALCULAR PÉRDIDA
    # ═══════════════════════════════════════════════════════════
    loss = loss_fn(output, target_img)  # Escalar
    # Compara predicción vs ground truth
    # Ejemplo: loss = 0.234
    
    # ═══════════════════════════════════════════════════════════
    # PASO 4: BACKWARD PASS (Calcular gradientes)
    # ═══════════════════════════════════════════════════════════
    loss.backward()
    # Calcula ∂loss/∂W para TODOS los pesos del modelo
    # Usa backpropagation y el computation graph
    # Ahora cada peso tiene su gradiente:
    #   model.conv1.weight.grad = [-0.001, 0.002, ...]
    #   model.conv2.weight.grad = [0.003, -0.001, ...]
    #   etc.
    
    # ═══════════════════════════════════════════════════════════
    # PASO 5: ACTUALIZAR PESOS ← ¡AQUÍ OCURRE LA MAGIA!
    # ═══════════════════════════════════════════════════════════
    optimizer.step()
    # Actualiza TODOS los pesos del modelo usando los gradientes
    # Para AdamW, la fórmula es:
    #   W_new = W_old - lr * gradient (simplificado)
    # En realidad AdamW usa momentum, adaptive learning rates, etc.
```

---

## Detalle de `optimizer.step()`

### ¿Qué hace exactamente?

```python
optimizer.step()
```

**Internamente (para AdamW):**

```python
for param in model.parameters():
    # 1. Obtener gradiente
    grad = param.grad
    
    # 2. Aplicar weight decay (regularización L2)
    grad = grad + weight_decay * param.data
    
    # 3. Actualizar momentum (primer momento)
    m = beta1 * m + (1 - beta1) * grad
    
    # 4. Actualizar velocidad (segundo momento)
    v = beta2 * v + (1 - beta2) * grad^2
    
    # 5. Corrección de bias
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    
    # 6. ACTUALIZAR PESO
    param.data = param.data - lr * m_hat / (sqrt(v_hat) + epsilon)
    #            └─ peso viejo ┘   └──────── actualización ────────┘
```

**Resultado:**
- Todos los pesos del modelo cambian ligeramente
- El modelo se vuelve "un poquito mejor" en esta tarea

---

## Detalle de `scheduler.step()`

### ¿Qué hace exactamente?

```python
scheduler.step()
```

**NO actualiza los pesos del modelo.**
**Solo actualiza el learning rate del optimizador.**

### Ejemplo con Cosine Annealing:

```python
# Época 0
lr = 0.001  # LR inicial
scheduler.step()  # lr = 0.000999 (baja un poquito)

# Época 1
scheduler.step()  # lr = 0.000995

# Época 2
scheduler.step()  # lr = 0.000988

# ...

# Época 499
scheduler.step()  # lr = 0.000001 (muy bajo al final)
```

### Ejemplo con Constant LR:

```python
# Época 0
lr = 0.001
scheduler.step()  # lr = 0.001 (no cambia)

# Época 1
scheduler.step()  # lr = 0.001 (no cambia)

# Época 2
scheduler.step()  # lr = 0.001 (no cambia)
```

---

## Visualización del Flujo Completo

```
┌─────────────────────────────────────────────────────────────┐
│                    ÉPOCA 0 (lr = 0.001)                     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │         BATCH 0 (8 imágenes)          │
        ├───────────────────────────────────────┤
        │ 1. optimizer.zero_grad()              │
        │ 2. output = model(input)              │
        │ 3. loss = loss_fn(output, target)     │
        │ 4. loss.backward()                    │
        │ 5. optimizer.step()  ← PESOS CAMBIAN  │
        └───────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │         BATCH 1 (8 imágenes)          │
        ├───────────────────────────────────────┤
        │ 1. optimizer.zero_grad()              │
        │ 2. output = model(input)              │
        │ 3. loss = loss_fn(output, target)     │
        │ 4. loss.backward()                    │
        │ 5. optimizer.step()  ← PESOS CAMBIAN  │
        └───────────────────────────────────────┘
                            │
                          [...]
                            │
        ┌───────────────────┴───────────────────┐
        │              VALIDACIÓN               │
        ├───────────────────────────────────────┤
        │ - Sin optimizer.step()                │
        │ - Solo evaluar                        │
        └───────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │        scheduler.step()               │
        ├───────────────────────────────────────┤
        │ lr = 0.001 → 0.000999                 │
        │ (LR CAMBIA, PESOS NO)                 │
        └───────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    ÉPOCA 1 (lr = 0.000999)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                          [...]
```

---

## Frecuencia de Actualización

| Operación | Frecuencia | Qué actualiza |
|-----------|-----------|---------------|
| `optimizer.zero_grad()` | Cada batch | Gradientes → 0 |
| `loss.backward()` | Cada batch | Calcula gradientes |
| **`optimizer.step()`** | **Cada batch** | **PESOS DEL MODELO** |
| `scheduler.step()` | Cada época | Learning rate |

---

## Ejemplo Numérico Real

### Estado Inicial:
```python
# Un peso del modelo
W = 0.523

# Configuración
lr = 0.001
```

### Batch 1:
```python
# 1. Forward pass
output = model(input)  # W se usa para calcular output

# 2. Loss
loss = 0.234

# 3. Backward
loss.backward()
# Gradiente: ∂loss/∂W = -0.15

# 4. Optimizer step
W_new = W - lr * gradient
W_new = 0.523 - 0.001 * (-0.15)
W_new = 0.523 + 0.00015
W_new = 0.52315  ← PESO ACTUALIZADO
```

### Batch 2:
```python
# Ahora W = 0.52315 (nuevo valor)

# 1. Forward pass
output = model(input)  # Usa W = 0.52315

# 2. Loss
loss = 0.229  # Mejoró un poco!

# 3. Backward
loss.backward()
# Gradiente: ∂loss/∂W = -0.12

# 4. Optimizer step
W_new = 0.52315 - 0.001 * (-0.12)
W_new = 0.52315 + 0.00012
W_new = 0.52327  ← PESO ACTUALIZADO OTRA VEZ
```

### Después de la época:
```python
scheduler.step()
# lr = 0.001 → 0.000999
# W sigue siendo 0.52327 (no cambia)
```

---

## Código Completo Anotado

```python
# INICIALIZACIÓN (1 vez)
model = UNet()                    # W = [random values]
optimizer = AdamW(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=500)

# ENTRENAMIENTO (500 épocas)
for epoch in range(500):
    # ÉPOCA COMPLETA
    for batch in train_loader:
        # ═══════════════════════════════════════════
        # ACTUALIZACIÓN DE PESOS (cada batch)
        # ═══════════════════════════════════════════
        optimizer.zero_grad()      # Gradientes = 0
        output = model(input)      # Forward
        loss = loss_fn(output, gt) # Calcular error
        loss.backward()            # Calcular gradientes
        optimizer.step()           # ← ACTUALIZAR PESOS
        # ═══════════════════════════════════════════
    
    # VALIDACIÓN (sin actualizar pesos)
    with torch.no_grad():
        val_loss = validate(...)
    
    # ═══════════════════════════════════════════
    # ACTUALIZACIÓN DE LEARNING RATE (cada época)
    # ═══════════════════════════════════════════
    scheduler.step()  # ← ACTUALIZAR LR (no pesos)
    # ═══════════════════════════════════════════
```

---

## Resumen Final

### `optimizer.step()`:
- ✅ **Actualiza los pesos del modelo**
- ✅ Se llama **cada batch** (muchas veces por época)
- ✅ Usa los gradientes calculados por `loss.backward()`
- ✅ Aplica el algoritmo de optimización (Adam, SGD, etc.)
- ✅ **Aquí es donde el modelo aprende**

### `scheduler.step()`:
- ❌ **NO actualiza los pesos**
- ✅ Actualiza el **learning rate**
- ✅ Se llama **cada época** (una vez por época)
- ✅ Controla qué tan grandes son los pasos de actualización
- ✅ Ayuda a la convergencia (LR alto al inicio, bajo al final)

---

## Analogía

Imagina que estás caminando hacia una montaña (mínimo de la loss):

- **`optimizer.step()`**: Dar un paso físico hacia la montaña
  - Cada batch = un paso
  - Los pesos = tu posición actual
  
- **`scheduler.step()`**: Ajustar el tamaño de tus pasos
  - Al inicio: pasos grandes (lr alto) para avanzar rápido
  - Al final: pasos pequeños (lr bajo) para no pasarte del mínimo
  
**No confundir:**
- Dar pasos (optimizer.step) ≠ Ajustar tamaño de pasos (scheduler.step)
- Los pesos cambian con optimizer.step
- El LR cambia con scheduler.step
