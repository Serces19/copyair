# Learning Rate Schedulers - Guía de Uso

## Resumen

Se ha implementado un sistema flexible de learning rate schedulers configurables desde `params.yaml`. Ahora puedes elegir entre 7 tipos diferentes de schedulers, incluyendo **constant LR** (sin scheduler).

## Schedulers Disponibles

### 1. **Constant LR** (Recomendado para empezar)
Mantiene el learning rate constante durante todo el entrenamiento.

```yaml
scheduler:
  type: "constant"
```

**Cuándo usar:**
- ✅ Cuando quieres control total sobre el LR
- ✅ Para experimentos iniciales
- ✅ Cuando el modelo ya converge bien sin scheduler

---

### 2. **Cosine Annealing**
Reduce el LR siguiendo una curva coseno suave.

```yaml
scheduler:
  type: "cosine"
  params:
    T_max: 500      # Número total de épocas
    eta_min: 0      # LR mínimo al final
```

**Cuándo usar:**
- ✅ Entrenamiento largo y estable
- ✅ Cuando quieres reducción suave del LR
- ✅ Estándar en muchos papers

**Fórmula:** `lr = eta_min + (lr_inicial - eta_min) * (1 + cos(π * epoch / T_max)) / 2`

---

### 3. **Step LR**
Reduce el LR en pasos discretos cada N épocas.

```yaml
scheduler:
  type: "step"
  params:
    step_size: 100  # Reducir cada 100 épocas
    gamma: 0.5      # Multiplicar LR por 0.5
```

**Cuándo usar:**
- ✅ Cuando sabes en qué épocas reducir el LR
- ✅ Para fine-tuning con reducciones agresivas
- ✅ Entrenamiento en etapas

**Ejemplo:** LR=0.001 → época 100: 0.0005 → época 200: 0.00025

---

### 4. **Exponential LR**
Decaimiento exponencial suave del LR cada época.

```yaml
scheduler:
  type: "exponential"
  params:
    gamma: 0.98  # Multiplicar LR por 0.98 cada época
```

**Cuándo usar:**
- ✅ Reducción muy gradual del LR
- ✅ Entrenamiento muy largo (>1000 épocas)
- ✅ Cuando quieres decay constante

**Fórmula:** `lr = lr_inicial * gamma^epoch`

---

### 5. **Reduce on Plateau**
Reduce el LR automáticamente cuando la métrica de validación se estanca.

```yaml
scheduler:
  type: "plateau"
  params:
    mode: "min"      # Minimizar val_loss
    factor: 0.5      # Reducir LR a la mitad
    patience: 10     # Esperar 10 épocas sin mejora
    verbose: true
```

**Cuándo usar:**
- ✅ **MUY RECOMENDADO** para entrenamiento adaptativo
- ✅ Cuando no sabes cuándo reducir el LR
- ✅ Para maximizar convergencia

**⚠️ Nota:** Este scheduler requiere llamar a `scheduler.step(val_loss)` con la métrica de validación.

---

### 6. **OneCycle LR**
Estrategia de "super-convergence": sube el LR al inicio, luego lo baja.

```yaml
scheduler:
  type: "onecycle"
  params:
    max_lr: 0.01  # LR máximo (opcional, usa learning_rate si no se especifica)
```

**Cuándo usar:**
- ✅ Entrenamiento rápido (pocas épocas)
- ✅ Para encontrar el mejor LR rápidamente
- ✅ Cuando quieres convergencia en <100 épocas

**Fases:**
1. Warmup: LR sube de 0 a max_lr (45% del entrenamiento)
2. Annealing: LR baja de max_lr a 0 (55% del entrenamiento)

---

### 7. **Cosine Annealing with Warm Restarts**
Cosine annealing que se "reinicia" periódicamente.

```yaml
scheduler:
  type: "cosine_warmup"
  params:
    T_0: 50        # Primer restart después de 50 épocas
    T_mult: 2      # Duplicar periodo en cada restart
    eta_min: 0
```

**Cuándo usar:**
- ✅ Para escapar de mínimos locales
- ✅ Entrenamiento muy largo
- ✅ Cuando quieres explorar múltiples soluciones

**Ejemplo:** Restart en épocas 50, 150, 350, 750...

---

## Recomendaciones por Caso de Uso

### 🎯 Experimentos Iniciales
```yaml
scheduler:
  type: "constant"
```
Mantén el LR constante para entender el comportamiento base.

### 🎯 Entrenamiento Estándar (500-1000 épocas)
```yaml
scheduler:
  type: "cosine"
  params:
    T_max: 500
    eta_min: 0
```

### 🎯 Entrenamiento Adaptativo (Recomendado)
```yaml
scheduler:
  type: "plateau"
  params:
    mode: "min"
    factor: 0.5
    patience: 20
    verbose: true
```

### 🎯 Entrenamiento Rápido (<100 épocas)
```yaml
scheduler:
  type: "onecycle"
  params:
    max_lr: 0.01
```

### 🎯 Fine-tuning
```yaml
scheduler:
  type: "step"
  params:
    step_size: 50
    gamma: 0.5
```

---

## Implementación Técnica

### Archivo: `src/training/schedulers.py`
Factory que crea schedulers basado en configuración.

### Uso en código:
```python
from src.training.schedulers import get_scheduler

scheduler_config = {
    'type': 'constant',
    'params': {}
}
scheduler = get_scheduler(optimizer, scheduler_config)

# En el loop de entrenamiento
for epoch in range(epochs):
    train(...)
    val_loss = validate(...)
    
    # Para la mayoría de schedulers
    scheduler.step()
    
    # Para ReduceLROnPlateau
    # scheduler.step(val_loss)
```

---

## Visualización del LR

Para ver cómo cambia el LR durante el entrenamiento:

```python
current_lr = scheduler.get_last_lr()[0]
print(f"Época {epoch}, LR: {current_lr}")

# Con MLflow (automático)
mlflow.log_metric('train/lr', current_lr, step=epoch)
```

---

## Comparación Visual

```
Constant:     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cosine:       ━━━━━━━━━━━━━━━━╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲
Step:         ━━━━━━━━━━┓        ┗━━━━━━━┓        ┗━━━━
Exponential:  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OneCycle:     ╱╱╱╱╱╱╱╱╱╱╱╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲
Plateau:      ━━━━━━━━━━━━━━━━┓     ┗━━━━━━━━━━━━━━━━
Cosine+Warm:  ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲
```

---

## Testing

Para probar un scheduler:

```bash
# Editar configs/params.yaml
# Cambiar scheduler.type a "constant", "cosine", etc.

python scripts/train.py --config configs/params.yaml
```

El LR se loggea automáticamente en MLflow como `train/lr`.

---

## Troubleshooting

### Problema: "Scheduler no soportado"
**Solución:** Verifica que el `type` sea uno de: `constant`, `cosine`, `step`, `exponential`, `plateau`, `onecycle`, `cosine_warmup`

### Problema: OneCycleLR no funciona
**Solución:** OneCycleLR necesita `steps_per_epoch`. Esto se calcula automáticamente en `train.py`.

### Problema: ReduceLROnPlateau no reduce el LR
**Solución:** Asegúrate de llamar `scheduler.step(val_loss)` en lugar de `scheduler.step()`.

---

## Migración desde Código Antiguo

**Antes:**
```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
```

**Ahora:**
```yaml
# En params.yaml
scheduler:
  type: "cosine"
  params:
    T_max: 500
    eta_min: 0
```

```python
# En train.py
scheduler = get_scheduler(optimizer, config['training']['scheduler'])
```

---

## Referencias

- [PyTorch LR Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [Super-Convergence (OneCycle)](https://arxiv.org/abs/1708.07120)
- [SGDR (Cosine with Warm Restarts)](https://arxiv.org/abs/1608.03983)
