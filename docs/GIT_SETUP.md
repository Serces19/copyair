# Iniciar Git

Este archivo contiene los comandos para versionar tu proyecto con Git.

## ¿Por qué usar Git?

- 📝 Mantener historial de cambios
- 🔄 Volver a versiones anteriores
- 👥 Colaborar con otros
- ☁️ Subir a GitHub/GitLab
- 🚀 CI/CD automatizado

## Comandos Iniciales

```bash
# 1. Inicializar repositorio
cd k:\Trabajos\Code\copyair
git init

# 2. Ver estado
git status

# 3. Agregar todos los archivos
git add .

# 4. Primer commit
git commit -m "🎉 Initial commit: Proyecto CopyAir modularizado

- Arquitectura U-Net profesional
- Módulos de datos, modelos, entrenamiento
- Configuración centralizada (YAML)
- Tests unitarios
- Docker + CI/CD ready"

# 5. (Opcional) Crear rama main
git branch -M main
```

## Subir a GitHub

```bash
# 1. Crear repositorio en GitHub (vacío, sin README)

# 2. Agregar remoto
git remote add origin https://github.com/usuario/copyair.git

# 3. Subir código
git push -u origin main

# 4. Verificar
git remote -v
```

## Workflow Diario

```bash
# Ver cambios
git status
git diff

# Hacer commit
git add .
git commit -m "Descripción clara del cambio"

# Subir
git push

# Traer cambios
git pull
```

## Plantilla de Commits

```
[TIPO] Descripción breve (50 caracteres)

Descripción detallada (si es necesario)

- Cambio 1
- Cambio 2

Relacionado: #123
```

Tipos:
- 🎉 **feat**: Nueva característica
- 🐛 **fix**: Corrección de bug
- 📚 **docs**: Documentación
- 🎨 **style**: Formato
- ♻️ **refactor**: Refactorización
- ⚡ **perf**: Mejora de rendimiento
- 🧪 **test**: Pruebas

## Ejemplo

```bash
git add src/models/unet.py
git commit -m "🎨 Optimizar U-Net para menor uso de memoria

- Reducir base_channels default de 64 a 32
- Implementar gradient checkpointing
- Mejora: 20% menos VRAM

Tests: ✅ All passing"

git push
```

## .gitignore ya incluido

El archivo `.gitignore` ya excluye:
- `__pycache__/`
- `venv/`
- `*.pyc`
- `data/` (usa DVC)
- `models/` (usa DVC)
- `.DS_Store`

## Ramas (Branching)

```bash
# Crear rama para feature
git checkout -b feature/agregar-mlflow

# Hacer commits
git commit -m "..."

# Subir
git push -u origin feature/agregar-mlflow

# Crear Pull Request en GitHub
# Después merguear a main
```

## Próximo Paso: DVC (versionado de datos)

```bash
pip install dvc

dvc init
dvc add data/03_processed
git add data/03_processed.dvc .gitignore
git commit -m "🗂️ Versionado de datos con DVC"
```

---

¡Tu proyecto está listo para colaboración! 🚀
