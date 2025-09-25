# Tests para el proyecto CompCars

Este directorio contiene tests unitarios para validar la funcionalidad
de los componentes principales del proyecto.

## Estructura

- `test_imports.py`: Validación de importación de módulos
- `test_config.py`: Tests de configuración
- `test_data.py`: Tests de dataset y procesamiento
- `test_models.py`: Tests básicos de modelos
- `test_utils.py`: Tests de utilidades

## Ejecutar tests

```bash
# Todos los tests
pytest tests/

# Test específico
pytest tests/test_imports.py -v

# Con coverage
pytest tests/ --cov=src --cov-report=html
```