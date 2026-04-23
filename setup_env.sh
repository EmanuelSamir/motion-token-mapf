#!/bin/bash

# Este script debe ser ejecutado con 'source setup_env.sh'
ENV_NAME=".env"

echo "🚀 Iniciando configuración para Motion-Token-MAPF..."

# 1. Verificar si python3.12 existe
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Error: python3.12 no está instalado en este sistema."
    echo "Por favor instálalo con: brew install python@3.12"
    # No usamos exit para no cerrar la terminal del usuario si usó source
    return 1 2>/dev/null || exit 1
fi

# 2. Crear entorno virtual si no existe
if [ ! -d "$ENV_NAME" ]; then
    echo "📦 Creando entorno virtual .env con Python 3.12..."
    python3.12 -m venv "$ENV_NAME"
fi

# 3. Actualizar pip e instalar dependencias
echo "⬆️  Actualizando pip..."
./"$ENV_NAME"/bin/pip install --upgrade pip

echo "🛠️  Instalando dependencias (esto puede tardar)..."
./"$ENV_NAME"/bin/pip install -r requirements.txt

# 4. Activar el entorno para el usuario
echo "✅ Entorno configurado correctamente."
source "$ENV_NAME"/bin/activate
echo "🌟 Entorno '$ENV_NAME' activado. ¡Listo para trabajar!"

# Limpiar variables si se usó source
return 0 2>/dev/null
