#!/bin/bash

# Nome da versão Python desejada
PYTHON_VERSION="3.8.18"

# Função para checar se o ambiente virtual já está ativo
function is_venv_active() {
    [[ "$VIRTUAL_ENV" != "" ]]
}

# Instala o Python se necessário
echo "Verificando instalação do Python $PYTHON_VERSION com pyenv..."
pyenv install -s $PYTHON_VERSION
pyenv local $PYTHON_VERSION

# Cria o ambiente virtual se ele não existir
if [ ! -d "venv" ]; then
    echo "Criando ambiente virtual..."
    python -m venv venv
else
    echo "Ambiente virtual 'venv' já existe."
fi

# Ativa o ambiente virtual apenas se não estiver ativo
if ! is_venv_active; then
    echo "Ativando ambiente virtual..."
    source venv/bin/activate
else
    echo "Ambiente virtual já está ativo."
fi

# Atualiza pip e setuptools
echo "Atualizando pip e setuptools..."
pip install --upgrade pip setuptools

# Instala dependências
echo "Instalando bibliotecas necessárias..."
pip install gym-retro==0.8.0 opencv-python numpy

echo "Ambiente configurado com sucesso!"
python --version

#Após rodar esse script ativar o ambiente com $source venv/bin/activate
#./venv/bin/retro-import ./roms
# retro-import ~/retro-roms
# python -m retro.import roms
