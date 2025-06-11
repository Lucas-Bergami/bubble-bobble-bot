# Bubble Bobble Repulsion Bot

Este projeto implementa um agente que joga *Bubble Bobble (NES)* automaticamente utilizando movimentação baseada em repulsão de inimigos.

## 🎮 Descrição

O agente detecta o personagem e os inimigos na tela usando visão computacional. Com base nas posições detectadas, ele calcula vetores de repulsão para movimentar o personagem de forma inteligente. Caso o personagem não seja detectado ou esteja parado, o agente realiza movimentos aleatórios.

## 🧰 Requisitos

- Python 3.8+
- `setup.sh` (fornecido) instala todas as dependências automaticamente.

## 📦 Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/repositorio-novo.git
   cd repositorio-novo
   ```
Torne o script executável e rode o setup:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
Ative o ambiente virtual:

No Linux/macOS:
   ```bash
    source .venv/bin/activate
   ```
🎮 Importando a ROM para o Gym-Retro
Obtenha o arquivo .nes correspondente à ROM do Bubble Bobble (NES).

⚠️ A ROM não é fornecida por este projeto por motivos legais.

Importe a ROM para o gym-retro:

   ```bash
   python -m retro.import /caminho/para/a/pasta/contendo/a/rom/
   ```
🚀 Execução
Com o ambiente ativado, execute o agente:
   ```bash
   python3 emulador.py
   ```
ou 
   ```bash
   python3 jogavel.py
   ```

Pressione ESC a qualquer momento para sair.


📌 Observações
Certifique-se de que a ROM usada corresponde ao nome reconhecido: BubbleBobble-Nes.

O jogo deve rodar com as cores originais para que a detecção funcione corretamente.
