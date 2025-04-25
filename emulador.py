import retro
import cv2
import numpy as np
import time

def captura_tela(obs):
    return obs.copy()

def detectar_objetos_vermelhos(tela):
    hsv = cv2.cvtColor(tela, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contornos, mask

env = retro.make(game='Breakout-Atari2600')
obs = env.reset()

while True:
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == 27:
        break

    action = [0] * 8
    altura, largura, _ = obs.shape

    # ===============================
    # CONFIGURA AQUI as áreas de detecção
    # ===============================
    altura_topo = 31           # pixels do topo da tela a ignorar
    altura_barra = 22          # pixels do final da tela para detectar a barra
    margem_lateral = 8         # pixels dos lados esquerdo e direito a ignorar
    # ===============================

    altura_bola = altura - altura_barra
    tela = captura_tela(obs)

    roi_barra = tela[altura - altura_barra:, margem_lateral:largura - margem_lateral]
    roi_bola = tela[altura_topo:altura_bola, margem_lateral:largura - margem_lateral]

    # --- Detecta barra ---
    barra_x = None
    contornos_barra, _ = detectar_objetos_vermelhos(roi_barra)
    for contorno in contornos_barra:
        x, y, w, h = cv2.boundingRect(contorno)
        x_corrigido = x + margem_lateral
        y_corrigido = y + altura - altura_barra
        barra_x = x_corrigido + w // 2
        cv2.rectangle(tela, (x_corrigido, y_corrigido), (x_corrigido + w, y_corrigido + h), (255, 0, 0), 1)

    # --- Detecta bola ---
    bola_x = None
    contornos_bola, _ = detectar_objetos_vermelhos(roi_bola)
    for contorno in contornos_bola:
        x, y, w, h = cv2.boundingRect(contorno)
        if 1 <= w <= 3 and 3 <= h <= 5:
            x_corrigido = x + margem_lateral
            y_corrigido = y + altura_topo
            bola_x = x_corrigido + w // 2
            cv2.rectangle(tela, (x_corrigido, y_corrigido), (x_corrigido + w, y_corrigido + h), (0, 255, 0), 1)
            break

    # --- Lógica de movimentação da barra ---
    if bola_x is not None and barra_x is not None:
        if barra_x < bola_x - 2:
            action[7] = 1  # direita
        elif barra_x > bola_x + 2:
            action[6] = 1  # esquerda
    elif bola_x is None:
        action[0] = 1  # dispara bola se não houver nenhuma

    # Moldura da área útil de detecção
    cv2.rectangle(
        tela,
        (margem_lateral, altura_topo),
        (largura - margem_lateral, altura_bola),
        (255, 255, 0),
        1
    )

    tela_maior = cv2.resize(tela, (largura * 3, altura * 3), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Tela do jogo", tela_maior)

    obs, reward, done, info = env.step(action)

    if done:
        obs = env.reset()

    time.sleep(1 / 60)

cv2.destroyAllWindows()
env.close()
